using System.Collections.Generic;
using UnityEngine;

namespace BalloonSim.Sim
{
    public class AutopilotController : MonoBehaviour
    {
        public SimulationConfig config;
        public BalloonState balloon;
        public TurbulenceField field;
        public ObservationBuffer observationBuffer;
        public ONNXPredictor onnxPredictor;
        public BalloonThermodynamics thermodynamics;

        private readonly Queue<Vector3> _hist = new();
        private Vector3 _safe;
        private Vector3 _safeForward;
        private Vector3 _safeVel;
        private float _holdIy;
        private bool _wasUser;
        private Vector3 _holdAnchor;
        private Vector3 _holdForwardAnchor;

        public Vector3 Predicted { get; private set; }
        public float PredErr { get; private set; }
        public float PredMSE { get; private set; }
        public float PredSigma { get; private set; }
        public float ThrottleNeed { get; private set; }
        public Vector3 LastCommandVel { get; private set; }
        public Vector3 HoldForwardTarget => _safeForward;

        public void ResetState()
        {
            _hist.Clear();
            _safe = balloon.transform.position;
            _holdAnchor = _safe;
            _safeForward = balloon.transform.forward;
            _holdForwardAnchor = _safeForward;
            _safeVel = balloon.velocity;
            _holdIy = 0f;
            _wasUser = false;
            Predicted = Vector3.zero;
            LastCommandVel = Vector3.zero;
            PredErr = 0f;
            PredMSE = 0f;
            PredSigma = 1f;
            ThrottleNeed = 0f;
        }

        public Vector3 Predict()
        {
            if (onnxPredictor != null && observationBuffer != null && config.enableAIPredictor)
            {
                var histFrames = observationBuffer.Snapshot();
                if (onnxPredictor.TryPredict(histFrames, out var mu, out var sigma))
                {
                    PredSigma = sigma;
                    return mu;
                }
            }

            if (_hist.Count < 3)
            {
                PredSigma = 1f;
                PredMSE = 0f;
                return Vector3.zero;
            }

            Vector3[] arr = new Vector3[_hist.Count];
            _hist.CopyTo(arr, 0);

            float[] w = { 0.08f, 0.14f, 0.2f, 0.28f, 0.4f };
            int n = arr.Length;
            int s = Mathf.Max(0, n - 5);

            Vector3 p = Vector3.zero;
            float ws = 0f;
            for (int i = s; i < n; i++)
            {
                float wi = w[Mathf.Min(i - s, 4)];
                p += arr[i] * wi;
                ws += wi;
            }

            if (ws > 1e-6f) p /= ws;
            if (n > 1) p += (arr[n - 1] - arr[n - 2]) * 0.28f;

            float var = 0f;
            for (int i = s; i < n; i++)
            {
                Vector3 d = arr[i] - p;
                var += d.sqrMagnitude;
            }
            var /= Mathf.Max(1, n - s);
            PredSigma = Mathf.Sqrt(var + 1e-6f);

            return p;
        }

        public Vector3 ComputeControl(Vector3 userTargetVel, bool userActive, float dt)
        {
            if (_wasUser && !userActive)
            {
                _safe = balloon.transform.position;
                _holdAnchor = _safe;
                _safeForward = balloon.transform.forward;
                _holdForwardAnchor = _safeForward;
                _safeVel = balloon.velocity;
                _holdIy = 0f;
            }
            else if (userActive)
            {
                _holdAnchor = balloon.transform.position;
                _holdForwardAnchor = balloon.transform.forward;
            }
            _wasUser = userActive;

            Vector3 u = field.Sample(balloon.transform.position);
            Vector3 p = Predict();
            Predicted = p;

            float buoy = thermodynamics != null
                ? thermodynamics.GetBuoyancyAcceleration()
                : (1f - config.densityRatio) * config.buoyK;

            float fl = 0.95f;
            Vector3 envMeasured = new Vector3(u.x * fl, u.y * fl + buoy, u.z * fl);
            Vector3 baseVel = userTargetVel;

            if (!userActive)
            {
                Vector3 toSafe = _holdAnchor - balloon.transform.position;

                _holdIy = Mathf.Clamp(_holdIy + toSafe.y * dt, -6f, 6f);

                // History-based hold target: only historical release state + current state.
                Vector3 holdVel =
                    toSafe * config.holdPosK
                    - balloon.velocity * config.holdVelK
                    + _holdForwardAnchor * config.holdForwardK
                    + _safeVel * config.holdReleaseVelK;

                holdVel.y += _holdIy * config.holdAltIK;

                if (thermodynamics != null)
                {
                    float buoyComp = Mathf.Clamp(buoy, 0f, 3f);
                    holdVel.y += buoyComp;
                }

                holdVel = Vector3.ClampMagnitude(holdVel, config.holdMaxSpeed);

                if (!config.aiEnabled)
                {
                    ThrottleNeed = holdVel.magnitude;
                    PredErr = 0f;
                    PredMSE = (p - u).sqrMagnitude / 3f;
                    PushHistory(u);
                    LastCommandVel = holdVel;
                    return holdVel;
                }

                if (config.strictHoldNoDrift)
                {
                    ThrottleNeed = holdVel.magnitude;
                    PredErr = (p - u).magnitude / (u.magnitude + 1e-6f);
                    PredMSE = (p - u).sqrMagnitude / 3f;
                    PushHistory(u);
                    LastCommandVel = holdVel;
                    return holdVel;
                }

                Vector3 assistHold = holdVel;
                if (config.holdEnvComp > 0f)
                {
                    Vector3 envEstimated = new Vector3(p.x * fl, p.y * fl + buoy, p.z * fl);
                    Vector3 envForComp = config.assistUseOracleEnvCancellation ? envMeasured : envEstimated;
                    float compGain = Mathf.Clamp01(config.holdEnvComp);
                    if (!config.assistUseOracleEnvCancellation && config.usePredSigmaConfidence)
                        compGain *= Mathf.Lerp(1f, 0.35f, Mathf.Clamp01(PredSigma / 2.0f));
                    assistHold -= envForComp * compGain;
                }
                assistHold = Vector3.ClampMagnitude(assistHold, config.holdMaxSpeed);

                ThrottleNeed = assistHold.magnitude;
                PredErr = (p - u).magnitude / (u.magnitude + 1e-6f);
                PredMSE = (p - u).sqrMagnitude / 3f;
                PushHistory(u);
                LastCommandVel = assistHold;
                return assistHold;
            }

            if (!config.aiEnabled)
            {
                PushHistory(u);
                ThrottleNeed = 0f;
                PredErr = 0f;
                PredMSE = (p - u).sqrMagnitude / 3f;
                LastCommandVel = baseVel;
                return baseVel;
            }

            Vector3 toSafe2 = _safe - balloon.transform.position;
            _holdIy = 0f;
            float dist = toSafe2.magnitude;
            Vector3 safeDir = dist > 1e-6f ? toSafe2 / dist : Vector3.zero;

            Vector3 want = userTargetVel;
            float wn = want.magnitude;
            Vector3 wdir = wn > 1e-6f ? want / wn : safeDir;

            float distTerm = Mathf.Clamp(dist * config.aiSafeDistK, 0f, 2.5f);
            float flowAlong = Vector3.Dot(new Vector3(p.x * fl, p.y * fl + buoy, p.z * fl), wdir);
            float velAlong = Vector3.Dot(balloon.velocity, wdir);

            float need = Mathf.Clamp(wn + distTerm - flowAlong - velAlong * config.aiSafeVelK, 0f,
                config.throttle * config.throttle * config.aiNeedK);

            if (config.usePredSigmaConfidence)
            {
                // Keep uncertainty as a soft attenuation, not a hard suppression.
                float uncertGain = Mathf.Lerp(1f, 0.75f, Mathf.Clamp01(PredSigma / 2.0f));
                need *= uncertGain;
            }

            ThrottleNeed = need;
            Vector3 assist = wdir * need;

            PredErr = (p - u).magnitude / (u.magnitude + 1e-6f);
            PredMSE = (p - u).sqrMagnitude / 3f;
            PushHistory(u);
            Vector3 command = baseVel + assist;
            LastCommandVel = command;
            return command;
        }

        private void PushHistory(Vector3 u)
        {
            _hist.Enqueue(u);
            while (_hist.Count > 12) _hist.Dequeue();
        }
    }
}
