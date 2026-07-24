using UnityEngine;

namespace BalloonSim.Sim
{
    public class GameController : MonoBehaviour
    {
        [Header("Refs")]
        public SimulationConfig config;
        public TurbulenceField field;
        public BalloonState balloon;
        public AutopilotController autopilot;
        public Stage3PolicyRunner stage3Policy;
        public BlockWorld world;
        public ObservationBuffer observationBuffer;

        [Header("Objects")]
        [SerializeField] private Transform target;
        [SerializeField] private Transform boxVisual;

        [Header("Input")]
        [Tooltip("If set, WASD/J/K directions are interpreted in this frame.")]
        [SerializeField] private Transform _inputFrame;
        public Transform inputFrame { get => _inputFrame; set => _inputFrame = value; }

        public Vector3 UserTargetVel { get; private set; }
        public bool UserActive { get; private set; }
        public bool explorationOverride = false;
        public Vector3 PlayerIntentVel { get; private set; }
        public bool PlayerIntentActive { get; private set; }
        public Vector3 LastDesiredVel { get; private set; }
        public Vector3 PreviousExpertCommand { get; private set; }
        public Vector3 LastExpertCommand { get; private set; }
        public Vector3 LastAcceleration { get; private set; }

        [Header("Stage3 Expert")]
        [Tooltip("Velocity-error scale where tracking residual approaches its smooth bound.")]
        public float expertVelocityErrorScale = 2f;
        public float expertTrackingResidualMax = 0.12f;
        [Tooltip("Native turbulence-field magnitude used to normalize wind correction.")]
        public float expertWindScale = 50f;
        public float expertLateralWindResidualMax = 0.16f;
        public float expertAlongWindResidualMax = 0.06f;
        [Tooltip("Maximum player intent magnitude presented to the Stage3 policy. Must match collection speed.")]
        public float stage3IntentMaxSpeed = 0.5f;
        [Tooltip("Final residual authority used identically by expert labels and runtime.")]
        public float stage3ResidualMaxSpeed = 0.25f;
        [Range(0f, 1f)] public float stage3ResidualBlend = 1f;
        [Tooltip("Maximum residual change per second before blending.")]
        public float stage3ResidualSlewRate = 2.0f;
        [Tooltip("Residual smoothing responsiveness in 1/seconds.")]
        public float stage3ResidualResponsiveness = 6.0f;
        [Range(0f, 1f)] public float stage3MinForwardIntentFraction = 0.65f;

        private Vector3 _smoothedStage3Residual;

        private void Awake()
        {
            if (config == null)
                config = ScriptableObject.CreateInstance<SimulationConfig>();

            if (field != null)
            {
                field.config = config;
                field.Initialize();
            }

            if (autopilot != null)
            {
                autopilot.config = config;
                autopilot.ResetState();
            }

            ApplyConfigToVisuals();
        }

        private void ApplyConfigToVisuals()
        {
            if (boxVisual != null)
            {
                boxVisual.localScale = Vector3.one * config.boxSize;
                boxVisual.position = new Vector3(config.boxSize, config.boxSize, config.boxSize) * 0.5f;
            }
        }

        public void ResetAll()
        {
            if (balloon != null)
            {
                balloon.transform.position = new Vector3(8f, 8f, 8f);
                balloon.velocity = Vector3.zero;
                LastDesiredVel = Vector3.zero;
                LastAcceleration = Vector3.zero;
            }

            if (field != null)
                field.Generate();

            if (autopilot != null)
                autopilot.ResetState();

            if (world != null)
                world.ResetWorld();
        }

        public void ToggleAI(bool enabled) => config.aiEnabled = enabled;
        public void ToggleTornado() => config.tornado = !config.tornado;

        public void SetUserTarget(Vector3 targetVel, bool active = true)
        {
            PlayerIntentVel = targetVel;
            PlayerIntentActive = active;
            UserTargetVel = targetVel;
            UserActive = active;
            LastDesiredVel = targetVel;
        }

        public void SetExplorationTarget(Vector3 targetVel, bool active = true)
        {
            if (!explorationOverride) return;
            PlayerIntentVel = targetVel;
            PlayerIntentActive = active;
            UserTargetVel = targetVel;
            UserActive = active;
            LastDesiredVel = targetVel;
        }

        public Vector3 ComputeExpertCommand(Vector3 intentVel, bool intentActive)
        {
            Vector3 target = intentActive ? intentVel : Vector3.zero;
            return target + ComputeExpertResidual(target, intentActive);
        }

        public Vector3 ComputeExpertResidual(Vector3 intentVel, bool intentActive)
        {
            if (!intentActive || intentVel.sqrMagnitude < 1e-8f) return Vector3.zero;

            Vector3 currentVel = balloon != null ? balloon.velocity : Vector3.zero;
            Vector3 wind = field != null && balloon != null ? field.Sample(balloon.transform.position) : Vector3.zero;
            Vector3 intentDir = intentVel.normalized;

            Vector3 velocityError = intentVel - currentVel;
            Vector3 trackingResidual = SmoothBound(
                velocityError,
                Mathf.Max(0.01f, expertVelocityErrorScale),
                Mathf.Max(0f, expertTrackingResidualMax));

            Vector3 windAlong = Vector3.Project(wind, intentDir);
            Vector3 windLateral = wind - windAlong;
            float windScale = Mathf.Max(0.01f, expertWindScale);
            Vector3 windResidual =
                -SmoothBound(windLateral, windScale, Mathf.Max(0f, expertLateralWindResidualMax))
                -SmoothBound(windAlong, windScale, Mathf.Max(0f, expertAlongWindResidualMax));

            return Vector3.ClampMagnitude(
                trackingResidual + windResidual,
                Mathf.Max(0.01f, stage3ResidualMaxSpeed));
        }

        private static Vector3 SmoothBound(Vector3 value, float scale, float maxMagnitude)
        {
            if (maxMagnitude <= 0f || value.sqrMagnitude < 1e-12f) return Vector3.zero;
            float magnitude = value.magnitude;
            float boundedMagnitude = maxMagnitude * (float)System.Math.Tanh(magnitude / Mathf.Max(0.01f, scale));
            return value * (boundedMagnitude / magnitude);
        }

        private void Update()
        {
            ReadInput();
        }

        private void FixedUpdate()
        {
            if (config == null || balloon == null) return;

            float dt = Time.fixedDeltaTime;

            if (world != null)
                world.UpdateCurrentBlock(balloon.transform.position);

            if (field != null)
                field.Step(dt, balloon.transform.position, balloon.velocity);

            observationBuffer?.Push(balloon.transform.position, balloon.velocity, field);

            Vector3 envMeasured = field != null ? field.Sample(balloon.transform.position) : Vector3.zero;
            Vector3 desiredVel;
            PreviousExpertCommand = LastExpertCommand;
            LastExpertCommand = ComputeExpertCommand(PlayerIntentVel, PlayerIntentActive);
            if (explorationOverride)
            {
                desiredVel = LastExpertCommand + envMeasured;
            }
            else if (config != null && config.controlMode == ControlMode.TrueManual)
            {
                desiredVel = UserTargetVel + envMeasured;
            }
            else
            {
                desiredVel = autopilot != null
                    ? autopilot.ComputeControl(UserTargetVel, UserActive, dt)
                    : UserTargetVel + envMeasured;
            }

            LastDesiredVel = UserTargetVel;
            Vector3 prevVel = balloon.velocity;
            balloon.velocity = Vector3.Lerp(balloon.velocity, desiredVel, Mathf.Clamp01(dt * 5f));
            balloon.velocity = Vector3.ClampMagnitude(balloon.velocity, 4f);
            LastAcceleration = dt > 1e-6f ? (balloon.velocity - prevVel) / dt : Vector3.zero;

            UpdateAttitude(dt, desiredVel);

            Vector3 pos = balloon.transform.position + balloon.velocity * dt;

            if (world != null)
                pos = world.ClampToAllowed(pos);
            else
            {
                float min = config.BoxMin;
                float max = config.BoxMax;
                pos.x = Mathf.Clamp(pos.x, min, max);
                pos.y = Mathf.Clamp(pos.y, min, max);
                pos.z = Mathf.Clamp(pos.z, min, max);
            }

            balloon.transform.position = pos;
        }

        private void ReadInput()
        {
            Vector3 raw = Vector3.zero;

            if (Input.GetKey(KeyCode.W)) raw.z += 1f;
            if (Input.GetKey(KeyCode.S)) raw.z -= 1f;
            if (Input.GetKey(KeyCode.A)) raw.x -= 1f;
            if (Input.GetKey(KeyCode.D)) raw.x += 1f;
            if (Input.GetKey(KeyCode.K)) raw.y += 1f;
            if (Input.GetKey(KeyCode.J)) raw.y -= 1f;

            float spd = config.throttle * config.throttle * config.manualSpeedScale;

            bool hasManualInput = raw.sqrMagnitude > 1e-6f;
            if (!explorationOverride)
            {
                if (hasManualInput)
                {
                    Vector3 localDir = raw.normalized;
                    Transform frame = inputFrame != null ? inputFrame : (Camera.main != null ? Camera.main.transform : null);
                    Vector3 frameForward = frame != null ? Vector3.ProjectOnPlane(frame.forward, Vector3.up).normalized : Vector3.forward;
                    Vector3 frameRight = frame != null ? Vector3.ProjectOnPlane(frame.right, Vector3.up).normalized : Vector3.right;
                    if (frameForward.sqrMagnitude < 1e-6f) frameForward = Vector3.forward;
                    if (frameRight.sqrMagnitude < 1e-6f) frameRight = Vector3.right;
                    Vector3 worldDir = frameRight * localDir.x + Vector3.up * localDir.y + frameForward * localDir.z;
                    float intentSpeed = config != null && config.controlMode == ControlMode.Stage3Policy
                        ? Mathf.Min(spd, Mathf.Max(0.05f, stage3IntentMaxSpeed))
                        : spd;
                    PlayerIntentVel = worldDir.normalized * intentSpeed;
                    PlayerIntentActive = true;
                }
                else
                {
                    PlayerIntentVel = Vector3.zero;
                    PlayerIntentActive = false;
                }

                UserTargetVel = PlayerIntentVel;
                UserActive = PlayerIntentActive;
                if (PlayerIntentActive && config != null && config.aiEnabled && stage3Policy != null && stage3Policy.TryPredictAction(out var predictedResidual))
                {
                    Vector3 intentDir = PlayerIntentVel.normalized;
                    Vector3 residual = Vector3.ClampMagnitude(predictedResidual, Mathf.Max(0.05f, stage3ResidualMaxSpeed));
                    float minAlong = -PlayerIntentVel.magnitude * (1f - Mathf.Clamp01(stage3MinForwardIntentFraction));
                    float along = Mathf.Max(Vector3.Dot(residual, intentDir), minAlong);
                    Vector3 lateral = residual - Vector3.Dot(residual, intentDir) * intentDir;
                    residual = intentDir * along + lateral;

                    float dt = Mathf.Max(Time.deltaTime, 1e-4f);
                    _smoothedStage3Residual = Vector3.MoveTowards(
                        _smoothedStage3Residual,
                        residual,
                        Mathf.Max(0.01f, stage3ResidualSlewRate) * dt);
                    _smoothedStage3Residual = Vector3.Lerp(
                        _smoothedStage3Residual,
                        residual,
                        1f - Mathf.Exp(-Mathf.Max(0.01f, stage3ResidualResponsiveness) * dt));

                    UserTargetVel = PlayerIntentVel + _smoothedStage3Residual * Mathf.Clamp01(stage3ResidualBlend);
                    LastDesiredVel = UserTargetVel;
                }
                else if (!PlayerIntentActive)
                {
                    _smoothedStage3Residual = Vector3.zero;
                }
            }

            if (config != null && !config.aiEnabled)
            {
                UserActive = hasManualInput;
            }

            if (Input.GetKeyDown(KeyCode.LeftBracket)) ToggleAI(false);
            if (Input.GetKeyDown(KeyCode.RightBracket)) ToggleAI(true);
            if (Input.GetKeyDown(KeyCode.R)) ResetAll();

            // During automatic collection the exploration agent owns PlayerIntentVel.
            // In Stage3Policy mode inference also runs for zero intent so the policy can brake and hold.
        }

        private void UpdateAttitude(float dt, Vector3 desiredVel)
        {
            if (balloon == null) return;

            Vector3 targetForward;
            if (config.aiEnabled && !UserActive && autopilot != null)
            {
                targetForward = autopilot.HoldForwardTarget;
            }
            else
            {
                Vector3 horizontal = new Vector3(desiredVel.x, 0f, desiredVel.z);
                targetForward = horizontal.sqrMagnitude > 1e-6f ? horizontal.normalized : balloon.transform.forward;
            }

            Vector3 cur = balloon.transform.forward;
            float maxDeg = config.attitudeMaxDegPerSec * dt;
            float t = Mathf.Clamp01(config.attitudeResponsiveness * dt);
            Vector3 blended = Vector3.Slerp(cur, targetForward, t);
            Vector3 next = Vector3.RotateTowards(cur, blended, maxDeg * Mathf.Deg2Rad, 0f);

            if (next.sqrMagnitude > 1e-8f)
                balloon.transform.rotation = Quaternion.LookRotation(next.normalized, Vector3.up);

            balloon.forward = balloon.transform.forward;
            balloon.yawDeg = balloon.transform.eulerAngles.y;
        }
    }
}
