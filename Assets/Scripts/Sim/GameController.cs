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
        public Vector3 LastDesiredVel { get; private set; }
        public Vector3 LastAcceleration { get; private set; }

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

            Vector3 desiredVel = autopilot != null
                ? autopilot.ComputeControl(UserTargetVel, UserActive, dt)
                : UserTargetVel;

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
            if (hasManualInput)
            {
                Vector3 localDir = raw.normalized;
                Transform frame = inputFrame != null ? inputFrame : (Camera.main != null ? Camera.main.transform : null);
                Vector3 worldDir = frame != null ? (frame.rotation * localDir) : localDir;

                UserTargetVel = worldDir * spd;

                if (config != null && config.aiEnabled && stage3Policy != null && stage3Policy.TryPredictAction(out var policyVel))
                {
                    float blend = 0.35f;
                    UserTargetVel = Vector3.Lerp(UserTargetVel, policyVel, blend);
                    LastDesiredVel = policyVel;
                }

                UserActive = true;
            }
            else
            {
                UserTargetVel = Vector3.zero;
                UserActive = false;
            }

            if (Input.GetKeyDown(KeyCode.LeftBracket)) ToggleAI(false);
            if (Input.GetKeyDown(KeyCode.RightBracket)) ToggleAI(true);
            if (Input.GetKeyDown(KeyCode.R)) ResetAll();

            var explorer = FindObjectOfType<RandomExplorationAgent>();
            if (explorer != null && explorer.IsExploring)
            {
                UserTargetVel = explorer.ExplorationTargetVel;
                UserActive = true;
                return;
            }

            // Stage3 policy only assists active player movement. With no WASD/J/K input,
            // keep UserActive=false so Stage2/Control Assist can hold the last safe pose.
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
