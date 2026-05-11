using UnityEngine;

namespace BalloonSim.Sim
{
    /// <summary>
    /// 随机探索 Agent - 用于自动采集训练数据
    /// 替代人工操作，确保多工况均匀覆盖
    /// </summary>
    public class RandomExplorationAgent : MonoBehaviour
    {
        [Header("References")]
        public SimulationConfig config;
        public BalloonState balloon;
        public GameController game;
        public TurbulenceField field;
        public TrainingDataLogger trainingLogger;

        [Header("Exploration")]
        [Tooltip("是否启用自动探索")]
        public bool enabled = false;
        [Tooltip("目标到达判定半径")]
        public float reachRadius = 2.0f;
        [Tooltip("探索速度系数")]
        [Range(0.1f, 3f)] public float speedScale = 1.0f;
        [Tooltip("随机扰动强度")]
        [Range(0f, 1f)] public float perturbStrength = 0.3f;
        [Tooltip("扰动变化间隔 (秒)")]
        public float perturbInterval = 0.8f;
        [Tooltip("工况切换间隔 (秒)")]
        public float conditionSwitchInterval = 30f;
        [Tooltip("每个 episode 的时长 (秒)")]
        public float episodeDuration = 120f;

        [Header("Condition Sweep")]
        [Tooltip("Beaufort 扫描范围")]
        public float beaufortMin = 1f;
        public float beaufortMax = 12f;
        [Tooltip("Gust 扫描范围")]
        public float gustMin = 0f;
        public float gustMax = 15f;
        [Tooltip("Tornado 出现概率")]
        [Range(0f, 1f)] public float tornadoProb = 0.15f;
        [Tooltip("Convection 扫描范围")]
        public float convectionMin = 0f;
        public float convectionMax = 10f;

        private Vector3 _target;
        private Vector3 _perturb;
        private float _perturbTimer;
        private float _conditionTimer;
        private float _episodeTimer;
        private int _episodeCount;
        private System.Random _rng;

        public Vector3 ExplorationTargetVel { get; private set; }
        public bool IsExploring => enabled;

        private void Awake()
        {
            _rng = new System.Random(System.DateTime.Now.Millisecond);
        }

        private void Start()
        {
            // 延迟到 Start，确保 config 已被 Bootstrap 赋值
            if (config != null)
            {
                PickNewTarget();
                RandomizeConditions();
            }
        }

        private void FixedUpdate()
        {
            if (!enabled || balloon == null || config == null) return;

            float dt = Time.fixedDeltaTime;

            _episodeTimer += dt;
            if (_episodeTimer >= episodeDuration)
            {
                EndEpisode();
                StartNewEpisode();
            }

            _conditionTimer += dt;
            if (_conditionTimer >= conditionSwitchInterval)
            {
                RandomizeConditions();
                _conditionTimer = 0f;
            }

            _perturbTimer += dt;
            if (_perturbTimer >= perturbInterval)
            {
                _perturb = new Vector3(
                    (float)(_rng.NextDouble() * 2 - 1),
                    (float)(_rng.NextDouble() * 2 - 1),
                    (float)(_rng.NextDouble() * 2 - 1)
                ) * perturbStrength;
                _perturbTimer = 0f;
            }

            Vector3 toTarget = _target - balloon.transform.position;
            float dist = toTarget.magnitude;

            if (dist < reachRadius)
            {
                PickNewTarget();
                toTarget = _target - balloon.transform.position;
                dist = toTarget.magnitude;
            }

            Vector3 dir = dist > 0.01f ? toTarget / dist : Vector3.zero;
            float speed = config.throttle * config.throttle * config.manualSpeedScale * speedScale;
            float approach = Mathf.Clamp01(dist / 4f);
            ExplorationTargetVel = (dir * speed * approach) + _perturb;
        }

        private void PickNewTarget()
        {
            if (config == null) return;
            
            float s = config.boxSize;
            float margin = config.boxMinMargin + 1f;
            _target = new Vector3(
                Rf(margin, s - margin),
                Rf(margin, s - margin),
                Rf(margin, s - margin)
            );
        }

        private void RandomizeConditions()
        {
            if (config == null) return;
            
            config.beaufort = Rf(beaufortMin, beaufortMax);
            config.gustStrength = Rf(gustMin, gustMax);
            config.gustDirDeg = Rf(-180f, 180f);
            config.convectionStrength = Rf(convectionMin, convectionMax);
            config.tornado = _rng.NextDouble() < tornadoProb;
            config.viscosity = Rf(0.05f, 2f);
            config.reynolds = Rf(500f, 50000f);
            config.randomStrength = Rf(0.5f, 5f);
            config.wakeStrength = Rf(0.2f, 5f);
            config.densityRatio = Rf(0.5f, 1.5f);

            if (field != null && _rng.NextDouble() < 0.3)
                field.Generate();

            Debug.Log($"[ExplorationAgent] Conditions: Bft={config.beaufort:F1} " +
                      $"Gust={config.gustStrength:F1} Conv={config.convectionStrength:F1} " +
                      $"Tornado={config.tornado} Re={config.reynolds:F0}");
        }

        private void StartNewEpisode()
        {
            if (config == null || balloon == null) return;
            
            _episodeTimer = 0f;
            _episodeCount++;
            PickNewTarget();

            if (_rng.NextDouble() < 0.3 && balloon != null)
            {
                float s = config.boxSize;
                float m = 2f;
                balloon.transform.position = new Vector3(Rf(m, s - m), Rf(m, s - m), Rf(m, s - m));
                balloon.velocity = Vector3.zero;
            }

            if (trainingLogger != null)
                trainingLogger.StartNewEpisode();
        }

        private void EndEpisode()
        {
            if (trainingLogger != null)
                trainingLogger.FlushEpisode();
        }

        private float Rf(float min, float max)
        {
            return (float)(_rng.NextDouble() * (max - min) + min);
        }
    }
}
