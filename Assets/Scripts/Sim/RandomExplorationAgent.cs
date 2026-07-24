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
        public DataLogger stage3Logger;

        [Header("Exploration")]
        [Tooltip("是否启用自动探索")]
        public bool explorationEnabled = false;
        [Tooltip("Stage2 capture includes move-release-hold cycles so data covers control-relevant local prediction.")]
        public bool includeReleaseHoldCycles = true;
        [Tooltip("Seconds of active movement before release/hold sampling.")]
        public float movePhaseDuration = 4.0f;
        [Tooltip("Seconds of release/hold sampling after movement.")]
        public float holdPhaseDuration = 3.0f;
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
        [Tooltip("Episode 切换后的稳定等待时间，等待后才创建新的 episode 文件。")]
        public float episodeWarmupSeconds = 8f;

        [Header("Condition Sweep")]
        [Tooltip("Beaufort 扫描范围")]
        public float beaufortMin = 7.5f;
        public float beaufortMax = 12.5f;
        [Tooltip("Gust 扫描范围")]
        public float gustMin = 6f;
        public float gustMax = 18f;
        [Tooltip("Tornado 出现概率")]
        [Range(0f, 1f)] public float tornadoProb = 0.65f;
        [Tooltip("Convection 扫描范围")]
        public float convectionMin = 4f;
        public float convectionMax = 13f;

        [Header("Stage2 Capture Coverage")]
        public bool stage2CoverageMode = true;
        [Tooltip("Probability of short gust/convection jumps inside an episode.")]
        [Range(0f, 1f)] public float burstProb = 0.55f;
        public float burstIntervalMin = 2.0f;
        public float burstIntervalMax = 7.0f;
        public float persistentRegimeInterval = 18f;
        public float altitudeBandPadding = 1.4f;

        private Vector3 _target;
        private Vector3 _perturb;
        private float _perturbTimer;
        private float _conditionTimer;
        private float _episodeTimer;
        private float _regimeTimer;
        private float _burstTimer;
        private float _nextBurstInterval;
        private float _phaseTimer;
        private bool _releaseHoldPhase;
        private int _episodeCount;
        private int _regimeIndex;
        private bool _stage2CapturePending;
        private bool _episodeTransitionPending;
        private bool _captureSessionRequested;
        private float _episodeTransitionTimer;
        private System.Random _rng;
        private Vector3 _simulatedIntent;

        [Header("Stage3 Keyboard Intent Coverage")]
        public float intentPhaseMinSeconds = 0.2f;
        public float intentPhaseMaxSeconds = 2.5f;
        [Range(0f, 1f)] public float idleIntentProbability = 0.25f;
        [Range(0f, 1f)] public float reverseIntentProbability = 0.15f;
        public int smallScaleTargetSamples = 50000;
        public int largeScaleTargetSamples = 200000;

        public Vector3 ExplorationTargetVel { get; private set; }
        public bool IsExploring => explorationEnabled;
        public bool Stage2CapturePending => _stage2CapturePending || _episodeTransitionPending;
        public bool CaptureSessionRequested => _captureSessionRequested;

        public void RequestStage2Capture(TrainingDataLogger logger)
        {
            trainingLogger = logger;
            explorationEnabled = true;
            _captureSessionRequested = true;
            _stage2CapturePending = true;
            _episodeTransitionPending = false;
            _episodeTransitionTimer = 0f;
            _episodeTimer = 0f;
            if (trainingLogger != null)
            {
                trainingLogger.logMode = TrainingDataLogger.LogMode.TrainingData;
                trainingLogger.trainingDataDirectory = "training_data";
                trainingLogger.SetCaptureEnabled(false);
                trainingLogger.SetSamplePhase("warmup");
                trainingLogger.CloseCurrentEpisode();
                trainingLogger.PauseLogging();
            }
        }

        public void RequestStage3Capture(DataLogger logger)
        {
            stage3Logger = logger;
            explorationEnabled = true;
            _captureSessionRequested = true;
            _stage2CapturePending = false;
            _episodeTransitionPending = false;
            _episodeTransitionTimer = 0f;
            _episodeTimer = 0f;
            _phaseTimer = 0f;
            _releaseHoldPhase = false;
            PickKeyboardIntent();
            if (stage3Logger != null)
            {
                stage3Logger.enabledLogging = true;
                stage3Logger.SetStage3Mode("exploration");
                stage3Logger.StartLogging();
                stage3Logger.PauseLogging(false);
                stage3Logger.BeginEpisode();
            }
            if (game != null)
            {
                game.explorationOverride = true;
                game.SetExplorationTarget(_simulatedIntent, _simulatedIntent.sqrMagnitude > 1e-6f);
            }
        }


        public void StopStage2Capture()
        {
            explorationEnabled = false;
            _captureSessionRequested = false;
            _stage2CapturePending = false;
            _episodeTransitionPending = false;
            _episodeTransitionTimer = 0f;
            ExplorationTargetVel = Vector3.zero;
            _simulatedIntent = Vector3.zero;
            if (game != null)
            {
                game.SetExplorationTarget(Vector3.zero, false);
                game.explorationOverride = false;
                game.SetUserTarget(Vector3.zero, false);
            }
            if (trainingLogger != null)
            {
                trainingLogger.SetCaptureEnabled(false);
                trainingLogger.SetSamplePhase("transition");
                trainingLogger.CloseCurrentEpisode();
                trainingLogger.PauseLogging();
            }
            if (stage3Logger != null)
            {
                stage3Logger.PauseLogging(true);
            }
        }

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
            if (!explorationEnabled || !_captureSessionRequested || balloon == null || config == null) return;
            var thermo = balloon.GetComponent<BalloonThermodynamics>();
            if (thermo != null && !thermo.IsWarmedUp)
            {
                _stage2CapturePending = false;
            }

            float dt = Time.fixedDeltaTime;

            if (game != null)
                game.SetExplorationTarget(
                    ExplorationTargetVel,
                    ExplorationTargetVel.sqrMagnitude > 1e-6f && _captureSessionRequested && !_episodeTransitionPending && !_stage2CapturePending);

            if (_episodeTransitionPending)
            {
                _episodeTransitionTimer += dt;
                if (_episodeTransitionTimer < episodeWarmupSeconds)
                {
                    if (stage3Logger != null)
                        stage3Logger.PauseLogging(true);
                    return;
                }

                _episodeTransitionPending = false;
                _episodeTransitionTimer = 0f;
                BeginStableEpisode();
            }

            _episodeTimer += dt;
            if (_episodeTimer >= episodeDuration)
            {
                EndEpisode();
                PrepareNextEpisodeConditions();
                _episodeTransitionPending = true;
                _episodeTransitionTimer = 0f;
                return;
            }

            _burstTimer += dt;
            _phaseTimer += dt;
            if (_phaseTimer >= movePhaseDuration)
            {
                _phaseTimer = 0f;
                PickKeyboardIntent();
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
                if (trainingLogger != null) trainingLogger.SetSamplePhase(_releaseHoldPhase ? "transition" : "stable");
            }

            float speed = Mathf.Max(0.5f, config.throttle * config.throttle * config.manualSpeedScale * speedScale);
            ExplorationTargetVel = _simulatedIntent.sqrMagnitude > 1e-6f
                ? _simulatedIntent.normalized * speed
                : Vector3.zero;
            if (game != null)
                game.SetExplorationTarget(ExplorationTargetVel, _simulatedIntent.sqrMagnitude > 1e-6f);
            if (stage3Logger != null)
                stage3Logger.PauseLogging(false);
        }

        private void PickKeyboardIntent()
        {
            movePhaseDuration = Rf(intentPhaseMinSeconds, intentPhaseMaxSeconds);
            if (_rng.NextDouble() < idleIntentProbability)
            {
                _simulatedIntent = Vector3.zero;
                return;
            }

            Vector3[] intents =
            {
                Vector3.forward, Vector3.back, Vector3.left, Vector3.right, Vector3.up, Vector3.down,
                new Vector3(-1f, 0f, 1f), new Vector3(1f, 0f, 1f),
                new Vector3(-1f, 0f, -1f), new Vector3(1f, 0f, -1f),
                new Vector3(0f, 1f, 1f), new Vector3(0f, -1f, 1f),
                new Vector3(1f, 1f, 0f), new Vector3(-1f, -1f, 0f)
            };
            Vector3 next = intents[_rng.Next(intents.Length)].normalized;
            if (_simulatedIntent.sqrMagnitude > 1e-6f && _rng.NextDouble() < reverseIntentProbability)
                next = -_simulatedIntent.normalized;
            _simulatedIntent = next;
        }

        private void PickNewTarget()
        {
            if (config == null) return;
            
            float s = config.boxSize;
            float margin = config.boxMinMargin + 1f;
            float y = stage2CoverageMode ? PickAltitudeByBand(s, margin) : Rf(margin, s - margin);
            _target = new Vector3(
                Rf(margin, s - margin),
                y,
                Rf(margin, s - margin)
            );
        }

        private void RandomizeConditions()
        {
            if (config == null) return;
            
            if (stage2CoverageMode)
                ApplyCoverageRegime();
            else
            {
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
            }

            if (field != null && (stage2CoverageMode || _rng.NextDouble() < 0.3))
                field.Generate();

            Debug.Log($"[ExplorationAgent] Conditions: Bft={config.beaufort:F1} " +
                      $"Gust={config.gustStrength:F1} Conv={config.convectionStrength:F1} " +
                      $"Tornado={config.tornado} Re={config.reynolds:F0}");
        }

        private void PrepareNextEpisodeConditions()
        {
            if (config == null || balloon == null) return;

            _episodeTimer = 0f;
            _conditionTimer = 0f;
            _regimeTimer = 0f;
            _burstTimer = 0f;
            _phaseTimer = 0f;
            _releaseHoldPhase = false;
            _nextBurstInterval = Rf(burstIntervalMin, burstIntervalMax);
            _episodeCount++;
            RandomizeConditions();
            PickNewTarget();
            ResetBalloonForCoverage();

            if (trainingLogger != null)
            {
                trainingLogger.SetCaptureEnabled(false);
                trainingLogger.SetSamplePhase("transition");
                trainingLogger.CloseCurrentEpisode();
            }
        }

        private void BeginStableEpisode()
        {
            if (_captureSessionRequested && stage3Logger != null)
            {
                stage3Logger.BeginEpisode();
                stage3Logger.SetStage3Mode(stage2CoverageMode ? "exploration" : "policy");
            }

            if (stage3Logger != null && stage3Logger.enabledLogging)
            {
                stage3Logger.SetStage3Mode(stage2CoverageMode ? "exploration" : "policy");
            }

            if (trainingLogger == null || !_captureSessionRequested) return;
            trainingLogger.ResumeLogging();
            trainingLogger.SetSamplePhase("stable");
            trainingLogger.SetCaptureEnabled(true);
            trainingLogger.StartNewEpisode();
            if (!trainingLogger.IsSessionOpen)
                _captureSessionRequested = false;
        }

        private void EndEpisode()
        {
            if (trainingLogger != null)
                trainingLogger.SetCaptureEnabled(false);
            if (trainingLogger != null)
                trainingLogger.SetSamplePhase("transition");
            if (trainingLogger != null)
                trainingLogger.FlushEpisode();
            if (stage3Logger != null)
                stage3Logger.PauseLogging(true);
        }

        private void ApplyCoverageRegime()
        {
            int regime = _regimeIndex++ % 5;
            switch (regime)
            {
                case 0: // sustained strong mixed wind
                    config.beaufort = Rf(8.0f, 10.0f);
                    config.gustStrength = Rf(6f, 10f);
                    config.convectionStrength = Rf(4f, 7f);
                    config.tornado = _rng.NextDouble() < 0.45;
                    break;
                case 1: // extreme gust-heavy regime
                    config.beaufort = Rf(9.5f, 12.5f);
                    config.gustStrength = Rf(12f, 20f);
                    config.convectionStrength = Rf(5f, 10f);
                    config.tornado = _rng.NextDouble() < 0.55;
                    break;
                case 2: // convection-heavy vertical mixing
                    config.beaufort = Rf(7.5f, 11.0f);
                    config.gustStrength = Rf(7f, 14f);
                    config.convectionStrength = Rf(9f, 16f);
                    config.tornado = _rng.NextDouble() < 0.6;
                    break;
                case 3: // tornado / rotational disturbance
                    config.beaufort = Rf(8.5f, 12.0f);
                    config.gustStrength = Rf(8f, 16f);
                    config.convectionStrength = Rf(6f, 13f);
                    config.tornado = true;
                    break;
                default: // recovery / moderate strong regime for transition coverage
                    config.beaufort = Rf(6.5f, 9.0f);
                    config.gustStrength = Rf(4f, 9f);
                    config.convectionStrength = Rf(3f, 7f);
                    config.tornado = _rng.NextDouble() < 0.35;
                    break;
            }

            config.gustDirDeg = ((regime * 72f) + Rf(-35f, 35f) + 360f) % 360f - 180f;
            config.viscosity = Rf(0.08f, 2.4f);
            config.reynolds = Rf(800f, 65000f);
            config.randomStrength = Rf(1.5f, 7.0f);
            config.wakeStrength = Rf(0.8f, 6.0f);
            config.densityRatio = Rf(0.7f, 1.25f);
        }

        private void ApplyShortBurst()
        {
            if (_rng.NextDouble() > burstProb) return;
            config.gustStrength = Mathf.Clamp(config.gustStrength + Rf(-4f, 8f), gustMin, gustMax + 6f);
            config.convectionStrength = Mathf.Clamp(config.convectionStrength + Rf(-3f, 6f), convectionMin, convectionMax + 6f);
            config.gustDirDeg = Mathf.Repeat(config.gustDirDeg + Rf(-80f, 80f) + 180f, 360f) - 180f;
            config.tornado = config.tornado || _rng.NextDouble() < 0.2;
            if (field != null && _rng.NextDouble() < 0.35)
                field.Generate();
        }

        private float PickAltitudeByBand(float size, float margin)
        {
            float low = margin + altitudeBandPadding;
            float high = size - margin - altitudeBandPadding;
            int band = _episodeCount % 4;
            if (band == 0) return Rf(low, Mathf.Lerp(low, high, 0.30f));
            if (band == 1) return Rf(Mathf.Lerp(low, high, 0.30f), Mathf.Lerp(low, high, 0.60f));
            if (band == 2) return Rf(Mathf.Lerp(low, high, 0.60f), high);
            return Rf(low, high);
        }

        private void ResetBalloonForCoverage()
        {
            float s = config.boxSize;
            float m = config.boxMinMargin + 1.2f;
            balloon.transform.position = new Vector3(Rf(m, s - m), PickAltitudeByBand(s, m), Rf(m, s - m));
            balloon.velocity = new Vector3(Rf(-0.8f, 0.8f), Rf(-0.5f, 0.5f), Rf(-0.8f, 0.8f));
        }

        private float Rf(float min, float max)
        {
            return (float)(_rng.NextDouble() * (max - min) + min);
        }
    }
}
