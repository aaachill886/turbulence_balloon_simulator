using UnityEngine;
using BalloonSim.Sim;

namespace BalloonSim.UI
{
    public class SimUIOnGUI : MonoBehaviour
    {
        public SimulationConfig config;
        public GameController game;
        public TurbulenceField field;
        public AutopilotController autopilot;
        public BalloonState balloon;
        public DataLogger logger;
        public BenchmarkRunner benchmarkRunner;
        public RandomExplorationAgent explorer;
        public TrainingDataLogger trainingLogger;
        public Stage3PolicyRunner policyRunner;

        private bool _showBasic = true;
        private bool _showAdv;
        private Vector2 _basicScroll;
        private Vector2 _advScroll;
        private Vector2 _topScroll;
        private bool _confirmClearTraining;
        private bool _confirmClearStage2;

        private void Awake()
        {
            if (logger == null)
                logger = FindObjectOfType<DataLogger>();
            if (benchmarkRunner == null)
                benchmarkRunner = FindObjectOfType<BenchmarkRunner>();
            if (explorer == null)
                explorer = FindObjectOfType<RandomExplorationAgent>();
            if (trainingLogger == null)
                trainingLogger = FindObjectOfType<TrainingDataLogger>();
            if (policyRunner == null)
                policyRunner = FindObjectOfType<Stage3PolicyRunner>();

            if (benchmarkRunner == null)
            {
                var host = FindObjectOfType<GameController>();
                if (host != null)
                    benchmarkRunner = host.gameObject.AddComponent<BenchmarkRunner>();
                else
                    benchmarkRunner = gameObject.AddComponent<BenchmarkRunner>();
            }
        }

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.Tab)) _showBasic = !_showBasic;

            if (Input.GetKey(KeyCode.LeftControl) && Input.GetKeyDown(KeyCode.I))
                _showAdv = !_showAdv;

            if (Input.GetKeyDown(KeyCode.F))
                config.showVolumetricFog = !config.showVolumetricFog;

            if (Input.GetKeyDown(KeyCode.H))
                config.showHeatmapPoints = !config.showHeatmapPoints;

            if (Input.GetKeyDown(KeyCode.M))
                config.showMinimap = !config.showMinimap;
        }

        private void OnGUI()
        {
            if (logger == null)
                logger = FindObjectOfType<DataLogger>();

            if (config == null || balloon == null) return;

            GUILayout.BeginArea(new Rect(10, 10, 980, 390), GUI.skin.box);
            _topScroll = GUILayout.BeginScrollView(_topScroll, GUILayout.Width(960), GUILayout.Height(370));
            string modeLabel = config.controlMode switch
            {
                ControlMode.TrueManual => "manual-true-flow",
                ControlMode.BaselineHold => "baseline-hold",
                ControlMode.AssistPredictor => "assist+predictor",
                ControlMode.Stage3Policy => "stage3-policy",
                _ => "unknown",
            };
            GUILayout.Label($"mode: {modeLabel}");
            GUILayout.Label($"controlMode: {config.controlMode}");
            GUILayout.Label($"pos: {balloon.transform.position.x:F1}, {balloon.transform.position.y:F1}, {balloon.transform.position.z:F1}");
            GUILayout.Label($"vel: {balloon.velocity.x:F1}, {balloon.velocity.y:F1}, {balloon.velocity.z:F1}");
            GUILayout.Label($"yaw:{balloon.yawDeg:F1}°  fwd:{balloon.forward.x:F2},{balloon.forward.z:F2}");

            Vector3 u = field != null ? field.Sample(balloon.transform.position) : Vector3.zero;
            float turb = u.magnitude;
            bool assistOn = config.controlMode != ControlMode.TrueManual;
            string pred = autopilot != null && assistOn ? autopilot.PredErr.ToString("F2") : "-";
            string pmse = autopilot != null && assistOn ? autopilot.PredMSE.ToString("F4") : "-";
            string sig = autopilot != null && assistOn ? autopilot.PredSigma.ToString("F2") : "-";
            string thr = autopilot != null && assistOn ? autopilot.ThrottleNeed.ToString("F2") : "-";
            string stage2Line = $"Controller: mode={(config.controlMode == ControlMode.TrueManual ? "manual" : "assist")} | hold={(assistOn ? "on" : "off")} | wind_sample={u.x:F2},{u.y:F2},{u.z:F2}";
            string stage3Mode = logger != null ? (logger.enabledLogging ? (logger.paused ? "paused" : "recording") : "off") : "n/a";
            string stage3Line = $"Stage3 policy log: {stage3Mode} dir={logger?.LogDirectory ?? "n/a"} episode={logger?.episodeId.ToString() ?? "n/a"} run={logger?.runId ?? "n/a"} mode={logger?.mode ?? "n/a"}";
            var policyRunner = FindObjectOfType<Stage3PolicyRunner>();
            string stage3Action = policyRunner != null ? $"{policyRunner.LastAction.x:F3},{policyRunner.LastAction.y:F3},{policyRunner.LastAction.z:F3}" : "n/a";
            string stage3PolicyLine = policyRunner != null
                ? $"Stage3 policy: mode={policyRunner.RuntimeMode} status={policyRunner.LastStatusMessage} ms={policyRunner.LastInferenceMs:F2} modelSrc={policyRunner.LastModelSource} normSrc={policyRunner.LastNormSource} metaSrc={policyRunner.LastMetaSource} action={stage3Action}"
                : "Stage3 policy: n/a";

            GUILayout.Label(stage2Line);
            GUILayout.Label(stage3Line);
            GUILayout.Label(stage3PolicyLine);
            GUILayout.Label($"turb:{turb:F2}  Bft:{config.beaufort:F1}  rhoB/rhoA:{config.densityRatio:F2}  pred:{pred} mse:{pmse} sigma:{sig} throttle:{thr}");
            string holdMode = config.controlMode switch
            {
                ControlMode.TrueManual => "manual",
                ControlMode.BaselineHold => "baseline-hold",
                ControlMode.AssistPredictor => "assist-hold",
                ControlMode.Stage3Policy => "stage3-policy",
                _ => "unknown",
            };
            string oracle = config.assistUseOracleEnvCancellation ? "on" : "off";
            GUILayout.Label($"holdMode: {holdMode}  oracleEnv: {oracle}");
            GUILayout.Label($"fog:{(config.showVolumetricFog ? "on" : "off")} heat:{(config.showHeatmapPoints ? "on" : "off")} minimap:{(config.showMinimap ? "on" : "off")}");

            string hint = GetBestKeyHint();
            GUILayout.Label($"suggested low-resistance keys: {hint}");

            var explorer = this.explorer;
            var trainLog = this.trainingLogger;

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Reset (R)", GUILayout.Height(24))) game?.ResetAll();
            if (GUILayout.Button("Manual", GUILayout.Height(24))) config.controlMode = ControlMode.TrueManual;
            if (GUILayout.Button("Control Assist", GUILayout.Height(24))) config.controlMode = ControlMode.BaselineHold;
            GUILayout.EndHorizontal();
            GUILayout.Space(6);

            GUILayout.Space(6);
            GUILayout.BeginVertical(GUI.skin.box);
            GUILayout.Label("Stage3 Training");
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Start Stage3 Training", GUILayout.Height(28)) && explorer != null && logger != null)
                ApplyStage3CaptureMode(explorer, logger);
            if (GUILayout.Button("Pause Training Log", GUILayout.Height(28)) && logger != null)
                logger.PauseLogging(true);
            if (GUILayout.Button("Resume Training Log", GUILayout.Height(28)) && logger != null)
                logger.PauseLogging(false);
            if (GUILayout.Button("Clear Training Log", GUILayout.Height(28)) && logger != null)
            {
                if (explorer != null) explorer.StopStage2Capture();
                logger.ClearAllLogs();
            }
            if (GUILayout.Button("Open Training Log", GUILayout.Height(28)) && logger != null)
                logger.OpenLogFolder();
            GUILayout.EndHorizontal();
            if (explorer != null && logger != null)
            {
                int targetSamples = logger.WrittenSamples < explorer.smallScaleTargetSamples
                    ? explorer.smallScaleTargetSamples
                    : explorer.largeScaleTargetSamples;
                float progress = targetSamples > 0 ? Mathf.Clamp01((float)logger.WrittenSamples / targetSamples) : 0f;
                GUILayout.Label($"Intent collection: {logger.WrittenSamples:N0} / {targetSamples:N0} ({progress:P1}) | phase={(logger.WrittenSamples < explorer.smallScaleTargetSamples ? "small-scale validation" : "large-scale")}");
            }
            GUILayout.EndVertical();

            GUILayout.BeginVertical(GUI.skin.box);
            GUILayout.Label("Stage3 Application");
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Apply Stage3 Policy Mode", GUILayout.Height(28)))
                ApplyStage3PolicyMode(explorer, logger, this.policyRunner);
            if (GUILayout.Button(this.policyRunner != null && this.policyRunner.enablePolicy ? "Policy: ON" : "Policy: OFF", GUILayout.Height(28)) && this.policyRunner != null)
            {
                this.policyRunner.enablePolicy = !this.policyRunner.enablePolicy;
                this.policyRunner.Reinitialize();
            }
            if (GUILayout.Button("Stop Policy", GUILayout.Height(28)))
            {
                if (explorer != null) explorer.StopStage2Capture();
                if (logger != null) logger.PauseLogging(true);
            }
            if (GUILayout.Button("Start Policy Log", GUILayout.Height(28)) && logger != null)
                StartOrResumeStage3Log(explorer);
            GUILayout.EndHorizontal();
            GUILayout.EndVertical();

            GUILayout.EndScrollView();
            GUILayout.EndArea();

            if (_showBasic)
            {
                GUILayout.BeginArea(new Rect(10, Screen.height - 430, 430, 420), GUI.skin.box);
                GUILayout.Label("Parameters (Tab hide) - Scroll to view more");
                _basicScroll = GUILayout.BeginScrollView(_basicScroll, GUILayout.Width(410), GUILayout.Height(380));
                Slider(ref config.throttle, 0f, 10f, "Throttle");
                Slider(ref config.manualSpeedScale, 0.05f, 2f, "Control Rate");
                Slider(ref config.densityRatio, 0.05f, 3f, "rhoB/rhoA");

                Slider(ref config.beaufort, 0f, 30f, "Beaufort");
                Slider(ref config.viscosity, 0f, 8f, "Viscosity");
                Slider(ref config.reynolds, 20f, 5e5f, "Reynolds");
                Slider(ref config.randomStrength, 0f, 50f, "Random");
                Slider(ref config.gustStrength, 0f, 60f, "Gust");
                Slider(ref config.gustDirDeg, -180f, 180f, "Gust Dir");
                Slider(ref config.convectionStrength, 0f, 60f, "Convection");
                Slider(ref config.wakeStrength, 0f, 30f, "Wake");
                Slider(ref config.balloonRadius, 0.1f, 6f, "Balloon Size");

                GUILayout.Space(8);
                GUILayout.Label("Volumetric Fog");
                float fogCount = config.fogParticleCount;
                Slider(ref fogCount, 100f, 3000f, "Fog Particle Count");
                config.fogParticleCount = (int)fogCount;
                Slider(ref config.fogParticleSize, 0.1f, 3f, "Fog Particle Size");
                Slider(ref config.fogAlpha, 0f, 1f, "Fog Alpha");
                GUILayout.EndScrollView();
                GUILayout.EndArea();
            }

            if (_showAdv)
            {
                GUILayout.BeginArea(new Rect(460, Screen.height - 450, 430, 440), GUI.skin.box);
                GUILayout.Label("Advanced (Ctrl+I hide) - Scroll to view more");

                GUILayout.BeginHorizontal();
                bool canRunBenchmark = benchmarkRunner != null;
                GUI.enabled = canRunBenchmark;
                if (GUILayout.Button("Start Benchmark", GUILayout.Height(26))) benchmarkRunner.StartBenchmark();
                GUI.enabled = true;
                GUILayout.Label(canRunBenchmark ? "baseline vs AI benchmark" : "benchmarkRunner not found");
                GUILayout.EndHorizontal();

                _advScroll = GUILayout.BeginScrollView(_advScroll, GUILayout.Width(410), GUILayout.Height(400));

                Toggle(ref config.enableBlockGeneration, "Block Generation (unlock new blocks)");
                GUILayout.Label($"Waypoint spawn mode: {(WaypointSpawnMode)(int)config.waypointSpawnMode}");
                Slider(ref config.waypointSpawnMode, 0f, 2f, "Waypoint Spawn Mode (0 unlocked / 1 new / 2 both)");

                Toggle(ref config.showMinimap, "Minimap (M)");
                Toggle(ref config.showVolumetricFog, "Volumetric Fog (F)");
                Toggle(ref config.showHeatmapPoints, "Semantic Field (H)");

                GUILayout.Space(8);
                GUILayout.Label("Semantic Field Weights");
                Slider(ref config.semanticAlignW, 0f, 3f, "semanticAlignW");
                Slider(ref config.semanticMagW, 0f, 3f, "semanticMagW");
                Slider(ref config.semanticGradW, 0f, 3f, "semanticGradW");
                Slider(ref config.semanticCalmBand, 0f, 1f, "semanticCalmBand");

                Slider(ref config.visRangeK, 0f, 3f, "visRangeK");
                Slider(ref config.visAmpK, 0f, 3f, "visAmpK");
                Slider(ref config.wakeBaseR, 0.1f, 20f, "wakeBaseR");
                Slider(ref config.wakeSizeR, 0f, 20f, "wakeSizeR");
                Slider(ref config.buoyK, 0f, 2f, "buoyK");
                Slider(ref config.aiSafePosK, 0f, 2f, "aiSafePosK");
                Slider(ref config.aiSafeMaxK, 0f, 2f, "aiSafeMaxK");
                Slider(ref config.aiSafeDistK, 0f, 2f, "aiSafeDistK");
                Slider(ref config.aiSafeVelK, 0f, 2f, "aiSafeVelK");
                Slider(ref config.aiNeedK, 0f, 2f, "aiNeedK");

                GUILayout.Space(8);
                GUILayout.Label("Hold / Return Tuning");
                Slider(ref config.holdPosK, 0f, 4f, "holdPosK");
                Slider(ref config.holdVelK, 0f, 4f, "holdVelK");
                Slider(ref config.holdMaxSpeed, 0.5f, 8f, "holdMaxSpeed");
                Slider(ref config.holdForwardK, 0f, 2f, "holdForwardK");
                Slider(ref config.holdReleaseVelK, 0f, 2f, "holdReleaseVelK");
                Slider(ref config.holdEnvComp, 0f, 1f, "holdEnvComp");
                Slider(ref config.holdAltIK, 0f, 2f, "holdAltIK");
                Toggle(ref config.strictHoldNoDrift, "strictHoldNoDrift (benchmark)");
                Toggle(ref config.assistUseOracleEnvCancellation, "assistUseOracleEnvCancellation (debug)");
                Toggle(ref config.usePredSigmaConfidence, "usePredSigmaConfidence (attenuate by sigma)");

                GUILayout.Space(8);
                GUILayout.Label("Attitude");
                Slider(ref config.attitudeMaxDegPerSec, 30f, 360f, "attitudeMaxDegPerSec");
                Slider(ref config.attitudeResponsiveness, 1f, 30f, "attitudeResponsiveness");
                GUILayout.EndScrollView();
                GUILayout.EndArea();
            }

            GUILayout.BeginArea(new Rect(Screen.width * 0.5f - 500, Screen.height - 28, 1000, 24));
            GUILayout.Label("W/S/A/D/J/K move · Space reset view · Tab params · Ctrl+I advanced · F fog · H semantic-field · M minimap · Top buttons control mode/predictor");
            GUILayout.EndArea();
        }

        private void ApplyStrongMixedDisturbance()
        {
            config.beaufort = 9.5f;
            config.gustStrength = 10.5f;
            config.gustDirDeg = 35f;
            config.convectionStrength = 7.5f;
            config.tornado = true;
            config.viscosity = 0.85f;
            config.reynolds = 28000f;
            config.randomStrength = 3.5f;
            config.wakeStrength = 2.5f;
            config.densityRatio = 1.0f;
            field?.Generate();
        }

        private void ApplyStage3CaptureMode(RandomExplorationAgent explorer, DataLogger stage3Log)
        {
            ApplyStrongMixedDisturbance();
            config.controlMode = ControlMode.TrueManual;
            config.enableAIPredictor = false;
            config.throttle = 1.0f;
            config.manualSpeedScale = 0.5f;
            if (game != null)
            {
                game.expertVelocityErrorScale = 2.0f;
                game.expertTrackingResidualMax = 0.12f;
                game.expertWindScale = 50.0f;
                game.expertLateralWindResidualMax = 0.16f;
                game.expertAlongWindResidualMax = 0.06f;
                game.stage3IntentMaxSpeed = 0.5f;
                game.stage3ResidualMaxSpeed = 0.25f;
                game.stage3ResidualBlend = 1.0f;
            }

            if (explorer != null)
            {
                explorer.includeReleaseHoldCycles = true;
                explorer.RequestStage3Capture(stage3Log);
            }
            if (stage3Log != null)
            {
                stage3Log.enabledLogging = true;
                stage3Log.PauseLogging(false);
                stage3Log.SetStage3Mode("exploration");
                if (!stage3Log.IsInvoking()) { }
            }
        }

        private void ApplyStage3PolicyMode(RandomExplorationAgent explorer, DataLogger stage3Log, Stage3PolicyRunner policyRunner)
        {
            ApplyStrongMixedDisturbance();
            config.controlMode = ControlMode.Stage3Policy;
            config.enableAIPredictor = true;
            config.throttle = 1.0f;
            config.manualSpeedScale = 0.5f;
            config.densityRatio = 1.0f;
            config.holdPosK = 2.5f;
            config.holdVelK = 3.5f;
            config.holdMaxSpeed = 8.0f;
            config.holdForwardK = 0f;
            config.holdReleaseVelK = 0f;
            config.holdEnvComp = 0f;
            config.holdAltIK = 1.2f;
            config.strictHoldNoDrift = true;

            if (explorer != null)
            {
                explorer.StopStage2Capture();
                explorer.stage3Logger = stage3Log;
            }
            if (stage3Log != null)
            {
                stage3Log.enabledLogging = true;
                stage3Log.PauseLogging(false);
            }
            if (logger != null) logger.PauseLogging(true);
            ConfigureThermodynamicsForNeutralFlight();
            game?.ResetAll();
            ApplyStrongMixedDisturbance();
            config.controlMode = ControlMode.Stage3Policy;
            config.enableAIPredictor = false;
            config.throttle = 1.0f;
            config.manualSpeedScale = 0.5f;
            config.densityRatio = 1.0f;
            if (game != null)
            {
                game.stage3ResidualMaxSpeed = 0.25f;
                game.stage3ResidualBlend = 1.0f;
                game.stage3ResidualSlewRate = 2.0f;
                game.stage3ResidualResponsiveness = 6.0f;
                game.stage3MinForwardIntentFraction = 0.65f;
            }
            if (policyRunner != null)
            {
                policyRunner.enablePolicy = true;
                policyRunner.maxPolicySpeed = game != null ? game.stage3ResidualMaxSpeed : 0.5f;
                policyRunner.Reinitialize();
            }
        }

        private void ConfigureThermodynamicsForNeutralFlight()
        {
            if (balloon == null) return;
            var thermo = balloon.GetComponent<BalloonThermodynamics>();
            if (thermo == null) return;
            thermo.initialGasMass = 0.5f;
            thermo.envelopeMass = 8.5f;
            thermo.payloadMass = 3.0f;
            thermo.volume = 10.0f;
            thermo.currentGasMass = thermo.initialGasMass;
            thermo.gasTemperature = 288f;
            thermo.ambientTemperature = 288f;
        }

        private static void OpenTrainingFolder(TrainingDataLogger trainLog)
        {
            string projectRoot = System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, ".."));
            string dir = System.IO.Path.GetFullPath(System.IO.Path.Combine(projectRoot, trainLog.trainingDataDirectory));
            System.IO.Directory.CreateDirectory(dir);
            Application.OpenURL("file:///" + dir.Replace("\\", "/"));
        }

        private void StartOrResumeStage3Log(RandomExplorationAgent explorer)
        {
            if (logger == null) return;
            if (explorer != null) explorer.StopStage2Capture();
            config.controlMode = ControlMode.Stage3Policy;
            config.enableAIPredictor = false;
            logger.SetStage3Mode("policy");
            logger.runId = string.IsNullOrWhiteSpace(logger.runId) ? System.DateTime.Now.ToString("yyyyMMdd_HHmmss") : logger.runId;
            if (!logger.enabledLogging)
            {
                logger.enabledLogging = true;
                logger.StartLogging();
            }
            logger.PauseLogging(false);
        }

        private string GetBestKeyHint()
        {
            if (field == null || balloon == null) return "-";

            Vector3 pos = balloon.transform.position;
            Transform frame = game != null && game.inputFrame != null ? game.inputFrame : (Camera.main != null ? Camera.main.transform : null);
            Quaternion rot = frame != null ? frame.rotation : Quaternion.identity;

            (string key, Vector3 localDir)[] dirs =
            {
                ("W", Vector3.forward), ("S", Vector3.back),
                ("A", Vector3.left), ("D", Vector3.right),
                ("J", Vector3.down), ("K", Vector3.up)
            };

            float best = float.MaxValue;
            string bestKey = "-";
            foreach (var d in dirs)
            {
                Vector3 worldDir = rot * d.localDir;
                Vector3 sampleP = pos + worldDir * 0.8f;
                Vector3 u = field.Sample(sampleP);
                float resist = -Vector3.Dot(u, worldDir);
                if (resist < best)
                {
                    best = resist;
                    bestKey = d.key;
                }
            }

            return bestKey;
        }

        private static void Slider(ref float v, float min, float max, string label)
        {
            GUILayout.Label($"{label}: {v:F2}");
            v = GUILayout.HorizontalSlider(v, min, max);
        }

        private static void Toggle(ref bool v, string label)
        {
            v = GUILayout.Toggle(v, label);
        }
    }
}
