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
            string modeLabel = config.aiEnabled ? (config.enableAIPredictor ? "control-assist+predictor" : "control-assist") : "manual-true-flow";
            GUILayout.Label($"mode: {modeLabel}");
            GUILayout.Label($"pos: {balloon.transform.position.x:F1}, {balloon.transform.position.y:F1}, {balloon.transform.position.z:F1}");
            GUILayout.Label($"vel: {balloon.velocity.x:F1}, {balloon.velocity.y:F1}, {balloon.velocity.z:F1}");
            GUILayout.Label($"yaw:{balloon.yawDeg:F1}°  fwd:{balloon.forward.x:F2},{balloon.forward.z:F2}");

            Vector3 u = field != null ? field.Sample(balloon.transform.position) : Vector3.zero;
            float turb = u.magnitude;
            bool assistOn = config.aiEnabled;
            bool predictorOn = config.enableAIPredictor;
            string pred = autopilot != null && assistOn ? autopilot.PredErr.ToString("F2") : "-";
            string pmse = autopilot != null && assistOn ? autopilot.PredMSE.ToString("F4") : "-";
            string sig = autopilot != null && assistOn ? autopilot.PredSigma.ToString("F2") : "-";
            string thr = autopilot != null && assistOn ? autopilot.ThrottleNeed.ToString("F2") : "-";
            string model = (autopilot != null && autopilot.onnxPredictor != null) ? autopilot.onnxPredictor.RuntimeMode : "n/a";
            string ms = (autopilot != null && autopilot.onnxPredictor != null && model == "onnx") ? autopilot.onnxPredictor.LastInferenceMs.ToString("F2") : "-";
            string status = (autopilot != null && autopilot.onnxPredictor != null) ? autopilot.onnxPredictor.LastStatusMessage : "-";
            string modelSrc = (autopilot != null && autopilot.onnxPredictor != null) ? autopilot.onnxPredictor.LastModelSource : "n/a";
            string normSrc = (autopilot != null && autopilot.onnxPredictor != null) ? autopilot.onnxPredictor.LastNormSource : "n/a";
            Vector3 onnxMu = (autopilot != null && autopilot.onnxPredictor != null) ? autopilot.onnxPredictor.LastMu : Vector3.zero;
            float onnxAbsErr = (autopilot != null && autopilot.onnxPredictor != null) ? Vector3.Distance(u, onnxMu) : 0f;
            string stage2Line = $"Stage2 baseline hold: {((assistOn && !predictorOn) ? "on" : (assistOn ? "on+predictor" : "off"))} | predictor mode={model} status={status} ms={ms} modelSrc={modelSrc} normSrc={normSrc}";
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
            GUILayout.Label($"obs_u: {u.x:F3},{u.y:F3},{u.z:F3} | onnx_mu: {onnxMu.x:F3},{onnxMu.y:F3},{onnxMu.z:F3} | |err|:{onnxAbsErr:F4}");
            GUILayout.Label($"onnx-status: {status}");
            string holdMode = config.aiEnabled ? (config.strictHoldNoDrift ? "baseline-hold" : "assist-hold") : "manual";
            string oracle = config.assistUseOracleEnvCancellation ? "on" : "off";
            GUILayout.Label($"holdMode: {holdMode}  oracleEnv: {oracle}");
            GUILayout.Label($"fog:{(config.showVolumetricFog ? "on" : "off")} heat:{(config.showHeatmapPoints ? "on" : "off")} minimap:{(config.showMinimap ? "on" : "off")}");

            string hint = GetBestKeyHint();
            GUILayout.Label($"suggested low-resistance keys: {hint}");

            var explorer = FindObjectOfType<RandomExplorationAgent>();
            var trainLog = FindObjectOfType<TrainingDataLogger>();

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Reset (R)", GUILayout.Height(24))) game?.ResetAll();
            if (GUILayout.Button("Manual", GUILayout.Height(24))) config.aiEnabled = false;
            if (GUILayout.Button("Control Assist", GUILayout.Height(24))) config.aiEnabled = true;
            if (GUILayout.Button(config.enableAIPredictor ? "Stage2 Predictor: ON" : "Stage2 Predictor: OFF", GUILayout.Height(24))) config.enableAIPredictor = !config.enableAIPredictor;
            GUILayout.EndHorizontal();

            var thermo = balloon != null ? balloon.GetComponent<BalloonThermodynamics>() : null;
            if (explorer != null && trainLog != null)
            {
                GUILayout.Space(4);
                GUILayout.BeginVertical(GUI.skin.box);
                GUILayout.Label("Stage2 Capture Mode — strong mixed disturbance, auto exploration, writes training_data, no hover target");
                GUILayout.BeginHorizontal();
                if (GUILayout.Button(explorer.explorationEnabled ? "Stop Stage2 Capture" : "Start Stage2 Capture", GUILayout.Height(30)))
                {
                    if (!explorer.CaptureSessionRequested)
                        ApplyStage2CaptureMode(explorer, trainLog);
                    else
                        explorer.StopStage2Capture();
                }
                if (GUILayout.Button(trainLog.IsPaused ? "Resume Capture" : "Pause Capture", GUILayout.Height(30)))
                {
                    if (trainLog.IsPaused)
                        explorer.RequestStage2Capture(trainLog);
                    else
                        explorer.StopStage2Capture();
                }
                if (GUILayout.Button(_confirmClearStage2 ? "Confirm Clear" : "Clear training_data", GUILayout.Height(30)))
                {
                    if (!_confirmClearStage2) _confirmClearStage2 = true;
                    else
                    {
                        explorer.StopStage2Capture();
                        trainLog.logMode = TrainingDataLogger.LogMode.TrainingData;
                        trainLog.SetCaptureEnabled(false);
                        trainLog.ClearAllTrainingData();
                        _confirmClearStage2 = false;
                    }
                }
                if (GUILayout.Button("Cancel", GUILayout.Height(30))) _confirmClearStage2 = false;
                if (GUILayout.Button("Open training_data", GUILayout.Height(30))) OpenTrainingFolder(trainLog);
                GUILayout.EndHorizontal();
                GUILayout.Label($"Stage2 wind dataset → training_data | episode: {trainLog.EpisodeIndex} | frames: {trainLog.TotalFramesLogged} | {(trainLog.IsPaused ? "paused" : "recording")} | exploration={(explorer.explorationEnabled ? "on" : "off")} | pending={(explorer.Stage2CapturePending ? "yes" : "no")} | warmup={(thermo != null ? (thermo.IsWarmedUp ? "done" : "warming") : "n/a")}");
                GUILayout.EndVertical();
            }

            GUILayout.Space(4);
            GUILayout.BeginVertical(GUI.skin.box);
            GUILayout.Label("Flight Control Test Mode — strong disturbance + higher control authority, Stage2 predictor enabled, optional Stage3 policy");
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Apply Flight Test Mode", GUILayout.Height(30)))
                ApplyFlightControlTestMode(explorer, trainLog, policyRunner);
            if (GUILayout.Button(policyRunner != null && policyRunner.enablePolicy ? "Stage3 Policy: ON" : "Stage3 Policy: OFF", GUILayout.Height(30)) && policyRunner != null)
            {
                policyRunner.enablePolicy = !policyRunner.enablePolicy;
                policyRunner.Reinitialize();
            }
            if (GUILayout.Button("Stop Capture/Exploration", GUILayout.Height(30)))
            {
                if (explorer != null) explorer.StopStage2Capture();
                if (trainLog != null) trainLog.PauseLogging();
            }
            if (GUILayout.Button("Start Stage3 Policy Log", GUILayout.Height(30)) && logger != null)
                StartOrResumeStage3Log(explorer);
            GUILayout.EndHorizontal();
            GUILayout.Label($"Flight test: authority throttle={config.throttle:F2} controlRate={config.manualSpeedScale:F2} | Stage2 predictor={(config.enableAIPredictor ? "on" : "off")} | Stage3 policy={(policyRunner != null && policyRunner.enablePolicy ? "on" : "off")}");
            GUILayout.EndVertical();

            if (logger != null)
            {
                GUILayout.BeginHorizontal();
                if (GUILayout.Button(logger.enabledLogging ? (logger.paused ? "Resume Stage3" : "Pause Stage3") : "Start Stage3 Log", GUILayout.Height(24)))
                {
                    if (!logger.enabledLogging)
                        StartOrResumeStage3Log(explorer);
                    else
                        logger.PauseLogging(!logger.paused);
                }
                if (GUILayout.Button("Open Log Folder", GUILayout.Height(24))) logger.OpenLogFolder();
                if (GUILayout.Button("Clear Logs", GUILayout.Height(24))) logger.ClearAllLogs();
                string logState = !logger.enabledLogging ? "off" : (logger.paused ? "paused" : "recording");
                GUILayout.Label($"Log: {logState}");
                GUILayout.EndHorizontal();
            }

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

        private void ApplyStage2CaptureMode(RandomExplorationAgent explorer, TrainingDataLogger trainLog)
        {
            ApplyStrongMixedDisturbance();
            config.aiEnabled = false;
            config.enableAIPredictor = false;
            config.throttle = 1.0f;
            config.manualSpeedScale = 0.22f;

            explorer.includeReleaseHoldCycles = true;
            explorer.RequestStage2Capture(trainLog);
        }

        private void ApplyFlightControlTestMode(RandomExplorationAgent explorer, TrainingDataLogger trainLog, Stage3PolicyRunner policyRunner)
        {
            ApplyStrongMixedDisturbance();
            config.aiEnabled = true;
            config.enableAIPredictor = true;
            config.throttle = 4.0f;
            config.manualSpeedScale = 1.2f;
            config.densityRatio = 1.0f;
            config.holdPosK = 2.4f;
            config.holdVelK = 2.2f;
            config.holdMaxSpeed = 8.0f;
            config.holdEnvComp = 0.35f;
            config.holdAltIK = 1.2f;

            if (explorer != null)
            {
                explorer.StopStage2Capture();
                explorer.trainingLogger = trainLog;
            }
            if (trainLog != null)
            {
                trainLog.SetCaptureEnabled(false);
                trainLog.PauseLogging();
            }
            if (logger != null) logger.PauseLogging(true);
            ConfigureThermodynamicsForNeutralFlight();
            game?.ResetAll();
            ApplyStrongMixedDisturbance();
            config.aiEnabled = true;
            config.enableAIPredictor = true;
            config.throttle = 4.0f;
            config.manualSpeedScale = 1.2f;
            config.densityRatio = 1.0f;
            if (policyRunner != null)
            {
                policyRunner.enablePolicy = false;
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
            config.aiEnabled = true;
            config.enableAIPredictor = true;
            logger.mode = "stage3_with_stage2_predictor";
            logger.runId = string.IsNullOrWhiteSpace(logger.runId) ? System.DateTime.Now.ToString("yyyyMMdd_HHmmss") : logger.runId;
            logger.episodeId = logger.episodeId < 0 ? 0 : logger.episodeId;
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
