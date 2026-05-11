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

        private bool _showBasic = true;
        private bool _showAdv;
        private Vector2 _basicScroll;
        private Vector2 _advScroll;
        private Vector2 _topScroll;
        private bool _confirmClearTraining;

        private void Awake()
        {
            if (logger == null)
                logger = FindObjectOfType<DataLogger>();
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

            GUILayout.BeginArea(new Rect(10, 10, 980, 300), GUI.skin.box);
            _topScroll = GUILayout.BeginScrollView(_topScroll, GUILayout.Width(960), GUILayout.Height(280));
            GUILayout.Label($"mode: {(config.aiEnabled ? "assist" : "manual")}  predictor:{(config.enableAIPredictor ? "ai" : "baseline")}");
            GUILayout.Label($"pos: {balloon.transform.position.x:F1}, {balloon.transform.position.y:F1}, {balloon.transform.position.z:F1}");
            GUILayout.Label($"vel: {balloon.velocity.x:F1}, {balloon.velocity.y:F1}, {balloon.velocity.z:F1}");
            GUILayout.Label($"yaw:{balloon.yawDeg:F1}°  fwd:{balloon.forward.x:F2},{balloon.forward.z:F2}");

            Vector3 u = field != null ? field.Sample(balloon.transform.position) : Vector3.zero;
            float turb = u.magnitude;
            string pred = autopilot != null && config.aiEnabled ? autopilot.PredErr.ToString("F2") : "-";
            string sig = autopilot != null && config.aiEnabled ? autopilot.PredSigma.ToString("F2") : "-";
            string thr = autopilot != null && config.aiEnabled ? autopilot.ThrottleNeed.ToString("F2") : "-";
            string model = (autopilot != null && autopilot.onnxPredictor != null) ? autopilot.onnxPredictor.RuntimeMode : "n/a";
            string ms = (autopilot != null && autopilot.onnxPredictor != null && model == "onnx") ? autopilot.onnxPredictor.LastInferenceMs.ToString("F2") : "-";
            string status = (autopilot != null && autopilot.onnxPredictor != null) ? autopilot.onnxPredictor.LastStatusMessage : "-";

            GUILayout.Label($"turb:{turb:F2}  Bft:{config.beaufort:F1}  rhoB/rhoA:{config.densityRatio:F2}  pred:{pred} sigma:{sig} throttle:{thr} model:{model} ms:{ms}");
            GUILayout.Label($"onnx-status: {status}");
            string holdMode = config.strictHoldNoDrift ? "benchmark" : "realistic-history";
            string oracle = config.assistUseOracleEnvCancellation ? "on" : "off";
            GUILayout.Label($"holdMode: {holdMode}  oracleEnv: {oracle}");
            GUILayout.Label($"fog:{(config.showVolumetricFog ? "on" : "off")} heat:{(config.showHeatmapPoints ? "on" : "off")} minimap:{(config.showMinimap ? "on" : "off")}");

            string hint = GetBestKeyHint();
            GUILayout.Label($"suggested low-resistance keys: {hint}");

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Manual")) config.aiEnabled = false;
            if (GUILayout.Button("Control Assist")) config.aiEnabled = true;
            if (GUILayout.Button(config.enableAIPredictor ? "Predictor: AI" : "Predictor: Baseline")) config.enableAIPredictor = !config.enableAIPredictor;
            if (GUILayout.Button("Reset (R)")) game?.ResetAll();
            if (GUILayout.Button("Tornado")) config.tornado = !config.tornado;
            GUILayout.EndHorizontal();

            var explorer = FindObjectOfType<RandomExplorationAgent>();
            var trainLog = FindObjectOfType<TrainingDataLogger>();
            if (explorer != null && trainLog != null)
            {
                GUILayout.BeginHorizontal();
                if (GUILayout.Button(explorer.enabled ? "Stop Exploration" : "Start Exploration"))
                {
                    explorer.enabled = !explorer.enabled;
                    if (explorer.enabled) trainLog.ResumeLogging();
                    else trainLog.PauseLogging();
                }

                if (GUILayout.Button(trainLog.IsPaused ? "Resume Training" : "Pause Training"))
                {
                    if (trainLog.IsPaused) trainLog.ResumeLogging();
                    else trainLog.PauseLogging();
                }

                if (GUILayout.Button("New Episode"))
                {
                    trainLog.FlushEpisode();
                    trainLog.StartNewEpisode();
                }

                if (GUILayout.Button(_confirmClearTraining ? "Confirm Clear Training" : "Clear Training Data"))
                {
                    if (!_confirmClearTraining) _confirmClearTraining = true;
                    else
                    {
                        trainLog.ClearAllTrainingData();
                        _confirmClearTraining = false;
                    }
                }

                if (GUILayout.Button("Cancel Clear")) _confirmClearTraining = false;

                GUILayout.Label($"Episode: {trainLog.EpisodeIndex} | Frames: {trainLog.TotalFramesLogged} | {(trainLog.IsPaused ? "paused" : "recording")}");
                GUILayout.EndHorizontal();
            }

            if (logger != null)
            {
                GUILayout.BeginHorizontal();
                if (GUILayout.Button(logger.paused ? "Resume Log" : "Pause Log", GUILayout.Height(24))) logger.PauseLogging(!logger.paused);
                if (GUILayout.Button("Open Log Folder", GUILayout.Height(24))) logger.OpenLogFolder();
                if (GUILayout.Button("Clear Logs", GUILayout.Height(24))) logger.ClearAllLogs();
                GUILayout.Label($"Log: {(logger.paused ? "paused" : "recording")}");
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
