using System;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace BalloonSim.Sim
{
    public class DataLogger : MonoBehaviour
    {
        public SimulationConfig config;
        public BalloonState balloon;
        public TurbulenceField field;
        public AutopilotController autopilot;
        public GameController game;
        public BlockWorld world;
        public Transform waypoint;
        public ObservationBuffer observationBuffer;

        [Header("Logging")]
        public bool enabledLogging = false;
        public bool paused = false;
        public int flushEvery = 30;

        [Header("Stage 3 Metadata")]
        public bool enableStage3Columns = true;
        public int episodeId = -1;
        public int frameIndex = 0;
        private int _currentEpisodeId = -1;
        public string runId = "";
        [Tooltip("baseline / manual / exploration / policy")]
        public string mode = "stage3";
        [TextArea(2, 4)] public string actionJson = "";
        [TextArea(2, 4)] public string stateJson = "";
        public float reward = 0f;

        public string LogDirectory { get; private set; }
        public string CurrentLogPath { get; private set; }
        public int WrittenSamples => _n;

        private StreamWriter _w;
        private int _n;

        private void OnEnable()
        {
            if (!enabledLogging) return;
            StartLogging();
        }

        private void OnDisable()
        {
            StopLogging();
        }

        public void StartLogging()
        {
            if (_w != null) return;

            if (string.IsNullOrWhiteSpace(runId))
                runId = DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
            if (string.IsNullOrWhiteSpace(mode) || string.Equals(mode, "unknown", StringComparison.OrdinalIgnoreCase))
                mode = "stage3";

            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            LogDirectory = Path.GetFullPath(Path.Combine(projectRoot, "stage3 log"));
            Directory.CreateDirectory(LogDirectory);
            DetectNextEpisodeId();
            CurrentLogPath = Path.Combine(LogDirectory, $"balloon_log_{DateTime.Now:yyyyMMdd_HHmmss}.csv");

            _w = new StreamWriter(CurrentLogPath);
            _w.WriteLine(Header());
            _w.Flush();
            BeginEpisode();

            Debug.Log($"[DataLogger] Logging to: {CurrentLogPath}");
        }

        public void StopLogging()
        {
            try
            {
                _w?.Flush();
                _w?.Dispose();
            }
            catch { }
            _w = null;
        }

        public void PauseLogging(bool pause)
        {
            paused = pause;
        }

        public void ResumeLogging()
        {
            paused = false;
        }

        public void OpenLogFolder()
        {
            if (string.IsNullOrEmpty(LogDirectory) || !Directory.Exists(LogDirectory))
            {
                string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
                LogDirectory = Path.GetFullPath(Path.Combine(projectRoot, "stage3 log"));
                Directory.CreateDirectory(LogDirectory);
            }

            string url = "file:///" + LogDirectory.Replace("\\", "/");
            Application.OpenURL(url);
        }

        public void ClearAllLogs()
        {
            StopLogging();

            if (string.IsNullOrEmpty(LogDirectory) || !Directory.Exists(LogDirectory))
            {
                string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
                LogDirectory = Path.GetFullPath(Path.Combine(projectRoot, "stage3 log"));
                Directory.CreateDirectory(LogDirectory);
            }

            if (!Directory.Exists(LogDirectory)) return;

            string[] files = Directory.GetFiles(LogDirectory, "balloon_log_*.csv");
            foreach (var f in files)
            {
                try { File.Delete(f); } catch { }
            }

            episodeId = -1;
            _currentEpisodeId = -1;
            frameIndex = 0;
            Debug.Log($"[DataLogger] Cleared {files.Length} log files.");
            StartLogging();
        }

        private void FixedUpdate()
        {
            if (!enabledLogging || paused || _w == null || config == null || balloon == null) return;

            if (_currentEpisodeId < 0)
                BeginEpisode();
            if (enableStage3Columns && mode != "stage3" && mode != "manual" && mode != "baseline" && mode != "exploration" && mode != "policy")
                mode = "stage3";

            float t = Time.time;
            Vector3 pos = balloon.transform.position;
            Vector3 vel = balloon.velocity;
            Vector3 waypointPos = waypoint != null ? waypoint.position : Vector3.zero;
            float[,] g = field != null ? field.SampleGradient(pos) : new float[3, 3];
            int histCount = observationBuffer != null ? observationBuffer.Count : 0;

            if (enableStage3Columns)
            {
                stateJson = BuildStage3StateJson(pos, vel, waypointPos, g, histCount);
                Vector3 actionLabel = game != null
                    ? game.ComputeExpertResidual(game.PlayerIntentVel, game.PlayerIntentActive)
                    : Vector3.zero;
                actionJson = BuildStage3ActionJson(actionLabel);
                reward = ComputeStage3Reward(pos, vel, waypointPos, field != null ? field.Sample(pos) : Vector3.zero);
            }

            string line = string.Join(",",
                F(t),
                enableStage3Columns ? _currentEpisodeId.ToString(CultureInfo.InvariantCulture) : "",
                enableStage3Columns ? frameIndex.ToString(CultureInfo.InvariantCulture) : "",
                enableStage3Columns ? CsvEscape(runId) : "",
                enableStage3Columns ? CsvEscape(mode) : "",
                enableStage3Columns ? CsvEscape(actionJson) : "",
                enableStage3Columns ? CsvEscape(stateJson) : "",
                enableStage3Columns ? F(reward) : ""
            );
            frameIndex++;

            _w.WriteLine(line);
            _n++;
            if (_n % Mathf.Max(1, flushEvery) == 0)
                _w.Flush();
        }

        private static string Header()
        {
            return string.Join(",",
                "t",
                "episode_id",
                "frame_index",
                "run_id",
                "mode",
                "action_json",
                "state_json",
                "reward"
            );
        }

        private string BuildStage3StateJson(Vector3 pos, Vector3 vel, Vector3 waypointPos, float[,] g, int histCount)
        {
            Vector3 wind = field != null ? field.Sample(pos) : Vector3.zero;
            Vector3 wpDir = NormalizeOrZero(waypointPos - pos);
            var sb = new System.Text.StringBuilder(512);
            sb.Append('{');
            sb.Append("\"action_semantics\":\"bounded_residual_v2\",");
            sb.Append("\"vel\":"); AppendJsonVec(sb, vel); sb.Append(',');
            sb.Append("\"wind\":"); AppendJsonVec(sb, wind); sb.Append(',');
            sb.Append("\"intent\":"); AppendJsonVec(sb, game != null ? game.PlayerIntentVel : Vector3.zero); sb.Append(',');
            sb.Append("\"intent_active\":").Append(game != null && game.PlayerIntentActive ? "1" : "0").Append(',');
            sb.Append("\"grad\":"); AppendJsonArray(sb, new[] { g[0,0], g[0,1], g[0,2], g[1,0], g[1,1], g[1,2], g[2,0], g[2,1], g[2,2] }); sb.Append(',');
            sb.Append("\"hist_count\":").Append(histCount);
            sb.Append('}');
            return sb.ToString();
        }

        private static string BuildStage3ActionJson(Vector3 userTarget)
        {
            var sb = new System.Text.StringBuilder(128);
            sb.Append('{');
            sb.Append("\"action\":"); AppendJsonVec(sb, userTarget);
            sb.Append('}');
            return sb.ToString();
        }

        private static void AppendJsonVec(System.Text.StringBuilder sb, Vector3 v)
        {
            sb.Append('{');
            sb.Append("\"x\":").Append(F(v.x)).Append(',');
            sb.Append("\"y\":").Append(F(v.y)).Append(',');
            sb.Append("\"z\":").Append(F(v.z));
            sb.Append('}');
        }

        private static void AppendJsonArray(System.Text.StringBuilder sb, float[] arr)
        {
            sb.Append('[');
            for (int i = 0; i < arr.Length; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(F(arr[i]));
            }
            sb.Append(']');
        }

        private static Vector3 NormalizeOrZero(Vector3 v)
        {
            return v.sqrMagnitude > 1e-8f ? v.normalized : Vector3.zero;
        }

        private static float ComputeStage3Reward(Vector3 pos, Vector3 vel, Vector3 waypointPos, Vector3 wind)
        {
            float dist = Vector3.Distance(waypointPos, pos);
            float approach = -dist;
            float reach = dist < 1f ? 10f : 0f;
            float energy = -0.01f * vel.magnitude;
            float windAssist = Vector3.Dot(wind, NormalizeOrZero(waypointPos - pos));
            float windReward = 0.02f * Mathf.Max(0f, windAssist);
            return approach + reach + energy + windReward;
        }

        public void BeginEpisode()
        {
            _currentEpisodeId = Mathf.Max(0, episodeId + 1);
            episodeId = _currentEpisodeId;
            frameIndex = 0;
        }

        public void SetStage3Mode(string newMode)
        {
            mode = string.IsNullOrWhiteSpace(newMode) ? "stage3" : newMode;
        }

        private void DetectNextEpisodeId()
        {
            if (!Directory.Exists(LogDirectory))
            {
                episodeId = 0;
                return;
            }

            int maxEpisode = -1;
            foreach (var file in Directory.GetFiles(LogDirectory, "balloon_log_*.csv"))
            {
                try
                {
                    foreach (var line in File.ReadLines(file))
                    {
                        if (line.StartsWith("t,episode_id,"))
                            continue;
                        var parts = line.Split(',');
                        if (parts.Length < 2) continue;
                        if (int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int eid) && eid > maxEpisode)
                            maxEpisode = eid;
                    }
                }
                catch { }
            }

            episodeId = maxEpisode + 1;
        }

        private static string CsvEscape(string s)
        {
            if (string.IsNullOrEmpty(s)) return "";
            var escaped = s.Replace("\"", "\"\"");
            return $"\"{escaped}\"";
        }

        private static string F(float v)
            => v.ToString("0.######", CultureInfo.InvariantCulture);
    }
}
