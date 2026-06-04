using System;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace BalloonSim.Sim
{
    public class TrainingDataLogger : MonoBehaviour
    {
        [Header("References")]
        public SimulationConfig config;
        public BalloonState balloon;
        public TurbulenceField field;
        public ObservationBuffer observationBuffer;
        public GameController game;
        public AutopilotController autopilot;

        public enum LogMode
        {
            TrainingData,
            LogsAI
        }

        [Header("Settings")]
        public bool enableLogging = true;
        public bool autoStartOnEnable = false;
        public float sampleInterval = 0.08f;
        [Tooltip("Select which dataset stream to write.")]
        public LogMode logMode = LogMode.TrainingData;
        [Tooltip("Relative folder under project root for model-training episodes.")]
        public string trainingDataDirectory = "training_data";
        [Tooltip("Relative folder under project root for assistant/analysis episodes.")]
        public string logsAIDirectory = "Logs_AI";
        [Tooltip("Resolved absolute output directory (auto-managed from log mode).")]
        public string outputDirectory = "Logs_AI";
        public int flushEvery = 50;
        public float sampleSpacing = 0.5f;

        private StreamWriter _writer;
        private int _episodeIndex;
        private int _lineCount;
        private float _sampleTimer;
        private string _currentPath;
        private bool _paused;

        public string OutputDirectory => outputDirectory;
        public string CurrentEpisodePath => _currentPath;
        public int EpisodeIndex => _episodeIndex;
        public int TotalFramesLogged => _lineCount;
        public bool IsPaused => _paused;
        public bool IsSessionOpen => _writer != null;
        public bool CaptureEnabled { get; private set; } = true;
        public string SamplePhase { get; private set; } = "warmup";

        private void OnEnable()
        {
            if (!enableLogging || !autoStartOnEnable) return;
            InitializeOutputDirectory();
            DetectNextEpisodeIndex();
        }

        private void OnDisable() => CloseWriter();
        private void OnDestroy() => CloseWriter();

        private void FixedUpdate()
        {
            if (!enableLogging || !CaptureEnabled || _paused || _writer == null || balloon == null || field == null) return;

            _sampleTimer += Time.fixedDeltaTime;
            if (_sampleTimer < sampleInterval) return;
            _sampleTimer -= sampleInterval;

            WriteFrame();
        }

        public void ResumeLogging()
        {
            enableLogging = true;
            _paused = false;
        }

        public void PauseLogging()
        {
            _paused = true;
            _writer?.Flush();
        }

        public void ResetEpisodeIndex()
        {
            _episodeIndex = 0;
        }

        public void CloseCurrentEpisode()
        {
            CloseWriter();
            _lineCount = 0;
            _sampleTimer = 0f;
        }

        public void SetCaptureEnabled(bool enabled)
        {
            CaptureEnabled = enabled;
            SamplePhase = enabled ? "stable" : "transition";
            if (!enabled)
                _writer?.Flush();
        }

        public void SetSamplePhase(string phase)
        {
            SamplePhase = string.IsNullOrWhiteSpace(phase) ? "transition" : phase;
        }

        public void StartNewEpisode()
        {
            if (!enableLogging || _paused || !CaptureEnabled || SamplePhase != "stable")
                return;

            CloseWriter();
            InitializeOutputDirectory();
            Directory.CreateDirectory(outputDirectory);
            _currentPath = Path.Combine(outputDirectory, $"episode_{_episodeIndex:D4}.csv");
            Directory.CreateDirectory(Path.GetDirectoryName(_currentPath));
            _writer = new StreamWriter(_currentPath, false, Encoding.UTF8);
            _writer.WriteLine(BuildHeader());
            _writer.Flush();
            _lineCount = 0;
            _sampleTimer = 0f;
            Debug.Log($"[TrainingDataLogger] Started episode {_episodeIndex} → {_currentPath}");
        }

        public void FlushEpisode()
        {
            if (_writer != null)
            {
                _writer.Flush();
                Debug.Log($"[TrainingDataLogger] Flushed episode {_episodeIndex}: {_lineCount} frames");
            }
            CloseWriter();
            _episodeIndex++;
            _lineCount = 0;
            _paused = false;
        }

        public void ClearAllTrainingData()
        {
            CloseWriter();

            if (Directory.Exists(outputDirectory))
            {
                var files = Directory.GetFiles(outputDirectory, "episode_*.csv");
                foreach (var f in files)
                {
                    try { File.Delete(f); } catch { }
                }
            }

            _episodeIndex = 0;
            _lineCount = 0;
            _sampleTimer = 0f;
            _paused = false;

            CaptureEnabled = false;
            SamplePhase = "warmup";
        }

        private void WriteFrame()
        {
            Vector3 pos = balloon.transform.position;

            var sb = new StringBuilder(1536);
            sb.Append(F(Time.time));
            sb.Append(','); sb.Append(SamplePhase);
            AppendVec(sb, pos);

            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)
            {
                Vector3 samplePos = pos + new Vector3(dx, dy, dz) * sampleSpacing;
                AppendVec(sb, field.Sample(samplePos));
            }

            float[,] grad = field.SampleGradient(pos);
            for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                sb.Append(','); sb.Append(F(grad[i, j]));
            }

            sb.Append(','); sb.Append(F(config.beaufort));
            sb.Append(','); sb.Append(F(config.gustStrength));
            sb.Append(','); sb.Append(F(config.viscosity));
            sb.Append(','); sb.Append(F(config.densityRatio));
            sb.Append(','); sb.Append(F(config.balloonRadius));

            _writer.WriteLine(sb.ToString());
            _lineCount++;

            if (_lineCount % Mathf.Max(1, flushEvery) == 0)
                _writer.Flush();
        }

        private string BuildHeader()
        {
            var sb = new StringBuilder(768);
            sb.Append("t,samplePhase,pos_x,pos_y,pos_z");

            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)
            {
                string tag = $"w_{dx + 1}{dy + 1}{dz + 1}";
                sb.Append($",{tag}_x,{tag}_y,{tag}_z");
            }

            for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                sb.Append($",grad_{i}{j}");

            sb.Append(",cfg_beaufort,cfg_gustStrength,cfg_viscosity,cfg_densityRatio,cfg_balloonRadius");
            return sb.ToString();
        }

        private void InitializeOutputDirectory()
        {
            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            string relative;

            // Stage2 capture is intentionally hard-routed to training_data.
            // Stage3 policy logging uses DataLogger, not TrainingDataLogger.
            logMode = LogMode.TrainingData;
            relative = string.IsNullOrWhiteSpace(trainingDataDirectory) ? "training_data" : trainingDataDirectory;

            string fullPath = Path.GetFullPath(Path.Combine(projectRoot, relative));
            Directory.CreateDirectory(fullPath);
            outputDirectory = fullPath;
        }

        private void DetectNextEpisodeIndex()
        {
            if (!Directory.Exists(outputDirectory))
            {
                _episodeIndex = 0;
                return;
            }

            int maxIndex = -1;
            var files = Directory.GetFiles(outputDirectory, "episode_*.csv");
            foreach (var f in files)
            {
                string name = Path.GetFileNameWithoutExtension(f);
                if (!name.StartsWith("episode_", StringComparison.OrdinalIgnoreCase)) continue;
                string idx = name.Substring("episode_".Length);
                if (int.TryParse(idx, out int n) && n > maxIndex)
                    maxIndex = n;
            }

            _episodeIndex = maxIndex + 1;
        }

        private void CloseWriter()
        {
            if (_writer != null)
            {
                try { _writer.Flush(); _writer.Dispose(); } catch { }
                _writer = null;
            }
        }

        private static void AppendVec(StringBuilder sb, Vector3 v)
        {
            sb.Append(','); sb.Append(F(v.x));
            sb.Append(','); sb.Append(F(v.y));
            sb.Append(','); sb.Append(F(v.z));
        }

        private static void AppendSixAxis(StringBuilder sb, Vector3 cmd)
        {
            sb.Append(','); sb.Append(F(Mathf.Max(0f, cmd.x)));
            sb.Append(','); sb.Append(F(Mathf.Max(0f, -cmd.x)));
            sb.Append(','); sb.Append(F(Mathf.Max(0f, cmd.y)));
            sb.Append(','); sb.Append(F(Mathf.Max(0f, -cmd.y)));
            sb.Append(','); sb.Append(F(Mathf.Max(0f, cmd.z)));
            sb.Append(','); sb.Append(F(Mathf.Max(0f, -cmd.z)));
        }

        private static string F(float v)
            => v.ToString("0.######", CultureInfo.InvariantCulture);
    }
}
