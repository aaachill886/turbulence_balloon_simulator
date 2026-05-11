using System;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace BalloonSim.Sim
{
    /// <summary>
    /// 训练数据记录器 - 记录完整的 27 点空间风场
    /// 输出格式: t(1) + pos(3) + vel(3) + wind_27×3(81) + grad(9) + config(11) = 108 列
    /// </summary>
    public class TrainingDataLogger : MonoBehaviour
    {
        [Header("References")]
        public SimulationConfig config;
        public BalloonState balloon;
        public TurbulenceField field;
        public ObservationBuffer observationBuffer;

        [Header("Settings")]
        public bool enableLogging = true;
        [Tooltip("采样间隔 (秒)")]
        public float sampleInterval = 0.08f;
        [Tooltip("输出目录")]
        public string outputDirectory = "training_data";
        [Tooltip("每N行 flush")]
        public int flushEvery = 50;
        [Tooltip("空间采样间距 (m)")]
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

        private void OnEnable()
        {
            if (!enableLogging) return;
            InitializeOutputDirectory();
            DetectNextEpisodeIndex();
            StartNewEpisode();
        }

        private void OnDisable()
        {
            CloseWriter();
        }

        private void OnDestroy()
        {
            CloseWriter();
        }

        private void FixedUpdate()
        {
            if (!enableLogging || _paused || _writer == null || balloon == null || field == null) return;

            _sampleTimer += Time.fixedDeltaTime;
            if (_sampleTimer < sampleInterval) return;
            _sampleTimer -= sampleInterval;

            WriteFrame();
        }

        public void ResumeLogging()
        {
            enableLogging = true;
            _paused = false;

            if (_writer == null)
            {
                InitializeOutputDirectory();
                DetectNextEpisodeIndex();
                StartNewEpisode();
            }
        }

        public void PauseLogging()
        {
            _paused = true;
            _writer?.Flush();
        }

        public void StartNewEpisode()
        {
            CloseWriter();

            _currentPath = Path.Combine(outputDirectory, $"episode_{_episodeIndex:D4}.csv");

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

            if (enableLogging)
                StartNewEpisode();
        }

        private void WriteFrame()
        {
            Vector3 pos = balloon.transform.position;
            Vector3 vel = balloon.velocity;

            var sb = new StringBuilder(1024);

            sb.Append(F(Time.time));

            sb.Append(','); sb.Append(F(pos.x));
            sb.Append(','); sb.Append(F(pos.y));
            sb.Append(','); sb.Append(F(pos.z));

            sb.Append(','); sb.Append(F(vel.x));
            sb.Append(','); sb.Append(F(vel.y));
            sb.Append(','); sb.Append(F(vel.z));

            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)
            {
                Vector3 samplePos = pos + new Vector3(dx, dy, dz) * sampleSpacing;
                Vector3 w = field.Sample(samplePos);
                sb.Append(','); sb.Append(F(w.x));
                sb.Append(','); sb.Append(F(w.y));
                sb.Append(','); sb.Append(F(w.z));
            }

            float[,] grad = field.SampleGradient(pos);
            for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                sb.Append(','); sb.Append(F(grad[i, j]));
            }

            sb.Append(','); sb.Append(F(config.beaufort));
            sb.Append(','); sb.Append(F(config.viscosity));
            sb.Append(','); sb.Append(F(config.reynolds));
            sb.Append(','); sb.Append(F(config.randomStrength));
            sb.Append(','); sb.Append(F(config.gustStrength));
            sb.Append(','); sb.Append(F(config.gustDirDeg));
            sb.Append(','); sb.Append(F(config.convectionStrength));
            sb.Append(','); sb.Append(config.tornado ? "1" : "0");
            sb.Append(','); sb.Append(F(config.wakeStrength));
            sb.Append(','); sb.Append(F(config.densityRatio));
            sb.Append(','); sb.Append(F(config.balloonRadius));

            _writer.WriteLine(sb.ToString());
            _lineCount++;

            if (_lineCount % Mathf.Max(1, flushEvery) == 0)
                _writer.Flush();
        }

        private string BuildHeader()
        {
            var sb = new StringBuilder(512);
            sb.Append("t,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z");

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

            sb.Append(",cfg_beaufort,cfg_viscosity,cfg_reynolds,cfg_randomStrength");
            sb.Append(",cfg_gustStrength,cfg_gustDirDeg,cfg_convectionStrength");
            sb.Append(",cfg_tornado,cfg_wakeStrength,cfg_densityRatio,cfg_balloonRadius");

            return sb.ToString();
        }

        private void InitializeOutputDirectory()
        {
            string fullPath = Path.Combine(Application.dataPath, "..", outputDirectory);
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
                string name = Path.GetFileNameWithoutExtension(f); // episode_0001
                if (name.StartsWith("episode_", StringComparison.OrdinalIgnoreCase))
                {
                    string idx = name.Substring("episode_".Length);
                    if (int.TryParse(idx, out int n) && n > maxIndex)
                        maxIndex = n;
                }
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

        private static string F(float v)
            => v.ToString("0.######", CultureInfo.InvariantCulture);
    }
}
