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
        public bool enabledLogging = true;
        public bool paused = false;
        public int flushEvery = 30;

        public string LogDirectory { get; private set; }
        public string CurrentLogPath { get; private set; }

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

            string baseDir = Directory.GetCurrentDirectory();
            if (string.IsNullOrEmpty(baseDir) || !Directory.Exists(baseDir))
                baseDir = Application.persistentDataPath;

            LogDirectory = Path.Combine(baseDir, "Logs_AI");
            Directory.CreateDirectory(LogDirectory);
            CurrentLogPath = Path.Combine(LogDirectory, $"balloon_log_{DateTime.Now:yyyyMMdd_HHmmss}.csv");

            _w = new StreamWriter(CurrentLogPath);
            _w.WriteLine(Header());
            _w.Flush();

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

        public void OpenLogFolder()
        {
            if (string.IsNullOrEmpty(LogDirectory))
                LogDirectory = Application.persistentDataPath;

            string url = "file:///" + LogDirectory.Replace("\\", "/");
            Application.OpenURL(url);
        }

        public void ClearAllLogs()
        {
            StopLogging();

            if (string.IsNullOrEmpty(LogDirectory))
                LogDirectory = Application.persistentDataPath;

            if (!Directory.Exists(LogDirectory)) return;

            string[] files = Directory.GetFiles(LogDirectory, "balloon_log_*.csv");
            foreach (var f in files)
            {
                try { File.Delete(f); } catch { }
            }

            Debug.Log($"[DataLogger] Cleared {files.Length} log files.");
            StartLogging();
        }

        private void FixedUpdate()
        {
            if (!enabledLogging || paused || _w == null || config == null || balloon == null) return;

            float t = Time.time;
            Vector3 pos = balloon.transform.position;
            Vector3 vel = balloon.velocity;

            bool userActive = game != null && game.UserActive;
            Vector3 userTarget = game != null ? game.UserTargetVel : Vector3.zero;

            Vector3 u = field != null ? field.Sample(pos) : Vector3.zero;
            Vector3 p = autopilot != null ? autopilot.Predicted : Vector3.zero;
            float predErr = autopilot != null ? autopilot.PredErr : 0f;
            float predSigma = autopilot != null ? autopilot.PredSigma : 1f;

            bool aiEnabled = config.aiEnabled;
            float throttleNeed = autopilot != null ? autopilot.ThrottleNeed : 0f;

            Int3 currentBlock = world != null ? world.CurrentBlock : new Int3(0, 0, 0);
            Vector3 waypointPos = waypoint != null ? waypoint.position : Vector3.zero;
            Int3 waypointBlock = world != null ? world.GetBlockOf(waypointPos) : new Int3(0, 0, 0);

            float[,] g = field != null ? field.SampleGradient(pos) : new float[3, 3];
            int histCount = observationBuffer != null ? observationBuffer.Count : 0;

            string line = string.Join(",",
                F(t),
                F(pos.x), F(pos.y), F(pos.z),
                F(vel.x), F(vel.y), F(vel.z),
                userActive ? "1" : "0",
                F(userTarget.x), F(userTarget.y), F(userTarget.z),
                F(u.x), F(u.y), F(u.z),
                F(p.x), F(p.y), F(p.z),
                F(predErr),
                F(predSigma),
                aiEnabled ? "1" : "0",
                F(throttleNeed),
                F(g[0,0]), F(g[0,1]), F(g[0,2]),
                F(g[1,0]), F(g[1,1]), F(g[1,2]),
                F(g[2,0]), F(g[2,1]), F(g[2,2]),
                histCount.ToString(),
                F(config.throttle),
                F(config.beaufort),
                F(config.viscosity),
                F(config.reynolds),
                F(config.randomStrength),
                F(config.gustStrength),
                F(config.gustDirDeg),
                F(config.convectionStrength),
                F(config.wakeStrength),
                F(config.balloonRadius),
                F(config.densityRatio),
                currentBlock.x.ToString(), currentBlock.y.ToString(), currentBlock.z.ToString(),
                F(waypointPos.x), F(waypointPos.y), F(waypointPos.z),
                waypointBlock.x.ToString(), waypointBlock.y.ToString(), waypointBlock.z.ToString()
            );

            _w.WriteLine(line);
            _n++;
            if (_n % Mathf.Max(1, flushEvery) == 0)
                _w.Flush();
        }

        private static string Header()
        {
            return string.Join(",",
                "t",
                "pos_x","pos_y","pos_z",
                "vel_x","vel_y","vel_z",
                "userActive",
                "userTarget_x","userTarget_y","userTarget_z",
                "obs_u_x","obs_u_y","obs_u_z",
                "pred_p_x","pred_p_y","pred_p_z",
                "predErr",
                "predSigma",
                "aiEnabled",
                "throttleNeed",
                "grad_00","grad_01","grad_02",
                "grad_10","grad_11","grad_12",
                "grad_20","grad_21","grad_22",
                "histCount",
                "cfg_throttle",
                "cfg_beaufort",
                "cfg_viscosity",
                "cfg_reynolds",
                "cfg_randomStrength",
                "cfg_gustStrength",
                "cfg_gustDirDeg",
                "cfg_convectionStrength",
                "cfg_wakeStrength",
                "cfg_balloonRadius",
                "cfg_densityRatio",
                "currentBlock_x","currentBlock_y","currentBlock_z",
                "waypointPos_x","waypointPos_y","waypointPos_z",
                "waypointBlock_x","waypointBlock_y","waypointBlock_z"
            );
        }

        private static string F(float v)
            => v.ToString("0.######", CultureInfo.InvariantCulture);
    }
}
