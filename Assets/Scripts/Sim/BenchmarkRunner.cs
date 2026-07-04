using System;
using System.Collections;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace BalloonSim.Sim
{
    public class BenchmarkRunner : MonoBehaviour
    {
        [Header("References")]
        public SimulationConfig config;
        public GameController game;
        public BalloonState balloon;
        public TurbulenceField field;
        public AutopilotController autopilot;
        public DataLogger dataLogger;

        [Header("Benchmark")]
        public bool autoStartOnPlay = false;
        [Min(1)] public int episodes = 30;
        [Min(5f)] public float episodeDurationSec = 30f;
        [Min(0f)] public float settleTimeSec = 1.5f;
        public bool runAIThenBaseline = false;
        public bool disableSigmaConfidenceForAI = true;

        [Header("Fixed Setup")]
        public float fixedThrottle = 5.72f;
        public float fixedControlRate = 1.17f;
        public Vector3 fixedStartPos = new Vector3(8f, 8f, 8f);
        public Vector3 fixedStartVel = Vector3.zero;

        [Header("Output")]
        public string outputDirectory = "baseline_vs_ai";
        public string outputFilePrefix = "baseline_vs_ai";

        private bool _running;
        private string _runId;
        private int _completedTrials;
        private string _detailPath;
        private string _summaryPath;

        private struct TrialResult
        {
            public int episodeId;
            public string mode;
            public float meanPredMse;
            public float meanPredErr;
            public float meanPredSigma;
            public float meanThrottleNeed;
            public float meanCmdMag;
            public float meanPosErr;
            public float endPosErr;
        }

        private readonly System.Collections.Generic.List<TrialResult> _results = new();

        private void Awake()
        {
            AutoWireReferences();
        }

        private void Start()
        {
            if (autoStartOnPlay)
                StartBenchmark();
        }

        [ContextMenu("Start Benchmark")]
        public void StartBenchmark()
        {
            if (_running) return;
            AutoWireReferences();
            if (config == null || game == null || balloon == null || field == null || autopilot == null)
            {
                Debug.LogError("[BenchmarkRunner] Missing references.");
                return;
            }

            _running = true;
            _results.Clear();
            _completedTrials = 0;
            _runId = DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
            InitializeOutputFiles();
            StartCoroutine(RunBenchmarkCoroutine());
        }

        private IEnumerator RunBenchmarkCoroutine()
        {
            float oldThrottle = config.throttle;
            float oldControlRate = config.manualSpeedScale;
            bool oldSigma = config.usePredSigmaConfidence;
            bool oldAIPredictor = config.enableAIPredictor;
            bool oldAIEnabled = config.aiEnabled;

            config.throttle = fixedThrottle;
            config.manualSpeedScale = fixedControlRate;

            for (int ep = 0; ep < episodes; ep++)
            {
                if (runAIThenBaseline)
                {
                    yield return RunOneTrial(ep, true);
                    yield return RunOneTrial(ep, false);
                }
                else
                {
                    yield return RunOneTrial(ep, false);
                    yield return RunOneTrial(ep, true);
                }
            }

            config.throttle = oldThrottle;
            config.manualSpeedScale = oldControlRate;
            config.usePredSigmaConfidence = oldSigma;
            config.enableAIPredictor = oldAIPredictor;
            config.aiEnabled = oldAIEnabled;

            WriteOutputs();
            _running = false;
            Debug.Log($"[BenchmarkRunner] Completed {_completedTrials} trials. RunId={_runId}");
        }

        private IEnumerator RunOneTrial(int episodeId, bool aiMode)
        {
            PrepareEpisode(episodeId, aiMode);

            if (settleTimeSec > 0f)
                yield return new WaitForSeconds(settleTimeSec);

            float elapsed = 0f;
            int n = 0;
            float sumMse = 0f, sumErr = 0f, sumSigma = 0f, sumNeed = 0f, sumCmd = 0f, sumPosErr = 0f;

            while (elapsed < episodeDurationSec)
            {
                yield return new WaitForFixedUpdate();
                elapsed += Time.fixedDeltaTime;

                float predMse = autopilot.PredMSE;
                float predErr = autopilot.PredErr;
                float predSigma = autopilot.PredSigma;
                float need = autopilot.ThrottleNeed;
                float cmd = autopilot.LastCommandVel.magnitude;
                float posErr = (balloon.transform.position - fixedStartPos).magnitude;

                sumMse += predMse;
                sumErr += predErr;
                sumSigma += predSigma;
                sumNeed += need;
                sumCmd += cmd;
                sumPosErr += posErr;
                n++;
            }

            var r = new TrialResult
            {
                episodeId = episodeId,
                mode = aiMode ? "ai" : "baseline",
                meanPredMse = n > 0 ? sumMse / n : 0f,
                meanPredErr = n > 0 ? sumErr / n : 0f,
                meanPredSigma = n > 0 ? sumSigma / n : 0f,
                meanThrottleNeed = n > 0 ? sumNeed / n : 0f,
                meanCmdMag = n > 0 ? sumCmd / n : 0f,
                meanPosErr = n > 0 ? sumPosErr / n : 0f,
                endPosErr = (balloon.transform.position - fixedStartPos).magnitude
            };
            _results.Add(r);
            _completedTrials++;
            AppendDetailRow(r);

            Debug.Log($"[BenchmarkRunner] ep={episodeId:D3} mode={r.mode} mse={r.meanPredMse:F5} posErr={r.meanPosErr:F3} endErr={r.endPosErr:F3}");
        }

        private void AutoWireReferences()
        {
            if (config == null) config = FindObjectOfType<SimulationConfig>();
            if (game == null) game = FindObjectOfType<GameController>();
            if (balloon == null) balloon = FindObjectOfType<BalloonState>();
            if (field == null) field = FindObjectOfType<TurbulenceField>();
            if (autopilot == null) autopilot = FindObjectOfType<AutopilotController>();
            if (dataLogger == null) dataLogger = FindObjectOfType<DataLogger>();
        }

        private void PrepareEpisode(int episodeId, bool aiMode)
        {
            int seed = 10000 + episodeId;
            UnityEngine.Random.InitState(seed);

            game.ResetAll();
            field.Generate();

            balloon.transform.position = fixedStartPos;
            balloon.velocity = fixedStartVel;
            autopilot.ResetState();

            config.aiEnabled = true;
            config.enableAIPredictor = aiMode;
            config.usePredSigmaConfidence = aiMode ? !disableSigmaConfidenceForAI : true;

            if (dataLogger != null)
            {
                dataLogger.enabledLogging = true;
                dataLogger.StartLogging();
                dataLogger.PauseLogging(false);
            }
        }

        private void InitializeOutputFiles()
        {
            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            string dir = Path.GetFullPath(Path.Combine(projectRoot, outputDirectory));
            Directory.CreateDirectory(dir);

            _detailPath = Path.Combine(dir, $"{outputFilePrefix}_{_runId}_detail.csv");
            _summaryPath = Path.Combine(dir, $"{outputFilePrefix}_{_runId}_summary.csv");

            File.WriteAllText(_detailPath,
                "run_id,episode_id,mode,mean_pred_mse,mean_pred_err,mean_pred_sigma,mean_throttle_need,mean_cmd_mag,mean_pos_err,end_pos_err\n",
                Encoding.UTF8);

            Debug.Log($"[BenchmarkRunner] Detail stream started: {_detailPath}");
        }

        private void AppendDetailRow(TrialResult r)
        {
            if (string.IsNullOrWhiteSpace(_detailPath))
                return;

            string line = string.Join(",",
                _runId,
                r.episodeId.ToString(CultureInfo.InvariantCulture),
                r.mode,
                F(r.meanPredMse),
                F(r.meanPredErr),
                F(r.meanPredSigma),
                F(r.meanThrottleNeed),
                F(r.meanCmdMag),
                F(r.meanPosErr),
                F(r.endPosErr)
            ) + "\n";

            File.AppendAllText(_detailPath, line, Encoding.UTF8);
        }

        private void WriteOutputs()
        {
            if (!string.IsNullOrWhiteSpace(_summaryPath))
            {
                File.WriteAllText(_summaryPath, BuildSummaryCsv(), Encoding.UTF8);
                Debug.Log($"[BenchmarkRunner] Wrote summary: {_summaryPath}");
            }

            if (!string.IsNullOrWhiteSpace(_detailPath))
                Debug.Log($"[BenchmarkRunner] Detail already streamed: {_detailPath}");
        }

        private string BuildDetailCsv()
        {
            var sb = new StringBuilder(4096);
            sb.AppendLine("run_id,episode_id,mode,mean_pred_mse,mean_pred_err,mean_pred_sigma,mean_throttle_need,mean_cmd_mag,mean_pos_err,end_pos_err");
            foreach (var r in _results)
            {
                sb.Append(_runId).Append(',')
                  .Append(r.episodeId).Append(',')
                  .Append(r.mode).Append(',')
                  .Append(F(r.meanPredMse)).Append(',')
                  .Append(F(r.meanPredErr)).Append(',')
                  .Append(F(r.meanPredSigma)).Append(',')
                  .Append(F(r.meanThrottleNeed)).Append(',')
                  .Append(F(r.meanCmdMag)).Append(',')
                  .Append(F(r.meanPosErr)).Append(',')
                  .Append(F(r.endPosErr)).Append('\n');
            }
            return sb.ToString();
        }

        private string BuildSummaryCsv()
        {
            ComputeModeStats("baseline", out var bMse, out var bErr, out var bPos, out var bEnd);
            ComputeModeStats("ai", out var aMse, out var aErr, out var aPos, out var aEnd);

            var sb = new StringBuilder(1024);
            sb.AppendLine("run_id,episodes,episode_duration_sec,sigma_confidence_for_ai,baseline_mean_mse,ai_mean_mse,delta_mse,baseline_mean_pred_err,ai_mean_pred_err,delta_pred_err,baseline_mean_pos_err,ai_mean_pos_err,delta_pos_err,baseline_end_pos_err,ai_end_pos_err,delta_end_pos_err");
            sb.Append(_runId).Append(',')
              .Append(episodes).Append(',')
              .Append(F(episodeDurationSec)).Append(',')
              .Append(disableSigmaConfidenceForAI ? "off" : "on").Append(',')
              .Append(F(bMse)).Append(',').Append(F(aMse)).Append(',').Append(F(aMse - bMse)).Append(',')
              .Append(F(bErr)).Append(',').Append(F(aErr)).Append(',').Append(F(aErr - bErr)).Append(',')
              .Append(F(bPos)).Append(',').Append(F(aPos)).Append(',').Append(F(aPos - bPos)).Append(',')
              .Append(F(bEnd)).Append(',').Append(F(aEnd)).Append(',').Append(F(aEnd - bEnd)).Append('\n');
            return sb.ToString();
        }

        private void ComputeModeStats(string mode, out float meanMse, out float meanErr, out float meanPos, out float meanEnd)
        {
            int n = 0;
            float sMse = 0f, sErr = 0f, sPos = 0f, sEnd = 0f;
            foreach (var r in _results)
            {
                if (!string.Equals(r.mode, mode, StringComparison.OrdinalIgnoreCase)) continue;
                n++;
                sMse += r.meanPredMse;
                sErr += r.meanPredErr;
                sPos += r.meanPosErr;
                sEnd += r.endPosErr;
            }

            if (n == 0)
            {
                meanMse = meanErr = meanPos = meanEnd = 0f;
                return;
            }

            meanMse = sMse / n;
            meanErr = sErr / n;
            meanPos = sPos / n;
            meanEnd = sEnd / n;
        }

        private static string F(float v) => v.ToString("0.######", CultureInfo.InvariantCulture);
    }
}
