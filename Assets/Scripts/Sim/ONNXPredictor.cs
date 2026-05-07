using System;
using System.Diagnostics;
using UnityEngine;

#if ENABLE_SENTIS
using Unity.Sentis;
#endif

namespace BalloonSim.Sim
{
    public class ONNXPredictor : MonoBehaviour
    {
        [Header("Runtime")]
        public bool enableONNX = false;
        public bool fallbackToHeuristicWhenUnavailable = true;
        public bool verboseDiagnostics = true;

        [Header("Model")]
        public UnityEngine.Object onnxModelAsset;
        public string muOutputName = "mu";
        public string logVarOutputName = "logvar";

        [Header("Output Auto-Detect")]
        public bool autoDetectOutputNames = true;
        [Tooltip("Comma separated candidates, first match wins")]
        public string muCandidates = "mu,mean,pred,velocity,output";
        [Tooltip("Comma separated candidates, first match wins")]
        public string logVarCandidates = "logvar,log_var,var,variance,sigma,std";

        [Header("Input Spec")]
        [Tooltip("Model history length K. Input will be left-padded if history < K.")]
        public int modelHistoryLength = 8;
        [Tooltip("Model spatial sample count N. Current ObservationBuffer uses 27 (3x3x3).")]
        public int modelSpatialSamples = 27;

        [Header("Heuristic fallback")]
        [Range(0f, 2f)] public float trendGain = 0.28f;
        [Range(0.1f, 3f)] public float sigmaScale = 1f;

        public bool IsModelAvailable { get; private set; }
        public Vector3 LastMu { get; private set; }
        public float LastSigma { get; private set; } = 1f;
        public float LastInferenceMs { get; private set; }
        public string RuntimeMode { get; private set; } = "off";
        public string LastStatusMessage { get; private set; } = "init";

#if ENABLE_SENTIS
        private Model _model;
        private IWorker _worker;
#endif

        private void Awake()
        {
            TryInitModel();
        }

        private void OnDestroy()
        {
#if ENABLE_SENTIS
            _worker?.Dispose();
            _worker = null;
#endif
        }

        [ContextMenu("Validate ONNX Bindings")]
        public void ValidateBindings()
        {
            if (!enableONNX)
            {
                LastStatusMessage = "ONNX disabled";
                LogDiag(LastStatusMessage);
                return;
            }

#if ENABLE_SENTIS
            if (!IsModelAvailable || _worker == null)
            {
                LastStatusMessage = "Model runtime unavailable";
                LogDiag(LastStatusMessage);
                return;
            }

            int k = Mathf.Max(1, modelHistoryLength);
            int n = Mathf.Max(1, modelSpatialSamples);
            int c = 3;

            try
            {
                using var test = new TensorFloat(new TensorShape(1, k, n, c), new float[k * n * c]);
                _worker.Schedule(test);

                if (autoDetectOutputNames)
                    AutoResolveOutputNames();

                var muTensor = _worker.PeekOutput(muOutputName) as TensorFloat;
                var lvTensor = _worker.PeekOutput(logVarOutputName) as TensorFloat;

                if (muTensor == null || lvTensor == null)
                {
                    LastStatusMessage = $"Output missing: mu='{muOutputName}' or logvar='{logVarOutputName}'";
                    LogDiag(LastStatusMessage);
                    return;
                }

                LastStatusMessage = $"OK input=(1,{k},{n},{c}) mu='{muOutputName}'({muTensor.length}) logvar='{logVarOutputName}'({lvTensor.length})";
                LogDiag(LastStatusMessage);
            }
            catch (Exception e)
            {
                LastStatusMessage = "Validate failed: " + e.Message;
                LogDiag(LastStatusMessage);
            }
#else
            LastStatusMessage = "ENABLE_SENTIS not defined";
            LogDiag(LastStatusMessage);
#endif
        }

        private void TryInitModel()
        {
            IsModelAvailable = false;
            RuntimeMode = enableONNX ? "fallback" : "off";

#if ENABLE_SENTIS
            if (!enableONNX || onnxModelAsset == null)
            {
                LastStatusMessage = !enableONNX ? "ONNX disabled" : "ModelAsset not assigned";
                return;
            }

            if (onnxModelAsset is not ModelAsset ma)
            {
                LastStatusMessage = "Assigned object is not Sentis ModelAsset";
                LogDiag(LastStatusMessage);
                return;
            }

            _model = ModelLoader.Load(ma);
            _worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, _model);
            IsModelAvailable = _worker != null;
            RuntimeMode = IsModelAvailable ? "onnx" : "fallback";
            LastStatusMessage = IsModelAvailable ? "Model loaded" : "Worker create failed";
            LogDiag(LastStatusMessage);

            if (IsModelAvailable) ValidateBindings();
#else
            if (enableONNX)
            {
                LastStatusMessage = "ENABLE_SENTIS not defined, using fallback";
                LogDiag(LastStatusMessage);
            }
#endif
        }

        public bool TryPredict(ObservationBuffer.ObservationFrame[] history, out Vector3 mu, out float sigma)
        {
            mu = Vector3.zero;
            sigma = 1f;
            LastInferenceMs = 0f;

            if (!enableONNX)
            {
                RuntimeMode = "off";
                return false;
            }

#if ENABLE_SENTIS
            if (IsModelAvailable && _worker != null)
            {
                if (history == null || history.Length == 0)
                    return false;

                int k = Mathf.Max(1, modelHistoryLength);
                int n = Mathf.Max(1, modelSpatialSamples);
                int c = 3;

                float[] input = new float[1 * k * n * c];
                FillInput(history, input, k, n);

                var sw = Stopwatch.StartNew();
                using var x = new TensorFloat(new TensorShape(1, k, n, c), input);
                _worker.Schedule(x);

                if (autoDetectOutputNames)
                    AutoResolveOutputNames();

                var muTensor = _worker.PeekOutput(muOutputName) as TensorFloat;
                var lvTensor = _worker.PeekOutput(logVarOutputName) as TensorFloat;
                sw.Stop();
                LastInferenceMs = (float)sw.Elapsed.TotalMilliseconds;

                if (muTensor == null || lvTensor == null)
                {
                    LastStatusMessage = "Output tensor missing, fallback used";
                    RuntimeMode = "fallback";
                    return false;
                }

                float mx = muTensor[0];
                float my = muTensor.length > 1 ? muTensor[1] : 0f;
                float mz = muTensor.length > 2 ? muTensor[2] : 0f;

                float lvx = lvTensor[0];
                float lvy = lvTensor.length > 1 ? lvTensor[1] : lvx;
                float lvz = lvTensor.length > 2 ? lvTensor[2] : lvx;

                float sv = Mathf.Sqrt(Mathf.Exp((lvx + lvy + lvz) / 3f));
                mu = new Vector3(mx, my, mz);
                sigma = Mathf.Max(1e-4f, sv);

                LastMu = mu;
                LastSigma = sigma;
                RuntimeMode = "onnx";
                LastStatusMessage = "ONNX inference OK";
                return true;
            }
#endif

            if (!fallbackToHeuristicWhenUnavailable)
                return false;

            if (history == null || history.Length < 2)
                return false;

            int nHist = history.Length;
            int s = Mathf.Max(0, nHist - 5);
            float[] w = { 0.08f, 0.14f, 0.2f, 0.28f, 0.4f };

            Vector3 p = Vector3.zero;
            float wsSum = 0f;
            for (int i = s; i < nHist; i++)
            {
                int wi = Mathf.Min(i - s, 4);
                Vector3 centerWind = history[i].windSamples != null && history[i].windSamples.Length > 13
                    ? history[i].windSamples[13]
                    : Vector3.zero;
                p += centerWind * w[wi];
                wsSum += w[wi];
            }

            if (wsSum > 1e-6f) p /= wsSum;

            Vector3 last = history[nHist - 1].windSamples != null && history[nHist - 1].windSamples.Length > 13
                ? history[nHist - 1].windSamples[13]
                : Vector3.zero;
            Vector3 prev = history[nHist - 2].windSamples != null && history[nHist - 2].windSamples.Length > 13
                ? history[nHist - 2].windSamples[13]
                : Vector3.zero;
            p += (last - prev) * trendGain;

            float var = 0f;
            for (int i = s; i < nHist; i++)
            {
                Vector3 centerWind = history[i].windSamples != null && history[i].windSamples.Length > 13
                    ? history[i].windSamples[13]
                    : Vector3.zero;
                Vector3 d = centerWind - p;
                var += d.sqrMagnitude;
            }
            var /= Mathf.Max(1, nHist - s);

            mu = p;
            sigma = Mathf.Sqrt(var + 1e-6f) * sigmaScale;
            LastMu = mu;
            LastSigma = sigma;
            RuntimeMode = "fallback";
            LastStatusMessage = "Heuristic fallback used";
            return true;
        }

        private void FillInput(ObservationBuffer.ObservationFrame[] history, float[] input, int k, int n)
        {
            int startHist = Mathf.Max(0, history.Length - k);
            int padCount = k - (history.Length - startHist);
            int idx = 0;

            for (int t = 0; t < k; t++)
            {
                int src = t < padCount ? startHist : startHist + (t - padCount);
                src = Mathf.Clamp(src, 0, history.Length - 1);
                var ws = history[src].windSamples;

                for (int i = 0; i < n; i++)
                {
                    Vector3 v = (ws != null && ws.Length > i) ? ws[i] : Vector3.zero;
                    input[idx++] = v.x;
                    input[idx++] = v.y;
                    input[idx++] = v.z;
                }
            }
        }

#if ENABLE_SENTIS
        private void AutoResolveOutputNames()
        {
            string resolvedMu = ResolveFirstExistingOutputName(muOutputName, muCandidates);
            string resolvedLv = ResolveFirstExistingOutputName(logVarOutputName, logVarCandidates);

            if (!string.IsNullOrEmpty(resolvedMu) && resolvedMu != muOutputName)
            {
                muOutputName = resolvedMu;
                LogDiag($"Auto-detected mu output: {muOutputName}");
            }
            if (!string.IsNullOrEmpty(resolvedLv) && resolvedLv != logVarOutputName)
            {
                logVarOutputName = resolvedLv;
                LogDiag($"Auto-detected logvar output: {logVarOutputName}");
            }
        }

        private string ResolveFirstExistingOutputName(string primary, string candidatesCsv)
        {
            if (TryHasOutput(primary)) return primary;

            if (string.IsNullOrWhiteSpace(candidatesCsv)) return null;
            string[] cands = candidatesCsv.Split(',');
            foreach (string raw in cands)
            {
                string name = raw.Trim();
                if (string.IsNullOrEmpty(name)) continue;
                if (TryHasOutput(name)) return name;
            }
            return null;
        }

        private bool TryHasOutput(string outputName)
        {
            try
            {
                var t = _worker.PeekOutput(outputName);
                return t != null;
            }
            catch
            {
                return false;
            }
        }
#endif

        private void LogDiag(string msg)
        {
            if (verboseDiagnostics)
                UnityEngine.Debug.Log("[ONNXPredictor] " + msg);
        }
    }
}
