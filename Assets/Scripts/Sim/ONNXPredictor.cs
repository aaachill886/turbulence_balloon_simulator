using System;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using UnityEngine;

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
        [Tooltip("StreamingAssets subfolder that contains policy_meta.json / policy_norm.json.")]
        public string streamingAssetsStage3Folder = "Stage3";
        [Tooltip("Optional Resources path for the imported Sentis ModelAsset (without extension).")]
        public string modelResourcePath = "wind_predictor";
        public string muOutputName = "mu";
        public string logVarOutputName = "logvar";

        [Header("Output Auto-Detect")]
        public bool autoDetectOutputNames = true;
        public bool verboseInferenceDiagnostics = false;

        [Header("Input Spec")]
        public int modelHistoryLength = 8;
        public int modelSpatialSamples = 27;

        [Header("Heuristic fallback")]
        [Range(0f, 2f)] public float trendGain = 0.28f;
        [Range(0.1f, 3f)] public float sigmaScale = 1f;

        [Header("Normalization")]
        public bool useNormalization = true;
        public string normParamsPath = "policy_norm.json";

        public bool IsModelAvailable { get; private set; }
        public Vector3 LastMu { get; private set; }
        public float LastSigma { get; private set; } = 1f;
        public float LastInferenceMs { get; private set; }
        public string RuntimeMode { get; private set; } = "off";
        public string LastStatusMessage { get; private set; } = "init";
        public string LastModelSource { get; private set; } = "n/a";
        public string LastNormSource { get; private set; } = "n/a";

        private Vector3 _windMean = Vector3.zero;
        private Vector3 _windStd = Vector3.one;

        // Sentis reflection handles
        private bool _sentisAvailable;
        private Type _modelAssetType;
        private Type _modelLoaderType;
        private Type _workerFactoryType;
        private Type _backendTypeType;
        private Type _tensorFloatType;
        private Type _tensorShapeType;
        private object _model;
        private object _worker;

        private void Awake()
        {
            Reinitialize();
        }

        public void Reinitialize()
        {
            DisposeWorker();
            _model = null;
            IsModelAvailable = false;
            LoadNormalizationParams();
            ProbeSentis();
            TryInitModel();
        }

        private void OnDestroy()
        {
            DisposeWorker();
        }

        private void ProbeSentis()
        {
            _modelAssetType = ResolveType("Unity.Sentis.ModelAsset");
            _modelLoaderType = ResolveType("Unity.Sentis.ModelLoader");
            _workerFactoryType = ResolveType("Unity.Sentis.Worker") ?? ResolveType("Unity.Sentis.WorkerFactory");
            _backendTypeType = ResolveType("Unity.Sentis.BackendType");
            _tensorFloatType = ResolveType("Unity.Sentis.Tensor`1")?.MakeGenericType(typeof(float)) ?? ResolveType("Unity.Sentis.Tensor") ?? ResolveType("Unity.Sentis.TensorFloat");
            _tensorShapeType = ResolveType("Unity.Sentis.TensorShape");

            _sentisAvailable = _modelAssetType != null && _modelLoaderType != null && _workerFactoryType != null &&
                               _backendTypeType != null && _tensorFloatType != null && _tensorShapeType != null;

            if (!_sentisAvailable)
            {
                string sentisAssemblies = string.Join(", ", AppDomain.CurrentDomain.GetAssemblies()
                    .Select(a => a.GetName().Name)
                    .Where(n => n.IndexOf("Sentis", StringComparison.OrdinalIgnoreCase) >= 0));
                LogDiag($"Sentis reflection probe failed; fallback mode only. Assemblies=[{sentisAssemblies}] " +
                        $"ModelAsset={_modelAssetType != null} ModelLoader={_modelLoaderType != null} " +
                        $"WorkerFactory={_workerFactoryType != null} BackendType={_backendTypeType != null} " +
                        $"TensorFloat={_tensorFloatType != null} TensorShape={_tensorShapeType != null}");
            }
        }

        private static Type ResolveType(string fullName)
        {
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                var type = asm.GetType(fullName, false);
                if (type != null)
                    return type;
            }
            return null;
        }

        private void LoadNormalizationParams()
        {
            if (!useNormalization)
            {
                _windMean = Vector3.zero;
                _windStd = Vector3.one;
                LastNormSource = "disabled";
                return;
            }

            string stage3Root = System.IO.Path.Combine(Application.streamingAssetsPath, string.IsNullOrWhiteSpace(streamingAssetsStage3Folder) ? "Stage3" : streamingAssetsStage3Folder);
            string normPath = System.IO.Path.Combine(stage3Root, normParamsPath);
            if (System.IO.File.Exists(normPath))
            {
                try
                {
                    TryParseNormJson(System.IO.File.ReadAllText(normPath), normPath);
                    LastNormSource = normPath;
                    return;
                }
                catch (Exception e)
                {
                    LogDiag("[WARN] Stage3 norm parse failed: " + e.Message);
                }
            }

            var textAsset = Resources.Load<TextAsset>("norm_params");
            if (textAsset != null)
            {
                TryParseNormJson(textAsset.text, "Resources/norm_params");
                LastNormSource = "Resources/norm_params";
                return;
            }

            _windMean = Vector3.zero;
            _windStd = Vector3.one;
            LastNormSource = "identity";
        }

        [Serializable]
        private class NormParams
        {
            public float[] wind_mean;
            public float[] wind_std;
            public float[] state_mean;
            public float[] state_std;
        }

        private void TryParseNormJson(string json, string source)
        {
            var data = JsonUtility.FromJson<NormParams>(json);
            if (data == null) throw new Exception("json parse returned null");

            // Stage2 format: wind_mean/wind_std
            if (data.wind_mean != null && data.wind_mean.Length >= 3)
                _windMean = new Vector3(data.wind_mean[0], data.wind_mean[1], data.wind_mean[2]);
            else
                _windMean = Vector3.zero;

            if (data.wind_std != null && data.wind_std.Length >= 3)
                _windStd = new Vector3(Mathf.Max(1e-6f, data.wind_std[0]), Mathf.Max(1e-6f, data.wind_std[1]), Mathf.Max(1e-6f, data.wind_std[2]));
            else
                _windStd = Vector3.one;

            LogDiag($"Loaded norm params from {source}: mean={_windMean}, std={_windStd}");
        }

        private void TryInitModel()
        {
            IsModelAvailable = false;
            RuntimeMode = enableONNX ? "fallback" : "off";

            if (!enableONNX)
            {
                LastStatusMessage = "ONNX disabled";
                return;
            }

            if (onnxModelAsset == null && !string.IsNullOrWhiteSpace(modelResourcePath))
            {
                onnxModelAsset = LoadSentisModelAssetFromResources(modelResourcePath);
                if (onnxModelAsset != null)
                {
                    LastModelSource = $"Resources/{modelResourcePath}";
                    LogDiag($"Loaded model asset candidate from {LastModelSource}: type={onnxModelAsset.GetType().FullName}, name={onnxModelAsset.name}");
                }
                else
                {
                    LogDiag($"Unable to load Sentis ModelAsset from Resources/{modelResourcePath}");
                }
            }

            if (onnxModelAsset == null)
            {
                LastStatusMessage = "ModelAsset not assigned";
                return;
            }

            if (!_sentisAvailable)
            {
                LastStatusMessage = "Sentis unavailable, using fallback";
                LogDiag(LastStatusMessage);
                return;
            }

            if (!_modelAssetType.IsInstanceOfType(onnxModelAsset))
            {
                LogDiag($"Assigned object is not Sentis ModelAsset: actual={onnxModelAsset.GetType().FullName}, expected={_modelAssetType.FullName}; retrying Resources/{modelResourcePath}");
                var resolved = LoadSentisModelAssetFromResources(modelResourcePath);
                if (resolved != null && _modelAssetType.IsInstanceOfType(resolved))
                {
                    onnxModelAsset = resolved;
                    LastModelSource = $"Resources/{modelResourcePath}";
                    LogDiag($"Resolved Sentis ModelAsset from Resources/{modelResourcePath}: type={onnxModelAsset.GetType().FullName}, name={onnxModelAsset.name}");
                }
                else
                {
                    LastStatusMessage = $"Assigned object is not Sentis ModelAsset: actual={onnxModelAsset.GetType().FullName}, expected={_modelAssetType.FullName}";
                    LogDiag(LastStatusMessage);
                    return;
                }
            }

            try
            {
                var loadMethod = _modelLoaderType.GetMethod("Load", BindingFlags.Public | BindingFlags.Static, null, new[] { _modelAssetType }, null);
                if (loadMethod == null)
                {
                    LastStatusMessage = "ModelLoader.Load(ModelAsset) not found";
                    LogDiag(LastStatusMessage);
                    return;
                }

                _model = loadMethod.Invoke(null, new[] { onnxModelAsset });
                LogDiag($"Sentis model loaded: type={_model?.GetType().FullName ?? "null"}");

                _worker = CreateSentisWorker(_model);
                if (_worker == null)
                {
                    LastStatusMessage = "Sentis worker creation failed";
                    return;
                }

                IsModelAvailable = _worker != null;
                RuntimeMode = IsModelAvailable ? "onnx" : "fallback";
                LastStatusMessage = IsModelAvailable ? "Model loaded" : "Worker create failed";
                LastModelSource = onnxModelAsset.name;
                LogDiag($"Sentis init result: available={IsModelAvailable}, worker={_worker?.GetType().FullName ?? "null"}, status={LastStatusMessage}");
            }
            catch (Exception e)
            {
                IsModelAvailable = false;
                RuntimeMode = "fallback";
                LastStatusMessage = "Sentis init failed: " + e.Message;
                LogDiag(LastStatusMessage);
            }
        }

        private UnityEngine.Object LoadSentisModelAssetFromResources(string resourcePath)
        {
            UnityEngine.Object loaded = Resources.Load(resourcePath);
            if (loaded != null && _modelAssetType != null && _modelAssetType.IsInstanceOfType(loaded))
                return loaded;

            if (loaded != null)
                LogDiag($"Resources.Load('{resourcePath}') returned {loaded.GetType().FullName}; trying imported sub-assets");

            UnityEngine.Object[] all = Resources.LoadAll(resourcePath);
            foreach (var asset in all)
            {
                if (asset == null) continue;
                LogDiag($"Resources.LoadAll('{resourcePath}') candidate: type={asset.GetType().FullName}, name={asset.name}");
                if (_modelAssetType != null && _modelAssetType.IsInstanceOfType(asset))
                    return asset;
            }

            string dir = System.IO.Path.GetDirectoryName(resourcePath)?.Replace('\\', '/');
            string file = System.IO.Path.GetFileName(resourcePath);
            UnityEngine.Object[] dirAssets = Resources.LoadAll(string.IsNullOrWhiteSpace(dir) ? string.Empty : dir);
            foreach (var asset in dirAssets)
            {
                if (asset == null || asset.name != file) continue;
                LogDiag($"Resources.LoadAll('{dir}') named candidate: type={asset.GetType().FullName}, name={asset.name}");
                if (_modelAssetType != null && _modelAssetType.IsInstanceOfType(asset))
                    return asset;
            }

#if UNITY_EDITOR
            string assetPath = $"Assets/Resources/{resourcePath}.onnx";
            UnityEngine.Object editorAsset = UnityEditor.AssetDatabase.LoadAssetAtPath(assetPath, _modelAssetType ?? typeof(UnityEngine.Object));
            if (editorAsset != null)
            {
                LogDiag($"AssetDatabase.LoadAssetAtPath('{assetPath}') candidate: type={editorAsset.GetType().FullName}, name={editorAsset.name}");
                if (_modelAssetType != null && _modelAssetType.IsInstanceOfType(editorAsset))
                    return editorAsset;
            }

            foreach (var asset in UnityEditor.AssetDatabase.LoadAllAssetsAtPath(assetPath))
            {
                if (asset == null) continue;
                LogDiag($"AssetDatabase.LoadAllAssetsAtPath('{assetPath}') candidate: type={asset.GetType().FullName}, name={asset.name}");
                if (_modelAssetType != null && _modelAssetType.IsInstanceOfType(asset))
                    return asset;
            }
#endif

            return null;
        }

        private object CreateSentisWorker(object model)
        {
            try
            {
                if (_workerFactoryType != null && _workerFactoryType.FullName == "Unity.Sentis.Worker")
                {
                    object backend = TryParseBackend("GPUCompute") ?? TryParseBackend("GPUCommandBuffer") ?? TryParseBackend("GPUPixel") ?? TryParseBackend("CPU");
                    foreach (var ctor in _workerFactoryType.GetConstructors(BindingFlags.Public | BindingFlags.Instance))
                    {
                        var ps = ctor.GetParameters();
                        LogDiag($"Worker ctor candidate: ({string.Join(", ", ps.Select(p => p.ParameterType.FullName))})");
                        try
                        {
                            if (ps.Length == 2 && ps[0].ParameterType.IsInstanceOfType(model) && ps[1].ParameterType == _backendTypeType && backend != null)
                                return ctor.Invoke(new[] { model, backend });
                            if (ps.Length == 2 && ps[0].ParameterType == _backendTypeType && ps[1].ParameterType.IsInstanceOfType(model) && backend != null)
                                return ctor.Invoke(new[] { backend, model });
                            if (ps.Length == 1 && ps[0].ParameterType.IsInstanceOfType(model))
                                return ctor.Invoke(new[] { model });
                        }
                        catch (Exception e)
                        {
                            LogDiag("Worker ctor invoke failed: " + e.Message);
                        }
                    }
                    LogDiag("No compatible Unity.Sentis.Worker constructor found");
                }

                var method = _workerFactoryType?.GetMethod("CreateWorker", BindingFlags.Public | BindingFlags.Static);
                if (method != null)
                {
                    object backend = TryParseBackend("GPUCompute") ?? TryParseBackend("CPU");
                    return method.Invoke(null, new[] { backend, model });
                }
            }
            catch (Exception e)
            {
                LogDiag("CreateSentisWorker failed: " + e.Message);
            }
            return null;
        }

        private object TryParseBackend(string name)
        {
            try
            {
                return Enum.Parse(_backendTypeType, name);
            }
            catch
            {
                return null;
            }
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

            if (IsModelAvailable && _worker != null && TrySentisPredict(history, out mu, out sigma))
            {
                RuntimeMode = "onnx";
                LastStatusMessage = "ONNX inference OK";
                LastMu = mu;
                LastSigma = sigma;
                return true;
            }

            if (!fallbackToHeuristicWhenUnavailable || history == null || history.Length < 2)
                return false;

            int nHist = history.Length;
            int s = Mathf.Max(0, nHist - 5);
            float[] w = { 0.08f, 0.14f, 0.2f, 0.28f, 0.4f };

            Vector3 p = Vector3.zero;
            float wsSum = 0f;
            for (int i = s; i < nHist; i++)
            {
                int wi = Mathf.Min(i - s, 4);
                Vector3 centerWind = history[i].windSamples != null && history[i].windSamples.Length > 13 ? history[i].windSamples[13] : Vector3.zero;
                p += centerWind * w[wi];
                wsSum += w[wi];
            }
            if (wsSum > 1e-6f) p /= wsSum;

            Vector3 last = history[nHist - 1].windSamples != null && history[nHist - 1].windSamples.Length > 13 ? history[nHist - 1].windSamples[13] : Vector3.zero;
            Vector3 prev = history[nHist - 2].windSamples != null && history[nHist - 2].windSamples.Length > 13 ? history[nHist - 2].windSamples[13] : Vector3.zero;
            p += (last - prev) * trendGain;

            float var = 0f;
            for (int i = s; i < nHist; i++)
            {
                Vector3 centerWind = history[i].windSamples != null && history[i].windSamples.Length > 13 ? history[i].windSamples[13] : Vector3.zero;
                Vector3 d = centerWind - p;
                var += d.sqrMagnitude;
            }
            var /= Mathf.Max(1, nHist - s);

            mu = p;
            sigma = Mathf.Sqrt(var + 1e-6f) * sigmaScale;
            LastMu = mu;
            LastSigma = sigma;
            RuntimeMode = IsModelAvailable ? "onnx" : "fallback";
            if (!IsModelAvailable)
                LastStatusMessage = "Heuristic fallback used";
            return true;
        }

        private bool TrySentisPredict(ObservationBuffer.ObservationFrame[] history, out Vector3 mu, out float sigma)
        {
            mu = Vector3.zero;
            sigma = 1f;
            if (history == null || history.Length == 0 || _worker == null || _tensorShapeType == null || _tensorFloatType == null)
            {
                if (verboseInferenceDiagnostics)
                    LogDiag($"TrySentisPredict skipped: history={history?.Length ?? -1}, worker={_worker != null}, tensorShape={_tensorShapeType != null}, tensor={_tensorFloatType != null}");
                return false;
            }

            try
            {
                int k = Mathf.Max(1, modelHistoryLength);
                int n = Mathf.Max(1, modelSpatialSamples);
                int c = 3;
                float[] input = new float[k * n * c];
                FillInput(history, input, k, n);

                object shape = CreateTensorShape(1, k, n, c);
                object tensor = CreateTensor(shape, input);
                if (shape == null || tensor == null)
                {
                    LastStatusMessage = "Sentis tensor creation failed";
                    return false;
                }

                var sw = Stopwatch.StartNew();
                bool scheduled = InvokeWorkerSchedule(tensor);
                if (!scheduled)
                {
                    DisposeIfNeeded(tensor);
                    LastStatusMessage = "Sentis worker schedule failed";
                    if (verboseInferenceDiagnostics) LogDiag(LastStatusMessage);
                    return false;
                }

                object muTensor = PeekOutput(muOutputName);
                object lvTensor = PeekOutput(logVarOutputName);
                float[] muArr = TensorToArray(muTensor, 3);
                float[] lvArr = TensorToArray(lvTensor, 3);
                sw.Stop();
                LastInferenceMs = (float)sw.Elapsed.TotalMilliseconds;

                DisposeIfNeeded(tensor);

                if (muArr == null || muArr.Length < 3)
                {
                    LastStatusMessage = "Sentis mu output read failed";
                    if (verboseInferenceDiagnostics) LogDiag(LastStatusMessage);
                    return false;
                }

                if (useNormalization)
                {
                    mu = new Vector3(
                        muArr[0] * _windStd.x + _windMean.x,
                        muArr[1] * _windStd.y + _windMean.y,
                        muArr[2] * _windStd.z + _windMean.z);
                }
                else
                {
                    mu = new Vector3(muArr[0], muArr[1], muArr[2]);
                }

                if (lvArr != null && lvArr.Length >= 3)
                {
                    float sv = Mathf.Sqrt(Mathf.Exp((lvArr[0] + lvArr[1] + lvArr[2]) / 3f));
                    sigma = Mathf.Max(1e-4f, sv * _windStd.magnitude / Mathf.Sqrt(3f));
                }
                else
                {
                    sigma = 1f;
                }

                return true;
            }
            catch (Exception e)
            {
                LastStatusMessage = "Sentis inference failed: " + e.Message;
                LogDiag(LastStatusMessage);
                return false;
            }
        }

        private void FillInput(ObservationBuffer.ObservationFrame[] history, float[] input, int k, int n)
        {
            int startHist = Mathf.Max(0, history.Length - k);
            int padCount = k - (history.Length - startHist);
            int idx = 0;
            Vector3[] firstSamples = history[startHist].windSamples;

            for (int t = 0; t < k; t++)
            {
                int srcIndex = t < padCount ? startHist : startHist + (t - padCount);
                Vector3[] samples = history[srcIndex].windSamples ?? firstSamples;
                for (int i = 0; i < n; i++)
                {
                    Vector3 v = samples != null && i < samples.Length ? samples[i] : Vector3.zero;
                    if (useNormalization)
                    {
                        input[idx++] = (v.x - _windMean.x) / _windStd.x;
                        input[idx++] = (v.y - _windMean.y) / _windStd.y;
                        input[idx++] = (v.z - _windMean.z) / _windStd.z;
                    }
                    else
                    {
                        input[idx++] = v.x;
                        input[idx++] = v.y;
                        input[idx++] = v.z;
                    }
                }
            }
        }

        private object CreateTensorShape(params int[] dims)
        {
            foreach (var ctor in _tensorShapeType.GetConstructors(BindingFlags.Public | BindingFlags.Instance))
            {
                var ps = ctor.GetParameters();
                try
                {
                    if (ps.Length == dims.Length && ps.All(p => p.ParameterType == typeof(int)))
                        return ctor.Invoke(dims.Cast<object>().ToArray());
                    if (ps.Length == 1 && ps[0].ParameterType == typeof(int[]))
                        return ctor.Invoke(new object[] { dims });
                }
                catch { }
            }
            if (verboseInferenceDiagnostics) LogDiag("No compatible TensorShape constructor found");
            return null;
        }

        private object CreateTensor(object shape, float[] data)
        {
            var candidates = new[]
            {
                ResolveType("Unity.Sentis.Tensor`1")?.MakeGenericType(typeof(float)),
                ResolveType("Unity.Sentis.TensorFloat"),
                _tensorFloatType,
            }.Where(t => t != null).Distinct().ToArray();

            foreach (var tensorType in candidates)
            {
                foreach (var ctor in tensorType.GetConstructors(BindingFlags.Public | BindingFlags.Instance))
                {
                    var ps = ctor.GetParameters();
                    if (verboseInferenceDiagnostics)
                        LogDiag($"Tensor ctor candidate {tensorType.FullName}: ({string.Join(", ", ps.Select(p => p.ParameterType.FullName))})");
                    try
                    {
                        if (ps.Length >= 2 && ps[0].ParameterType.IsInstanceOfType(shape) && ps[1].ParameterType == typeof(float[]))
                        {
                            object[] args = ps.Select(p => p.HasDefaultValue ? p.DefaultValue : null).ToArray();
                            args[0] = shape;
                            args[1] = data;
                            return ctor.Invoke(args);
                        }
                        if (ps.Length >= 2 && ps[0].ParameterType == typeof(float[]) && ps[1].ParameterType.IsInstanceOfType(shape))
                        {
                            object[] args = ps.Select(p => p.HasDefaultValue ? p.DefaultValue : null).ToArray();
                            args[0] = data;
                            args[1] = shape;
                            return ctor.Invoke(args);
                        }
                    }
                    catch (Exception e)
                    {
                        if (verboseInferenceDiagnostics) LogDiag("Tensor ctor invoke failed: " + e.Message);
                    }
                }
            }

            var dataType = ResolveType("Unity.Sentis.DataType");
            object floatType = null;
            if (dataType != null)
            {
                foreach (var name in new[] { "Float", "Float32" })
                {
                    try { floatType = Enum.Parse(dataType, name); break; } catch { }
                }
            }

            foreach (var method in _tensorFloatType.GetMethods(BindingFlags.Public | BindingFlags.Static))
            {
                if (!method.Name.Contains("From") && !method.Name.Contains("Create")) continue;
                var ps = method.GetParameters();
                if (verboseInferenceDiagnostics)
                    LogDiag($"Tensor static candidate {method.Name}: ({string.Join(", ", ps.Select(p => p.ParameterType.FullName))})");
                try
                {
                    if (ps.Length == 2 && ps[0].ParameterType.IsInstanceOfType(shape) && ps[1].ParameterType == typeof(float[]))
                        return method.Invoke(null, new object[] { shape, data });
                    if (ps.Length == 2 && ps[0].ParameterType == typeof(float[]) && ps[1].ParameterType.IsInstanceOfType(shape))
                        return method.Invoke(null, new object[] { data, shape });
                    if (ps.Length == 3 && ps[0].ParameterType.IsInstanceOfType(shape) && ps[1].ParameterType == dataType && ps[2].ParameterType == typeof(float[]) && floatType != null)
                        return method.Invoke(null, new object[] { shape, floatType, data });
                }
                catch (Exception e)
                {
                    if (verboseInferenceDiagnostics) LogDiag($"Tensor static {method.Name} failed: {e.Message}");
                }
            }

            if (verboseInferenceDiagnostics) LogDiag("No compatible Tensor constructor found");
            return null;
        }

        private bool InvokeWorkerSchedule(object tensor)
        {
            foreach (string methodName in new[] { "Schedule", "Execute" })
            {
                foreach (var method in _worker.GetType().GetMethods(BindingFlags.Public | BindingFlags.Instance).Where(m => m.Name == methodName))
                {
                    var ps = method.GetParameters();
                    try
                    {
                        if (ps.Length == 1 && ps[0].ParameterType.IsInstanceOfType(tensor))
                        {
                            method.Invoke(_worker, new[] { tensor });
                            return true;
                        }
                        if (ps.Length == 2 && ps[0].ParameterType == typeof(string) && ps[1].ParameterType.IsInstanceOfType(tensor))
                        {
                            method.Invoke(_worker, new object[] { "wind_history", tensor });
                            return true;
                        }
                    }
                    catch (Exception e)
                    {
                        if (verboseInferenceDiagnostics) LogDiag($"Worker {methodName} invoke failed: {e.Message}");
                    }
                }
            }
            return false;
        }

        private object PeekOutput(string outputName)
        {
            foreach (string methodName in new[] { "PeekOutput", "CopyOutput" })
            {
                var method = _worker.GetType().GetMethods(BindingFlags.Public | BindingFlags.Instance)
                    .FirstOrDefault(m => m.Name == methodName && m.GetParameters().Length == 1 && m.GetParameters()[0].ParameterType == typeof(string));
                if (method == null) continue;
                try { return method.Invoke(_worker, new object[] { outputName }); }
                catch (Exception e) { LogDiag($"{methodName}('{outputName}') failed: {e.Message}"); }
            }
            return null;
        }

        private float[] TensorToArray(object tensor, int minLen)
        {
            if (tensor == null) return null;
            foreach (string methodName in new[] { "DownloadToArray", "ToReadOnlyArray", "ReadbackAndClone" })
            {
                var method = tensor.GetType().GetMethod(methodName, BindingFlags.Public | BindingFlags.Instance, null, Type.EmptyTypes, null);
                if (method == null) continue;
                try
                {
                    object result = method.Invoke(tensor, null);
                    if (result is float[] arr) return arr;
                    if (result != null && result.GetType().IsArray && result.GetType().GetElementType() == typeof(float))
                        return ((Array)result).Cast<float>().ToArray();
                }
                catch (Exception e)
                {
                    LogDiag($"Tensor {methodName} failed: {e.Message}");
                }
            }

            var indexer = tensor.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .FirstOrDefault(p => p.GetIndexParameters().Length == 1 && p.GetIndexParameters()[0].ParameterType == typeof(int));
            if (indexer != null)
            {
                try
                {
                    float[] arr = new float[minLen];
                    for (int i = 0; i < minLen; i++) arr[i] = Convert.ToSingle(indexer.GetValue(tensor, new object[] { i }));
                    return arr;
                }
                catch { }
            }
            return null;
        }

        private void DisposeIfNeeded(object obj)
        {
            try { obj?.GetType().GetMethod("Dispose", BindingFlags.Public | BindingFlags.Instance)?.Invoke(obj, null); }
            catch { }
        }

        private void DisposeWorker()
        {
            if (_worker == null) return;
            try
            {
                var m = _worker.GetType().GetMethod("Dispose", BindingFlags.Public | BindingFlags.Instance);
                m?.Invoke(_worker, null);
            }
            catch { }
            _worker = null;
        }

        private void LogDiag(string msg)
        {
            if (verboseDiagnostics)
                UnityEngine.Debug.Log("[ONNXPredictor] " + msg);
        }
    }
}
