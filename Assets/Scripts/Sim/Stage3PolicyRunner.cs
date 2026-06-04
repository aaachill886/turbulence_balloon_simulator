using System;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using UnityEngine;

namespace BalloonSim.Sim
{
    public class Stage3PolicyRunner : MonoBehaviour
    {
        [Header("Runtime")]
        public bool enablePolicy = true;
        public float maxPolicySpeed = 2.0f;
        public bool verboseDiagnostics = true;
        public bool verboseInferenceDiagnostics = false;

        [Header("Model")]
        public UnityEngine.Object policyModelAsset;
        public string modelResourcePath = "Stage3/policy_net";
        public string streamingAssetsFolder = "Stage3";
        public string normFileName = "policy_norm.json";
        public string metaFileName = "policy_meta.json";
        public string stateInputName = "state_vector";
        public string actionOutputName = "action_logits";

        [Header("Refs")]
        public SimulationConfig config;
        public BalloonState balloon;
        public TurbulenceField field;
        public AutopilotController autopilot;
        public GameController game;
        public Transform waypoint;

        public bool IsPolicyAvailable { get; private set; }
        public string RuntimeMode { get; private set; } = "off";
        public string LastStatusMessage { get; private set; } = "init";
        public string LastModelSource { get; private set; } = "n/a";
        public string LastNormSource { get; private set; } = "n/a";
        public string LastMetaSource { get; private set; } = "n/a";
        public float LastInferenceMs { get; private set; }
        public Vector3 LastAction { get; private set; }

        private float[] _stateMean = Array.Empty<float>();
        private float[] _stateStd = Array.Empty<float>();
        private int _stateDim = 34;
        private int _actionDim = 6;

        private bool _sentisAvailable;
        private Type _modelAssetType;
        private Type _modelLoaderType;
        private Type _workerType;
        private Type _backendTypeType;
        private Type _tensorType;
        private Type _tensorShapeType;
        private object _model;
        private object _worker;

        private void Awake()
        {
            Reinitialize();
        }

        private void OnDestroy()
        {
            DisposeWorker();
        }

        public void Reinitialize()
        {
            DisposeWorker();
            LoadNormAndMeta();
            ProbeSentis();
            TryInitModel();
        }

        public bool TryPredictAction(out Vector3 action)
        {
            action = Vector3.zero;
            LastInferenceMs = 0f;

            if (!ShouldRunPolicy())
            {
                RuntimeMode = "off";
                LastStatusMessage = config != null && !config.aiEnabled ? "Stage3 policy waiting for AI mode" : "Stage3 policy disabled";
                return false;
            }

            if (!IsPolicyAvailable || _worker == null)
                return false;

            if (!BuildStateVector(out var state))
            {
                LastStatusMessage = "Stage3 state build failed";
                return false;
            }

            NormalizeStateInPlace(state);

            try
            {
                object shape = CreateTensorShape(1, _stateDim);
                object tensor = CreateTensor(shape, state);
                if (shape == null || tensor == null)
                {
                    LastStatusMessage = "Stage3 tensor creation failed";
                    return false;
                }

                var sw = Stopwatch.StartNew();
                bool scheduled = InvokeWorkerSchedule(tensor, stateInputName);
                if (!scheduled)
                {
                    DisposeIfNeeded(tensor);
                    LastStatusMessage = "Stage3 worker schedule failed";
                    return false;
                }

                object outTensor = PeekOutput(actionOutputName);
                float[] arr = TensorToArray(outTensor, _actionDim);
                sw.Stop();
                LastInferenceMs = (float)sw.Elapsed.TotalMilliseconds;
                DisposeIfNeeded(tensor);

                if (arr == null || arr.Length < 6)
                {
                    LastStatusMessage = "Stage3 action output read failed";
                    return false;
                }

                action = SixAxisToVector(arr);
                action = Vector3.ClampMagnitude(action, Mathf.Max(0.1f, maxPolicySpeed));
                LastAction = action;
                RuntimeMode = "onnx";
                LastStatusMessage = "Stage3 policy inference OK";
                return true;
            }
            catch (Exception e)
            {
                RuntimeMode = "fallback";
                LastStatusMessage = "Stage3 inference failed: " + e.Message;
                LogDiag(LastStatusMessage);
                return false;
            }
        }

        private bool ShouldRunPolicy()
        {
            return enablePolicy && config != null && config.aiEnabled;
        }

        private bool BuildStateVector(out float[] state)
        {
            state = new float[_stateDim];
            if (balloon == null) return false;

            Vector3 pos = balloon.transform.position;
            Vector3 vel = balloon.velocity;
            Vector3 pred = autopilot != null ? autopilot.Predicted : Vector3.zero;
            float sigma = autopilot != null ? autopilot.PredSigma : 1f;
            Vector3 waypointPos = waypoint != null ? waypoint.position : pos;
            Vector3 wpDelta = waypointPos - pos;
            Vector3 wpDir = wpDelta.sqrMagnitude > 1e-8f ? wpDelta.normalized : Vector3.zero;
            float[,] g = field != null ? field.SampleGradient(pos) : new float[3, 3];
            Vector3 prev = LastAction;

            int i = 0;
            state[i++] = pos.x; state[i++] = pos.y; state[i++] = pos.z;
            state[i++] = vel.x; state[i++] = vel.y; state[i++] = vel.z;
            state[i++] = pred.x; state[i++] = pred.y; state[i++] = pred.z;
            state[i++] = sigma;
            state[i++] = waypointPos.y - pos.y;
            state[i++] = wpDir.x; state[i++] = wpDir.y; state[i++] = wpDir.z;
            state[i++] = wpDelta.magnitude;
            state[i++] = g[0, 0]; state[i++] = g[0, 1]; state[i++] = g[0, 2];
            state[i++] = g[1, 0]; state[i++] = g[1, 1]; state[i++] = g[1, 2];
            state[i++] = g[2, 0]; state[i++] = g[2, 1]; state[i++] = g[2, 2];
            state[i++] = config != null ? config.beaufort : 0f;
            state[i++] = config != null ? config.viscosity : 0f;
            state[i++] = config != null ? config.gustStrength : 0f;
            state[i++] = config != null ? config.densityRatio : 1f;
            state[i++] = Mathf.Max(0f, prev.x); state[i++] = Mathf.Max(0f, -prev.x);
            state[i++] = Mathf.Max(0f, prev.y); state[i++] = Mathf.Max(0f, -prev.y);
            state[i++] = Mathf.Max(0f, prev.z); state[i++] = Mathf.Max(0f, -prev.z);
            return i == _stateDim;
        }

        private static Vector3 SixAxisToVector(float[] arr)
        {
            return new Vector3(
                arr[0] - arr[1],
                arr[2] - arr[3],
                arr[4] - arr[5]
            );
        }

        private void NormalizeStateInPlace(float[] state)
        {
            if (_stateMean.Length != state.Length || _stateStd.Length != state.Length) return;
            for (int i = 0; i < state.Length; i++)
                state[i] = (state[i] - _stateMean[i]) / Mathf.Max(1e-6f, _stateStd[i]);
        }

        private void LoadNormAndMeta()
        {
            string root = System.IO.Path.Combine(Application.streamingAssetsPath, string.IsNullOrWhiteSpace(streamingAssetsFolder) ? "Stage3" : streamingAssetsFolder);
            string normPath = System.IO.Path.Combine(root, normFileName);
            string metaPath = System.IO.Path.Combine(root, metaFileName);

            if (System.IO.File.Exists(normPath))
            {
                try
                {
                    var norm = JsonUtility.FromJson<PolicyNorm>(System.IO.File.ReadAllText(normPath));
                    _stateMean = norm.state_mean ?? Array.Empty<float>();
                    _stateStd = norm.state_std ?? Array.Empty<float>();
                    _stateDim = norm.state_dim > 0 ? norm.state_dim : _stateMean.Length;
                    _actionDim = norm.action_dim > 0 ? norm.action_dim : 6;
                    LastNormSource = normPath;
                }
                catch (Exception e)
                {
                    LastNormSource = "invalid: " + e.Message;
                }
            }
            else
            {
                LastNormSource = "missing: " + normPath;
            }

            if (System.IO.File.Exists(metaPath))
            {
                LastMetaSource = metaPath;
                try
                {
                    var meta = JsonUtility.FromJson<PolicyMeta>(System.IO.File.ReadAllText(metaPath));
                    if (!string.IsNullOrWhiteSpace(meta.onnx_file))
                        LogDiag($"Stage3 meta loaded: onnx={meta.onnx_file}, state_dim={meta.state_dim}, action_dim={meta.action_dim}");
                }
                catch (Exception e)
                {
                    LastMetaSource = "invalid: " + e.Message;
                }
            }
            else
            {
                LastMetaSource = "missing: " + metaPath;
            }
        }

        [Serializable]
        private class PolicyNorm
        {
            public float[] state_mean;
            public float[] state_std;
            public int state_dim;
            public int action_dim;
        }

        [Serializable]
        private class PolicyMeta
        {
            public int schema_version;
            public string model_file;
            public string onnx_file;
            public string norm_file;
            public int state_dim;
            public int action_dim;
        }

        private void ProbeSentis()
        {
            _modelAssetType = ResolveType("Unity.Sentis.ModelAsset");
            _modelLoaderType = ResolveType("Unity.Sentis.ModelLoader");
            _workerType = ResolveType("Unity.Sentis.Worker") ?? ResolveType("Unity.Sentis.WorkerFactory");
            _backendTypeType = ResolveType("Unity.Sentis.BackendType");
            _tensorType = ResolveType("Unity.Sentis.Tensor`1")?.MakeGenericType(typeof(float)) ?? ResolveType("Unity.Sentis.Tensor") ?? ResolveType("Unity.Sentis.TensorFloat");
            _tensorShapeType = ResolveType("Unity.Sentis.TensorShape");
            _sentisAvailable = _modelAssetType != null && _modelLoaderType != null && _workerType != null && _backendTypeType != null && _tensorType != null && _tensorShapeType != null;
            if (!_sentisAvailable) LogDiag("Stage3 Sentis unavailable; policy fallback only");
        }

        private void TryInitModel()
        {
            IsPolicyAvailable = false;
            RuntimeMode = enablePolicy ? "fallback" : "off";
            if (!enablePolicy)
            {
                LastStatusMessage = "Stage3 policy disabled";
                return;
            }
            if (!_sentisAvailable)
            {
                LastStatusMessage = "Stage3 Sentis unavailable";
                return;
            }

            if (policyModelAsset == null && !string.IsNullOrWhiteSpace(modelResourcePath))
                policyModelAsset = LoadSentisModelAsset(modelResourcePath);

            if (policyModelAsset == null)
            {
                LastStatusMessage = "Stage3 policy ModelAsset not assigned";
                return;
            }

            if (!_modelAssetType.IsInstanceOfType(policyModelAsset))
            {
                var resolved = LoadSentisModelAsset(modelResourcePath);
                if (resolved != null && _modelAssetType.IsInstanceOfType(resolved))
                    policyModelAsset = resolved;
                else
                {
                    LastStatusMessage = $"Stage3 assigned object is not Sentis ModelAsset: {policyModelAsset.GetType().FullName}";
                    return;
                }
            }

            try
            {
                var load = _modelLoaderType.GetMethod("Load", BindingFlags.Public | BindingFlags.Static, null, new[] { _modelAssetType }, null);
                _model = load?.Invoke(null, new[] { policyModelAsset });
                _worker = CreateWorker(_model);
                IsPolicyAvailable = _worker != null;
                RuntimeMode = IsPolicyAvailable ? "onnx" : "fallback";
                LastStatusMessage = IsPolicyAvailable ? "Stage3 policy loaded" : "Stage3 worker create failed";
                if (string.IsNullOrWhiteSpace(LastModelSource) || LastModelSource == "n/a")
                    LastModelSource = policyModelAsset.name;
                LogDiag($"Stage3 policy init: available={IsPolicyAvailable}, model={LastModelSource}, worker={_worker?.GetType().FullName ?? "null"}");
            }
            catch (Exception e)
            {
                LastStatusMessage = "Stage3 policy init failed: " + e.Message;
                RuntimeMode = "fallback";
                LogDiag(LastStatusMessage);
            }
        }

        private UnityEngine.Object LoadSentisModelAsset(string resourcePath)
        {
            UnityEngine.Object loaded = Resources.Load(resourcePath);
            if (loaded != null && _modelAssetType.IsInstanceOfType(loaded))
            {
                LastModelSource = $"Resources/{resourcePath}";
                return loaded;
            }
            foreach (var asset in Resources.LoadAll(resourcePath))
            {
                if (asset != null && _modelAssetType.IsInstanceOfType(asset))
                {
                    LastModelSource = $"Resources/{resourcePath}";
                    return asset;
                }
            }

#if UNITY_EDITOR
            string resourceAssetPath = $"Assets/Resources/{resourcePath}.onnx";
            foreach (var asset in UnityEditor.AssetDatabase.LoadAllAssetsAtPath(resourceAssetPath))
            {
                if (asset != null && _modelAssetType.IsInstanceOfType(asset))
                {
                    LastModelSource = resourceAssetPath;
                    return asset;
                }
            }

#endif
            return null;
        }

        private object CreateWorker(object model)
        {
            if (model == null) return null;
            try
            {
                if (_workerType.FullName == "Unity.Sentis.Worker")
                {
                    object backend = TryParseBackend("GPUCompute") ?? TryParseBackend("GPUCommandBuffer") ?? TryParseBackend("GPUPixel") ?? TryParseBackend("CPU");
                    foreach (var ctor in _workerType.GetConstructors(BindingFlags.Public | BindingFlags.Instance))
                    {
                        var ps = ctor.GetParameters();
                        if (ps.Length == 2 && ps[0].ParameterType.IsInstanceOfType(model) && ps[1].ParameterType == _backendTypeType && backend != null)
                            return ctor.Invoke(new[] { model, backend });
                        if (ps.Length == 2 && ps[0].ParameterType == _backendTypeType && ps[1].ParameterType.IsInstanceOfType(model) && backend != null)
                            return ctor.Invoke(new[] { backend, model });
                        if (ps.Length == 1 && ps[0].ParameterType.IsInstanceOfType(model))
                            return ctor.Invoke(new[] { model });
                    }
                }
                var method = _workerType.GetMethod("CreateWorker", BindingFlags.Public | BindingFlags.Static);
                if (method != null)
                    return method.Invoke(null, new[] { TryParseBackend("GPUCompute") ?? TryParseBackend("CPU"), model });
            }
            catch (Exception e)
            {
                LogDiag("Stage3 worker create failed: " + e.Message);
            }
            return null;
        }

        private object TryParseBackend(string name)
        {
            try { return Enum.Parse(_backendTypeType, name); }
            catch { return null; }
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
                }
                catch { }
            }
            return null;
        }

        private object CreateTensor(object shape, float[] data)
        {
            foreach (var ctor in _tensorType.GetConstructors(BindingFlags.Public | BindingFlags.Instance))
            {
                var ps = ctor.GetParameters();
                try
                {
                    if (ps.Length >= 2 && ps[0].ParameterType.IsInstanceOfType(shape) && ps[1].ParameterType == typeof(float[]))
                    {
                        object[] args = ps.Length == 2 ? new object[] { shape, data } : new object[] { shape, data, 0 };
                        return ctor.Invoke(args);
                    }
                }
                catch (Exception e)
                {
                    if (verboseInferenceDiagnostics) LogDiag("Stage3 tensor ctor failed: " + e.Message);
                }
            }
            return null;
        }

        private bool InvokeWorkerSchedule(object tensor, string inputName)
        {
            foreach (string methodName in new[] { "Schedule", "Execute" })
            {
                foreach (var method in _worker.GetType().GetMethods(BindingFlags.Public | BindingFlags.Instance).Where(m => m.Name == methodName))
                {
                    var ps = method.GetParameters();
                    try
                    {
                        if (ps.Length == 1 && ps[0].ParameterType.IsInstanceOfType(tensor)) { method.Invoke(_worker, new[] { tensor }); return true; }
                        if (ps.Length == 2 && ps[0].ParameterType == typeof(string) && ps[1].ParameterType.IsInstanceOfType(tensor)) { method.Invoke(_worker, new object[] { inputName, tensor }); return true; }
                    }
                    catch (Exception e)
                    {
                        if (verboseInferenceDiagnostics) LogDiag($"Stage3 worker {methodName} failed: {e.Message}");
                    }
                }
            }
            return false;
        }

        private object PeekOutput(string outputName)
        {
            foreach (string methodName in new[] { "PeekOutput", "CopyOutput" })
            {
                var method = _worker.GetType().GetMethods(BindingFlags.Public | BindingFlags.Instance).FirstOrDefault(m => m.Name == methodName && m.GetParameters().Length == 1 && m.GetParameters()[0].ParameterType == typeof(string));
                if (method == null) continue;
                try { return method.Invoke(_worker, new object[] { outputName }); }
                catch (Exception e) { if (verboseInferenceDiagnostics) LogDiag($"Stage3 {methodName} failed: {e.Message}"); }
            }
            return null;
        }

        private float[] TensorToArray(object tensor, int minLen)
        {
            if (tensor == null) return null;
            foreach (string methodName in new[] { "DownloadToArray", "ToReadOnlyArray" })
            {
                var method = tensor.GetType().GetMethod(methodName, BindingFlags.Public | BindingFlags.Instance, null, Type.EmptyTypes, null);
                if (method == null) continue;
                try
                {
                    object result = method.Invoke(tensor, null);
                    if (result is float[] arr) return arr;
                }
                catch { }
            }
            var readback = tensor.GetType().GetMethod("ReadbackAndClone", BindingFlags.Public | BindingFlags.Instance, null, Type.EmptyTypes, null);
            if (readback != null)
            {
                try
                {
                    object clone = readback.Invoke(tensor, null);
                    if (clone != null && !ReferenceEquals(clone, tensor))
                    {
                        var arr = TensorToArray(clone, minLen);
                        DisposeIfNeeded(clone);
                        if (arr != null) return arr;
                    }
                }
                catch { }
            }
            var indexer = tensor.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance).FirstOrDefault(p => p.GetIndexParameters().Length == 1 && p.GetIndexParameters()[0].ParameterType == typeof(int));
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

        private static Type ResolveType(string fullName)
        {
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                var type = asm.GetType(fullName, false);
                if (type != null) return type;
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
            DisposeIfNeeded(_worker);
            _worker = null;
        }

        private void LogDiag(string msg)
        {
            if (verboseDiagnostics)
                UnityEngine.Debug.Log("[Stage3PolicyRunner] " + msg);
        }
    }
}
