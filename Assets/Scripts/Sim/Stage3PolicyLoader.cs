using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEngine;

namespace BalloonSim.Sim
{
    /// <summary>
    /// Stage3 policy runtime loader/inference bridge.
    /// Requires these files in StreamingAssets/Stage3:
    /// - policy_meta.json
    /// - policy_norm.json
    /// and a Sentis-imported ONNX model in Resources (default: policy_net)
    /// </summary>
    public class Stage3PolicyLoader : MonoBehaviour
    {
        [Header("Runtime")]
        public bool enablePolicy = true;
        public bool verboseDiagnostics = true;

        [Header("Paths")]
        public string streamingAssetsStage3Folder = "Stage3";
        public string policyMetaFile = "policy_meta.json";
        public string policyNormFile = "policy_norm.json";
        public string modelResourcePath = "Stage3/policy_net";

        [Header("I/O")]
        public string inputName = "state_vector";
        public string actionOutputName = "action";

        public bool IsReady { get; private set; }
        public string LastStatus { get; private set; } = "init";
        public string LastModelSource { get; private set; } = "n/a";
        public string LastNormSource { get; private set; } = "n/a";

        private float[] _stateMean;
        private float[] _stateStd;
        private int _stateDim;
        private object _worker;
        private object _model;

        private Type _modelAssetType;
        private Type _modelLoaderType;
        private Type _workerType;
        private Type _backendType;
        private Type _tensorShapeType;
        private Type _tensorFloatType;

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

        private void Awake()
        {
            Reinitialize();
        }

        public void Reinitialize()
        {
            DisposeWorker();
            IsReady = false;
            LastStatus = "init";

            if (!enablePolicy)
            {
                LastStatus = "disabled";
                return;
            }

            if (!ProbeSentis())
            {
                LastStatus = "Sentis unavailable";
                return;
            }

            if (!LoadNormAndMeta())
                return;

            if (!LoadModelAndWorker())
                return;

            IsReady = true;
            LastStatus = "ready";
        }

        public bool TryInfer(float[] stateVector, out float[] action)
        {
            action = null;
            if (!IsReady || _worker == null || stateVector == null)
                return false;
            if (_stateDim <= 0 || stateVector.Length != _stateDim)
            {
                LastStatus = $"state dim mismatch: got {stateVector?.Length ?? -1}, expected {_stateDim}";
                return false;
            }

            try
            {
                var norm = new float[_stateDim];
                for (int i = 0; i < _stateDim; i++)
                {
                    float std = Mathf.Max(1e-6f, _stateStd[i]);
                    norm[i] = (stateVector[i] - _stateMean[i]) / std;
                }

                object shape = _tensorShapeType.GetConstructor(new[] { typeof(int), typeof(int) })?.Invoke(new object[] { 1, _stateDim });
                if (shape == null)
                {
                    LastStatus = "TensorShape(1,state_dim) ctor not found";
                    return false;
                }

                object tensor = _tensorFloatType.GetConstructor(new[] { _tensorShapeType, typeof(float[]) })?.Invoke(new object[] { shape, norm });
                if (tensor == null)
                {
                    LastStatus = "Tensor<float>(shape,data) ctor not found";
                    return false;
                }

                var exec = _worker.GetType().GetMethod("Schedule", BindingFlags.Public | BindingFlags.Instance)
                           ?? _worker.GetType().GetMethod("Execute", BindingFlags.Public | BindingFlags.Instance);
                if (exec == null)
                {
                    LastStatus = "Worker Schedule/Execute not found";
                    return false;
                }

                var ps = exec.GetParameters();
                if (ps.Length == 1)
                    exec.Invoke(_worker, new[] { tensor });
                else if (ps.Length == 2 && ps[0].ParameterType == typeof(string))
                    exec.Invoke(_worker, new object[] { inputName, tensor });
                else
                {
                    LastStatus = "Worker execute signature unsupported";
                    return false;
                }

                var peek = _worker.GetType().GetMethod("PeekOutput", BindingFlags.Public | BindingFlags.Instance)
                           ?? _worker.GetType().GetMethod("CopyOutput", BindingFlags.Public | BindingFlags.Instance);
                if (peek == null)
                {
                    LastStatus = "PeekOutput/CopyOutput not found";
                    return false;
                }

                var outTensor = peek.Invoke(_worker, new object[] { actionOutputName });
                if (outTensor == null)
                {
                    LastStatus = $"output not found: {actionOutputName}";
                    return false;
                }

                var arr = outTensor.GetType().GetMethod("DownloadToArray", Type.EmptyTypes)?.Invoke(outTensor, null) as float[];
                if (arr == null)
                {
                    var rb = outTensor.GetType().GetMethod("ReadbackAndClone", BindingFlags.Public | BindingFlags.Instance)?.Invoke(outTensor, null);
                    arr = rb?.GetType().GetMethod("DownloadToArray", Type.EmptyTypes)?.Invoke(rb, null) as float[];
                }

                if (arr == null || arr.Length < 3)
                {
                    LastStatus = "action output read failed";
                    return false;
                }

                action = arr.Take(3).ToArray();
                LastStatus = "inference ok";
                return true;
            }
            catch (Exception e)
            {
                LastStatus = "inference failed: " + e.Message;
                Log(LastStatus);
                return false;
            }
        }

        private bool ProbeSentis()
        {
            _modelAssetType = ResolveType("Unity.Sentis.ModelAsset");
            _modelLoaderType = ResolveType("Unity.Sentis.ModelLoader");
            _workerType = ResolveType("Unity.Sentis.Worker");
            _backendType = ResolveType("Unity.Sentis.BackendType");
            _tensorShapeType = ResolveType("Unity.Sentis.TensorShape");
            _tensorFloatType = ResolveType("Unity.Sentis.Tensor`1")?.MakeGenericType(typeof(float));
            return _modelAssetType != null && _modelLoaderType != null && _workerType != null && _backendType != null && _tensorShapeType != null && _tensorFloatType != null;
        }

        private bool LoadNormAndMeta()
        {
            string root = Path.Combine(Application.streamingAssetsPath, string.IsNullOrWhiteSpace(streamingAssetsStage3Folder) ? "Stage3" : streamingAssetsStage3Folder);
            string metaPath = Path.Combine(root, policyMetaFile);
            string normPath = Path.Combine(root, policyNormFile);

            if (!File.Exists(metaPath))
            {
                LastStatus = "missing meta: " + metaPath;
                return false;
            }
            if (!File.Exists(normPath))
            {
                LastStatus = "missing norm: " + normPath;
                return false;
            }

            var meta = JsonUtility.FromJson<PolicyMeta>(File.ReadAllText(metaPath));
            var norm = JsonUtility.FromJson<PolicyNorm>(File.ReadAllText(normPath));
            if (norm?.state_mean == null || norm.state_std == null)
            {
                LastStatus = "invalid policy_norm schema";
                return false;
            }

            _stateMean = norm.state_mean;
            _stateStd = norm.state_std;
            _stateDim = norm.state_dim > 0 ? norm.state_dim : norm.state_mean.Length;
            LastNormSource = normPath;

            if (!string.IsNullOrWhiteSpace(meta?.onnx_file))
                modelResourcePath = "Stage3/" + Path.GetFileNameWithoutExtension(meta.onnx_file);

            return true;
        }

        private bool LoadModelAndWorker()
        {
            var obj = Resources.Load(modelResourcePath);
            if (obj == null)
            {
                LastStatus = "missing Resources model: " + modelResourcePath;
                return false;
            }
            if (!_modelAssetType.IsInstanceOfType(obj))
            {
                LastStatus = "Resources object is not Sentis ModelAsset: " + obj.GetType().FullName;
                return false;
            }

            var load = _modelLoaderType.GetMethod("Load", BindingFlags.Public | BindingFlags.Static, null, new[] { _modelAssetType }, null);
            _model = load?.Invoke(null, new[] { obj });
            if (_model == null)
            {
                LastStatus = "ModelLoader.Load failed";
                return false;
            }

            object backend = null;
            foreach (var n in new[] { "GPUCompute", "CPU" })
            {
                try { backend = Enum.Parse(_backendType, n); break; } catch { }
            }

            var ctor = _workerType.GetConstructor(new[] { _model.GetType(), _backendType });
            if (ctor == null)
            {
                LastStatus = "Worker(model, backend) ctor not found";
                return false;
            }

            _worker = ctor.Invoke(new[] { _model, backend });
            LastModelSource = "Resources/" + modelResourcePath;
            return _worker != null;
        }

        private static Type ResolveType(string fullName)
        {
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                var t = asm.GetType(fullName, false);
                if (t != null) return t;
            }
            return null;
        }

        private void DisposeWorker()
        {
            try { _worker?.GetType().GetMethod("Dispose", BindingFlags.Public | BindingFlags.Instance)?.Invoke(_worker, null); } catch { }
            _worker = null;
        }

        private void Log(string msg)
        {
            if (verboseDiagnostics) Debug.Log("[Stage3PolicyLoader] " + msg);
        }
    }
}
