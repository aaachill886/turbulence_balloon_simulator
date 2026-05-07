# Unity 气球湍流模拟器 - 物理重构计划

## 执行摘要

当前 Unity 模拟器使用 Perlin 噪声生成湍流，**无法产生真实物理特性**。本文档提供完整的重构路径，将其升级为基于 Von Kármán 谱的真实湍流模拟器，支持现实部署。

---

## Phase 1: 核心湍流场重构 (P0 - 3-4天)

### 1.1 创建 `TurbulenceFieldPhysical.cs`

**位置**: `Assets/Scripts/Sim/TurbulenceFieldPhysical.cs`

**核心改动**:
```csharp
// 替换现有的 TurbulenceField.cs 中的 Generate() 方法

// 旧方法 (Perlin噪声):
for (int m = 1; m <= 5; m++) {
    float k = m * 2π / boxSize;
    float a = k^(-0.9) * 0.22 * bf * rb;  // 无物理依据
    vx += a * sin(k*wx + phase);
}

// 新方法 (Von Kármán谱):
1. 生成随机复数场: u_hat[c] = randn + i*randn
2. 应用谱衰减: u_hat *= sqrt(E_k), E_k = k^(-slope), slope ∈ [1.65, 1.90]
3. 各向异性: u_hat[0] *= 1.25, u_hat[1] *= 0.80
4. Helmholtz投影: u_hat -= k(k·u_hat)/|k|²  → 保证无散度
5. IFFT回物理空间: u(x) = Σ_k [Re*cos(k·x) - Im*sin(k·x)]
6. 叠加线性剪切: u[0] += 0.25 * shear_profile
```

**关键参数**:
```csharp
[Header("Physics")]
public float reTau = 1000f;              // 雷诺数 → nu = 1/reTau
public float nuMultiplier = 1.15f;       // Target domain: nu *= 1.15
public float slopeMin = 1.65f;           // Von Kármán谱斜率范围
public float slopeMax = 1.90f;

[Header("Advection-Diffusion")]
public float diffusionDt = 0.02f;        // 粘性衰减时间步
public float advectionCoeffX = 0.10f;    // 非线性对流系数
public float cubicDrag = 0.015f;         // 立方阻力: u -= 0.015*u*|u|
```

### 1.2 添加演化方程 `AdvectDiffuse()`

**物理方程**: u^{t+1} = A(u^t) + ν∇²u^t + f^t

```csharp
private void AdvectDiffuse() {
    // 1. 粘性扩散 (Laplacian有限差分)
    float laplacian = (u[xp] + u[xm] + u[yp] + u[ym] + u[zp] + u[zm] - 6*u[center]) / dx²;
    u_next = u + nu*dt*laplacian;
    
    // 2. 非线性对流: u·∇u
    float grad_x = (u[xp] - u[xm]) * 0.5;
    u_next[0] -= 0.10 * u[0] * grad_x;
    
    // 3. 交叉耦合 (Python中的roll操作)
    u_next[0] += 0.07 * u[1, ix, (iy-1+N)%N, iz];  // roll(u[1], shift=1, axis=1)
    
    // 4. 交叉乘积
    u_next[0] += 0.04 * u[0] * u[1];
    u_next[2] += 0.03 * tanh(u[0] * u[2]);
    
    // 5. 立方阻力
    u_next -= 0.015 * u * |u|;
    
    // 6. Target domain强制项
    u_next[0] += 0.10 * z;  // Wall drive
    u_next[0] += 0.03 * gaussian_shear * u[1](x+1);
}
```

### 1.3 更新 `Bootstrap.cs` 和 `GameController.cs`

```csharp
// Bootstrap.cs
var fieldGo = new GameObject("TurbulenceField");
var field = fieldGo.AddComponent<TurbulenceFieldPhysical>();  // 改这里
field.Initialize();

// GameController.cs - 保持接口兼容
Vector3 u = field.Sample(balloon.transform.position);  // 已有接口，无需改动
```

---

## Phase 2: 飞控系统重构 (P0 - 2-3天)

### 2.1 创建 `ObservationBuffer.cs`

**目的**: 管理 K=8 步历史观测窗口

```csharp
public class ObservationBuffer : MonoBehaviour
{
    public int historyLength = 8;
    public int spatialSamples = 27;  // 3×3×3邻域采样
    
    private Queue<ObservationFrame> history;
    
    [Serializable]
    public struct ObservationFrame
    {
        public float timestamp;
        public Vector3 position;
        public Vector3 velocity;
        public Vector3[] windSamples;     // 27个空间采样点
        public float[,] windGradient;     // 3×3 Jacobian
    }
    
    public void PushObservation(Vector3 pos, Vector3 vel, TurbulenceFieldPhysical field)
    {
        var frame = new ObservationFrame {
            timestamp = Time.time,
            position = pos,
            velocity = vel,
            windSamples = SampleNeighborhood(pos, field),
            windGradient = field.SampleGradient(pos)
        };
        
        history.Enqueue(frame);
        if (history.Count > historyLength) history.Dequeue();
    }
    
    private Vector3[] SampleNeighborhood(Vector3 center, TurbulenceFieldPhysical field)
    {
        Vector3[] samples = new Vector3[27];
        int idx = 0;
        for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dz = -1; dz <= 1; dz++)
        {
            Vector3 offset = new Vector3(dx, dy, dz) * 0.5f;
            samples[idx++] = field.SampleRaw(center + offset);
        }
        return samples;
    }
    
    public ObservationFrame[] GetHistory() => history.ToArray();
}
```

### 2.2 创建 `AdaptiveFlightController.cs`

**核心思想**: 基于预测不确定性的自适应增益

```csharp
public class AdaptiveFlightController : MonoBehaviour
{
    public ObservationBuffer obsBuffer;
    public TurbulenceFieldPhysical field;
    
    // 当前使用Baseline预测器，未来替换为ONNX模型
    private BaselinePredictor predictor;
    
    public Vector3 ComputeControl(Vector3 userInput, Vector3 currentVel, float dt)
    {
        // 1. 获取历史观测
        var history = obsBuffer.GetHistory();
        
        // 2. 预测未来风场 (μ, σ)
        var prediction = predictor.Predict(history);
        Vector3 mu = prediction.mean;
        float sigma = prediction.uncertainty;
        
        // 3. 自适应增益 (不确定性高 → 减小补偿)
        float gain = ComputeAdaptiveGain(sigma);
        
        // 4. 风场补偿
        Vector3 compensation = -mu * gain;
        
        // 5. 阻尼
        Vector3 damping = -currentVel * 0.3f;
        
        // 6. 输出
        return userInput + compensation + damping;
    }
    
    private float ComputeAdaptiveGain(float sigma)
    {
        // σ大 → gain小 (保守策略)
        // σ小 → gain大 (激进补偿)
        float maxGain = 0.8f;
        float minGain = 0.2f;
        float sigmaThreshold = 0.5f;
        
        return Mathf.Lerp(maxGain, minGain, 
            Mathf.Clamp01(sigma / sigmaThreshold));
    }
}

[Serializable]
public struct PredictionResult
{
    public Vector3 mean;          // μ: 预测均值
    public float uncertainty;     // σ: 预测标准差
    public float confidence;      // 1/σ: 置信度
}
```

### 2.3 Baseline预测器 (过渡方案)

```csharp
public class BaselinePredictor
{
    // 保留现有的加权平均+线性外推
    // 但输出 (μ, σ)，其中 σ 基于历史方差估计
    
    public PredictionResult Predict(ObservationFrame[] history)
    {
        if (history.Length < 3) 
            return new PredictionResult { 
                mean = Vector3.zero, 
                uncertainty = 1.0f 
            };
        
        // 加权平均
        float[] weights = { 0.08f, 0.14f, 0.2f, 0.28f, 0.4f };
        Vector3 mu = Vector3.zero;
        float wSum = 0f;
        
        int start = Mathf.Max(0, history.Length - 5);
        for (int i = start; i < history.Length; i++)
        {
            int wi = Mathf.Min(i - start, 4);
            mu += history[i].windSamples[13] * weights[wi];  // 中心点
            wSum += weights[wi];
        }
        mu /= wSum;
        
        // 线性外推
        if (history.Length > 1)
        {
            Vector3 trend = history[history.Length-1].windSamples[13] 
                          - history[history.Length-2].windSamples[13];
            mu += trend * 0.28f;
        }
        
        // 估计不确定性 (历史方差)
        float variance = 0f;
        for (int i = start; i < history.Length; i++)
        {
            Vector3 diff = history[i].windSamples[13] - mu;
            variance += diff.sqrMagnitude;
        }
        variance /= (history.Length - start);
        float sigma = Mathf.Sqrt(variance + 1e-6f);
        
        return new PredictionResult { 
            mean = mu, 
            uncertainty = sigma,
            confidence = 1f / (sigma + 1e-6f)
        };
    }
}
```

---

## Phase 3: 数据采集升级 (P1 - 0.5天)

### 3.1 扩展 `DataLogger.cs`

```csharp
// 新增字段
private void FixedUpdate()
{
    // ... 现有字段 ...
    
    // 新增: K步历史观测
    var history = obsBuffer.GetHistory();
    string historyJson = JsonUtility.ToJson(history);
    
    // 新增: 预测结果
    var pred = flightController.GetLastPrediction();
    
    // 新增: 空间梯度
    var grad = field.SampleGradient(pos);
    
    string line = string.Join(",",
        // ... 现有字段 ...
        pred.mean.x, pred.mean.y, pred.mean.z,
        pred.uncertainty,
        grad[0,0], grad[0,1], grad[0,2],
        grad[1,0], grad[1,1], grad[1,2],
        grad[2,0], grad[2,1], grad[2,2],
        historyJson.Replace(",", ";")  // 避免CSV冲突
    );
}
```

### 3.2 新增 Header

```csharp
private static string Header()
{
    return string.Join(",",
        // ... 现有字段 ...
        "pred_mu_x", "pred_mu_y", "pred_mu_z",
        "pred_sigma",
        "grad_00", "grad_01", "grad_02",
        "grad_10", "grad_11", "grad_12",
        "grad_20", "grad_21", "grad_22",
        "history_json"
    );
}
```

---

## Phase 4: ONNX推理集成 (P1 - 1-2天)

### 4.1 安装 Unity Sentis

```
Window → Package Manager → Add package by name
com.unity.sentis
```

### 4.2 创建 `ONNXPredictor.cs`

```csharp
using Unity.Sentis;

public class ONNXPredictor : MonoBehaviour
{
    public ModelAsset modelAsset;  // 拖入训练好的.onnx文件
    
    private IWorker worker;
    private Model runtimeModel;
    
    void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, runtimeModel);
    }
    
    public PredictionResult Predict(ObservationFrame[] history)
    {
        // 1. 构造输入张量 (B=1, K=8, N=27, C=3)
        TensorFloat input = new TensorFloat(
            new TensorShape(1, 8, 27, 3), 
            FlattenHistory(history)
        );
        
        // 2. 推理
        worker.Execute(input);
        
        // 3. 读取输出 (mu, logvar)
        TensorFloat muTensor = worker.PeekOutput("mu") as TensorFloat;
        TensorFloat logvarTensor = worker.PeekOutput("logvar") as TensorFloat;
        
        Vector3 mu = new Vector3(
            muTensor[0, 0], muTensor[0, 1], muTensor[0, 2]);
        
        float logvar = (logvarTensor[0, 0] + logvarTensor[0, 1] + logvarTensor[0, 2]) / 3f;
        float sigma = Mathf.Sqrt(Mathf.Exp(logvar));
        
        input.Dispose();
        
        return new PredictionResult { 
            mean = mu, 
            uncertainty = sigma,
            confidence = 1f / (sigma + 1e-6f)
        };
    }
    
    private float[] FlattenHistory(ObservationFrame[] history)
    {
        // 转换为 (K, N, C) flat array
        // ...
    }
    
    void OnDestroy()
    {
        worker?.Dispose();
    }
}
```

---

## Phase 5: 验证与对比 (P1 - 1天)

### 5.1 创建对比UI

```csharp
// SimUIOnGUI.cs 新增
if (_showAdv)
{
    GUILayout.Label("=== Predictor Comparison ===");
    GUILayout.Label($"Baseline σ: {baselinePredictor.lastSigma:F3}");
    GUILayout.Label($"ONNX σ: {onnxPredictor.lastSigma:F3}");
    GUILayout.Label($"Baseline MAE: {baselineMAE:F3}");
    GUILayout.Label($"ONNX MAE: {onnxMAE:F3}");
}
```

### 5.2 实时误差统计

```csharp
public class PredictorEvaluator : MonoBehaviour
{
    private Queue<float> baselineErrors;
    private Queue<float> onnxErrors;
    
    public void RecordPrediction(Vector3 predicted, Vector3 actual, bool isONNX)
    {
        float error = (predicted - actual).magnitude;
        if (isONNX) onnxErrors.Enqueue(error);
        else baselineErrors.Enqueue(error);
        
        if (onnxErrors.Count > 100) onnxErrors.Dequeue();
        if (baselineErrors.Count > 100) baselineErrors.Dequeue();
    }
    
    public float GetMAE(bool isONNX)
    {
        var queue = isONNX ? onnxErrors : baselineErrors;
        return queue.Count > 0 ? queue.Average() : 0f;
    }
}
```

---

## 保留的模块 (无需改动)

| 模块 | 状态 | 理由 |
|---|---|---|
| `BlockWorld.cs` | ✅ 保留 | 区块探索与飞控物理无关 |
| `MinimapCubeRenderer.cs` | ✅ 保留 | UI功能 |
| `BoxWireframe.cs` | ✅ 保留 | 可视化 |
| `SimpleCameraRig.cs` | ✅ 保留 | 相机控制 |
| `BalloonState.cs` | ✅ 保留 | 简单状态容器 |
| `GameController.cs` | ⚠️ 小改 | 只需改field引用 |

---

## 实施顺序建议

### Week 1: 核心物理
1. Day 1-2: 实现 `TurbulenceFieldPhysical.GenerateVelocityField()`
2. Day 3-4: 实现 `AdvectDiffuse()` 并验证谱分布
3. Day 5: 集成到 `Bootstrap.cs`，确保可运行

### Week 2: 飞控重构
4. Day 6-7: 实现 `ObservationBuffer` + `BaselinePredictor`
5. Day 8: 实现 `AdaptiveFlightController`
6. Day 9: 扩展 `DataLogger`
7. Day 10: 测试与调参

### Week 3: ONNX集成
8. Day 11-12: Python训练脚本（基于现有代码）
9. Day 13: 导出ONNX并集成 `ONNXPredictor`
10. Day 14-15: 对比验证与性能优化

---

## 验证清单

### 物理正确性
- [ ] 谱分布: E(k) ∝ k^(-α), α ∈ [1.65, 1.90]
- [ ] 无散度: ∇·u < 1e-3 (数值误差范围内)
- [ ] 雷诺数效应: nu增大 → 高频衰减加快
- [ ] 各向异性: u_x 方差 / u_y 方差 ≈ 1.25/0.80

### 飞控性能
- [ ] Baseline预测误差 < 0.5 (归一化)
- [ ] ONNX预测误差 < Baseline * 0.7
- [ ] 自适应增益: σ↑ → gain↓ (可观测)
- [ ] 无震荡: 控制输出平滑

### 数据质量
- [ ] CSV包含完整K步历史
- [ ] 梯度数值稳定 (无NaN/Inf)
- [ ] 采样率 ≥ 20Hz

---

## 现实部署路径

```
Unity仿真 (Von Kármán湍流)
    ↓ 生成10k+ episodes
Python训练 (Attention-Residual Transformer)
    ↓ NLL loss + Helmholtz-PGD
Unity验证 (ONNX推理)
    ↓ MAE < 0.3, σ校准良好
嵌入式移植 (ONNX Runtime C++)
    ↓ 树莓派4 / Jetson Nano
真实氢气球飞艇
```

---

## 关键差异对比

| 维度 | 当前Perlin | 重构后Von Kármán |
|---|---|---|
| 谱分布 | 无物理依据 | E(k)∝k^(-1.7±0.15) |
| 演化 | 静态采样 | 平流-扩散PDE |
| 无散度 | ❌ | ✅ Helmholtz投影 |
| 雷诺数 | 假参数 | 真实粘性衰减 |
| 预测器 | 5帧平均 | Transformer+(μ,σ) |
| 控制律 | 固定增益 | 自适应(基于σ) |
| 可迁移性 | ❌ | ✅ 物理一致 |

---

## 联系人与资源

- **参考代码**: `transfer_learning_experiment.py` (已提供)
- **Unity Sentis文档**: https://docs.unity3d.com/Packages/com.unity.sentis@latest
- **Von Kármán谱**: Pope, "Turbulent Flows" (2000), Chapter 6
- **Helmholtz分解**: Chorin & Marsden, "A Mathematical Introduction to Fluid Mechanics" (1993)

---

**最后提醒**: 当前的Perlin噪声模拟器**不能**用于任何现实部署。必须完成Phase 1和Phase 2才能产生有意义的飞控训练数据。
