using UnityEngine;

namespace BalloonSim.Sim
{
    /// <summary>
    /// 操控代价体积雾 — "哪个方向最省力"可视化
    /// 
    /// 语义：
    ///   🔵 蓝色 = 流场加速区（吸引力），往这里飞会被气流推着走
    ///   🟢 绿色 = 省力走廊，几乎不受阻力也不受加速
    ///   🔴 红色 = 阻力区（斥力），往这里飞需要对抗逆流
    ///   
    /// 物理定义：
    ///   cost(p) = -dot(u(p), d_hat) + λ·wake(p)
    ///   d_hat = normalize(p - balloon_pos)
    ///   
    ///   cost < 0 → 蓝（流场顺着你飞向p的方向推）
    ///   cost ≈ 0 → 绿（中性）
    ///   cost > 0 → 红（流场逆着你飞向p的方向推）
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public class VolumetricFogSimple : MonoBehaviour
    {
        public SimulationConfig config;
        public TurbulenceField field;
        public BalloonState balloon;

        [Header("Quality")]
        [Range(24, 800)] public int points = 300;
        [Range(0.05f, 1.5f)] public float lineLength = 0.4f;
        [Range(0.01f, 0.2f)] public float lineThickness = 0.04f;

        [Header("Alpha & Visibility")]
        [Range(0f, 1f)] public float baseAlpha = 0.32f;
        [Range(0f, 1f)] public float minAlpha = 0.08f;

        [Header("Range")]
        public float radius = 7.5f;
        [Range(0.0f, 1.0f)] public float innerMask = 0.85f;

        [Header("Cost Field Tuning")]
        [Tooltip("cost ∈ [-greenBand, +greenBand] 显示为绿色")]
        [Range(0.01f, 0.5f)] public float greenBand = 0.12f;

        [Tooltip("尾流对代价的贡献权重")]
        [Range(0f, 2f)] public float wakeCostWeight = 0.6f;

        [Tooltip("代价饱和值（超过此值颜色不再变深）")]
        [Range(0.1f, 3f)] public float costSaturation = 1.2f;

        [Tooltip("是否叠加沿途积分（路径代价）而非单点")]
        public bool usePathIntegral = false;

        [Tooltip("路径积分采样步数")]
        [Range(2, 6)] public int pathSteps = 3;

        // ── 预定义颜色 ──
        private static readonly Color DeepBlue = new Color(0.1f, 0.3f, 1.0f);
        private static readonly Color LightBlue = new Color(0.3f, 0.6f, 0.9f);
        private static readonly Color Green = new Color(0.15f, 0.9f, 0.35f);
        private static readonly Color LightRed = new Color(0.9f, 0.4f, 0.2f);
        private static readonly Color DeepRed = new Color(1.0f, 0.08f, 0.08f);

        private Material _mat;
        private Mesh _lineMesh;
        private Vector3[] _samplePoints;

        // ── 帧间平滑 ──
        private float[] _prevCost;
        private const float SmoothRate = 0.15f;

        private void Awake()
        {
            _mat = new Material(Shader.Find("BalloonSim/FlowHeatmap"));
            _lineMesh = BuildLineArrow();
            RegenerateSamples();
        }

        /// <summary>
        /// 重新生成采样点分布。
        /// 使用分层球面采样（Stratified Spherical）替代纯随机，
        /// 确保各方向均匀覆盖——这对"哪个方向省力"的判断至关重要。
        /// </summary>
        private void RegenerateSamples()
        {
            _samplePoints = new Vector3[points];
            _prevCost = new float[points];

            // 分层球面采样：黄金螺旋 + 半径分层
            float goldenAngle = Mathf.PI * (3f - Mathf.Sqrt(5f));

            for (int i = 0; i < points; i++)
            {
                // 球面均匀分布（Fibonacci sphere）
                float t = (float)i / (points - 1);
                float phi = goldenAngle * i;
                float cosTheta = 1f - 2f * t;
                float sinTheta = Mathf.Sqrt(1f - cosTheta * cosTheta);

                Vector3 dir = new Vector3(
                    sinTheta * Mathf.Cos(phi),
                    cosTheta,
                    sinTheta * Mathf.Sin(phi)
                );

                // 半径分层：内层密、外层疏（立方根分布保证体积均匀）
                float rFrac = Mathf.Pow(Random.Range(0.15f, 1f), 1f / 3f);
                _samplePoints[i] = dir * rFrac;
                _prevCost[i] = 0f;
            }
        }

        /// <summary>
        /// 计算从气球飞向目标点 p 的操控代价。
        /// 
        /// 单点模式：cost = -dot(u(p), d_hat) + λ·wake(p)
        /// 路径积分模式：cost = Σ_k [-dot(u(p_k), d_hat) + λ·wake(p_k)] / K
        ///   其中 p_k 是 balloon → p 路径上的均匀采样点
        /// </summary>
        private float ComputeCost(Vector3 bpos, Vector3 bvel, Vector3 targetPoint)
        {
            Vector3 delta = targetPoint - bpos;
            float dist = delta.magnitude;
            if (dist < 0.01f) return 0f;

            Vector3 dHat = delta / dist;

            if (!usePathIntegral)
            {
                // ── 单点代价 ──
                Vector3 u = field.Sample(targetPoint);
                float flowAssist = Vector3.Dot(u, dHat);         // 正=顺风，负=逆风
                float wakePenalty = field.WakeDot(targetPoint, bpos, bvel);  // 正=排斥，负=吸引
                return -flowAssist + wakeCostWeight * wakePenalty;
            }
            else
            {
                // ── 路径积分代价 ──
                // 沿 balloon→target 路径均匀采样，累加代价
                // 这能捕捉"中途有逆风墙"的情况
                float totalCost = 0f;
                for (int k = 1; k <= pathSteps; k++)
                {
                    float frac = (float)k / (pathSteps + 1);
                    Vector3 pk = bpos + delta * frac;
                    Vector3 uk = field.Sample(pk);
                    float assist = Vector3.Dot(uk, dHat);
                    float wake = field.WakeDot(pk, bpos, bvel);
                    totalCost += -assist + wakeCostWeight * wake;
                }
                return totalCost / pathSteps;
            }
        }

        /// <summary>
        /// 将 cost 值映射为颜色。
        /// 
        /// cost < -greenBand  →  蓝色（加速区/吸引力）
        /// |cost| ≤ greenBand →  绿色（省力走廊）
        /// cost > +greenBand  →  红色（阻力区/斥力）
        /// 
        /// 颜色深浅 ∝ |cost| 的大小（饱和于 costSaturation）
        /// </summary>
        private Color CostToColor(float cost, float alpha)
        {
            if (cost < -greenBand)
            {
                // 蓝色区：加速/吸引
                float t = Mathf.Clamp01((-cost - greenBand) / costSaturation);
                Color c = Color.Lerp(LightBlue, DeepBlue, t);
                c.a = Mathf.Lerp(alpha * 0.7f, alpha, t);  // 越强越不透明
                return c;
            }
            else if (cost > greenBand)
            {
                // 红色区：阻力/排斥
                float t = Mathf.Clamp01((cost - greenBand) / costSaturation);
                Color c = Color.Lerp(LightRed, DeepRed, t);
                c.a = Mathf.Lerp(alpha * 0.7f, alpha, t);
                return c;
            }
            else
            {
                // 绿色区：省力走廊
                // 越接近 cost=0，绿色越亮
                float centeredness = 1f - Mathf.Abs(cost) / greenBand;
                Color c = Green;
                c.a = alpha * (0.5f + 0.5f * centeredness);
                return c;
            }
        }

        private void OnRenderObject()
        {
            if (config == null || field == null || _mat == null || _samplePoints == null) return;
            if (!config.showVolumetricFog) return;

            Camera cam = Camera.current;
            if (cam == null || cam != GetComponent<Camera>()) return;

            Vector3 bpos = balloon != null ? balloon.transform.position : new Vector3(8f, 8f, 8f);
            Vector3 bvel = balloon != null ? balloon.velocity : Vector3.zero;

            float rSkip = config.balloonRadius * innerMask;

            _mat.SetPass(0);

            for (int i = 0; i < _samplePoints.Length; i++)
            {
                Vector3 offset = _samplePoints[i] * radius;
                if (offset.magnitude < rSkip) continue;

                Vector3 p = bpos + offset;

                // ── 采样流场方向（用于箭头朝向） ──
                Vector3 u = field.Sample(p);
                if (float.IsNaN(u.x) || float.IsNaN(u.y) || float.IsNaN(u.z)) continue;

                float mag = u.magnitude;
                if (mag < 0.01f) continue;

                // ── 计算操控代价 ──
                float rawCost = ComputeCost(bpos, bvel, p);

                // ── 帧间平滑（避免闪烁） ──
                _prevCost[i] = Mathf.Lerp(_prevCost[i], rawCost, SmoothRate);
                float cost = _prevCost[i];

                // ── 透明度：基于流场强度（弱流场更透明） ──
                float intensity = Mathf.Clamp01(mag / (0.8f + config.beaufort * 0.2f));
                float alpha = Mathf.Lerp(minAlpha, baseAlpha, intensity);

                // ── 着色 ──
                Color c = CostToColor(cost, alpha);

                // ── 箭头尺寸：基于流场强度 ──
                float scale = lineLength * (0.6f + 0.4f * intensity);

                // ── 绘制 ──
                Quaternion rot = Quaternion.LookRotation(u.normalized);
                var mtx = Matrix4x4.TRS(p, rot, new Vector3(lineThickness, lineThickness, scale));
                _mat.SetColor("_Color", c);
                Graphics.DrawMeshNow(_lineMesh, mtx);
            }
        }

        /// <summary>
        /// 查询特定方向的操控代价（供UI显示"最佳方向"用）
        /// </summary>
        public float QueryDirectionCost(Vector3 direction, float distance = 3f)
        {
            if (balloon == null || field == null) return 0f;
            Vector3 bpos = balloon.transform.position;
            Vector3 bvel = balloon.velocity;
            Vector3 target = bpos + direction.normalized * distance;
            return ComputeCost(bpos, bvel, target);
        }

        /// <summary>
        /// 找到当前最省力的方向（供飞控参考）
        /// 在6个主轴方向 + 8个对角方向中搜索
        /// </summary>
        public Vector3 FindBestDirection(float distance = 3f)
        {
            if (balloon == null || field == null) return Vector3.zero;

            Vector3[] candidates = {
                Vector3.right, Vector3.left,
                Vector3.up, Vector3.down,
                Vector3.forward, Vector3.back,
                (Vector3.right + Vector3.forward).normalized,
                (Vector3.right + Vector3.back).normalized,
                (Vector3.left + Vector3.forward).normalized,
                (Vector3.left + Vector3.back).normalized,
                (Vector3.right + Vector3.up).normalized,
                (Vector3.left + Vector3.up).normalized,
                (Vector3.right + Vector3.down).normalized,
                (Vector3.left + Vector3.down).normalized,
            };

            float bestCost = float.MaxValue;
            Vector3 bestDir = Vector3.zero;

            Vector3 bpos = balloon.transform.position;
            Vector3 bvel = balloon.velocity;

            foreach (var dir in candidates)
            {
                Vector3 target = bpos + dir * distance;
                float cost = ComputeCost(bpos, bvel, target);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestDir = dir;
                }
            }

            return bestDir;
        }

        private static Mesh BuildLineArrow()
        {
            var m = new Mesh();
            m.vertices = new[]
            {
                new Vector3(0, 0, 0), new Vector3(0, 0, 1),
                new Vector3(0, 0, 0.8f), new Vector3(-0.08f, 0, 0.95f),
                new Vector3(0, 0, 0.8f), new Vector3(0.08f, 0, 0.95f),
            };
            m.SetIndices(new[] { 0, 1, 2, 3, 4, 5 }, MeshTopology.Lines, 0);
            m.RecalculateBounds();
            return m;
        }
    }
}
