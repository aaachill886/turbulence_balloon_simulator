using UnityEngine;

namespace BalloonSim.Sim
{
    [RequireComponent(typeof(Camera))]
    public class VolumetricFogPhysical : MonoBehaviour
    {
        [Header("References")]
        public SimulationConfig config;
        public TurbulenceField field;
        public BalloonState balloon;

        [Header("Particle Distribution")]
        [Range(100, 3000)] public int particleCount = 1400;
        public float radius = 8f;
        [Range(0f, 2f)] public float innerMaskRadius = 0.6f;

        [Header("Particle Appearance")]
        [Range(0.1f, 3.0f)] public float baseParticleSize = 1.4f;
        [Range(0f, 5f)] public float sizeByIntensity = 3.0f;
        [Range(0.5f, 8f)] public float maxParticleSize = 5.5f;
        [Range(0f, 0.6f)] public float baseAlpha = 0.11f;
        [Range(0f, 1f)] public float maxAlpha = 0.28f;
        [Range(0f, 0.1f)] public float cutoffAlpha = 0.008f;

        [Header("Color Mapping")]
        public Color calmColor = new Color(0.7f, 0.75f, 0.8f);
        [Range(0f, 5f)] public float colorIntensityScale = 2.5f;

        [Header("Dynamics")]
        [Range(0f, 1f)] public float driftSpeed = 0.3f;
        [Range(0f, 0.95f)] public float smoothing = 0.85f;
        [Range(0f, 0.5f)] public float jitter = 0.15f;
        public float respawnInterval = 4f;

        [Header("Debug")]
        [SerializeField] private float _dbgCostMean;
        [SerializeField] private float _dbgCostStd;
        [SerializeField] private float _dbgMaxMag;

        private Material _mat;
        private Mesh _quadMesh;
        private ParticleState[] _particles;
        private Camera _cam;
        private float _time;
        private float _smoothCostMean;
        private float _smoothCostStd;
        private float _smoothMaxMag;

        private struct ParticleState
        {
            public Vector3 currentOffset;
            public float age;
            public float lifetime;
            public Color smoothedColor;
            public float smoothedSize;
            public float phaseOffset;
            public float lastCost;
            public float lastMag;
        }

        private void Awake()
        {
            _cam = GetComponent<Camera>();
            _mat = CreateFogMaterial();
            _quadMesh = BuildQuad();
            RegenerateParticles();
            _smoothCostStd = 0.1f;
            _smoothMaxMag = 0.5f;
        }

        private Material CreateFogMaterial()
        {
            Shader sh = Shader.Find("BalloonSim/FogParticle");
            if (sh == null) sh = Shader.Find("Particles/Standard Unlit");
            if (sh == null) sh = Shader.Find("Unlit/Transparent");
            if (sh == null) sh = Shader.Find("Hidden/InternalErrorShader");

            Material mat = new Material(sh);
            mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.One);
            mat.SetInt("_ZWrite", 0);
            mat.renderQueue = 3100;
            mat.mainTexture = GenerateRadialGradient(64);
            return mat;
        }

        private Texture2D GenerateRadialGradient(int size)
        {
            Texture2D tex = new Texture2D(size, size, TextureFormat.RGBA32, false);
            float center = size * 0.5f;
            for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                float dx = (x - center) / center;
                float dy = (y - center) / center;
                float dist = Mathf.Sqrt(dx * dx + dy * dy);
                float alpha = dist > 1f ? 0f : Mathf.Exp(-2.5f * dist * dist) * Mathf.SmoothStep(1f, 0.5f, dist);
                tex.SetPixel(x, y, new Color(1f, 1f, 1f, alpha));
            }
            tex.Apply();
            tex.wrapMode = TextureWrapMode.Clamp;
            tex.filterMode = FilterMode.Bilinear;
            return tex;
        }

        private void RegenerateParticles()
        {
            int count = config != null ? (int)config.fogParticleCount : particleCount;
            _particles = new ParticleState[count];
            float golden = Mathf.PI * (3f - Mathf.Sqrt(5f));
            for (int i = 0; i < count; i++)
            {
                float t = (float)i / (count - 1);
                float phi = golden * i;
                float cosTheta = 1f - 2f * t;
                float sinTheta = Mathf.Sqrt(1f - cosTheta * cosTheta);
                Vector3 dir = new Vector3(sinTheta * Mathf.Cos(phi), cosTheta, sinTheta * Mathf.Sin(phi));
                float rFrac = Mathf.Pow(Random.Range(0.05f, 1f), 1f / 3f);
                _particles[i] = new ParticleState
                {
                    currentOffset = dir * rFrac * radius,
                    age = Random.Range(0f, respawnInterval),
                    lifetime = respawnInterval * Random.Range(0.7f, 1.3f),
                    phaseOffset = Random.Range(0f, Mathf.PI * 2f),
                };
            }
        }

        private void Update()
        {
            if (field == null || config == null || !config.showVolumetricFog) return;
            _time += Time.deltaTime;
            Vector3 bpos = balloon != null ? balloon.transform.position : Vector3.one * 8f;
            float dt = Time.deltaTime;
            float costSum = 0f, costSqSum = 0f, maxMag = 0f;
            int validCount = 0;

            for (int i = 0; i < _particles.Length; i++)
            {
                ref ParticleState p = ref _particles[i];
                p.age += dt;
                if (p.age > p.lifetime) { RespawnParticle(ref p); }

                Vector3 worldPos = bpos + p.currentOffset;
                Vector3 wind = field.Sample(worldPos);
                if (float.IsNaN(wind.x)) continue;

                p.currentOffset += wind * driftSpeed * dt;
                p.currentOffset += new Vector3(
                    Mathf.Sin(_time * 2.3f + p.phaseOffset) * jitter * dt,
                    Mathf.Sin(_time * 1.7f + p.phaseOffset + 1f) * jitter * dt,
                    Mathf.Sin(_time * 3.1f + p.phaseOffset + 2f) * jitter * dt);
                if (p.currentOffset.magnitude > radius * 1.2f) { RespawnParticle(ref p); }

                float dist = p.currentOffset.magnitude;
                float mag = wind.magnitude;
                float cost = 0f;
                if (dist > 0.01f && mag > 1e-6f)
                {
                    cost = -Vector3.Dot(wind, p.currentOffset / dist);
                }
                p.lastCost = cost;
                p.lastMag = mag;
                costSum += cost;
                costSqSum += cost * cost;
                if (mag > maxMag) maxMag = mag;
                validCount++;
            }

            if (validCount > 10)
            {
                float mean = costSum / validCount;
                float std = Mathf.Sqrt(Mathf.Max(0f, (costSqSum / validCount) - mean * mean));
                float adaptRate = 3f * dt;
                _smoothCostMean = Mathf.Lerp(_smoothCostMean, mean, adaptRate);
                _smoothCostStd = Mathf.Lerp(_smoothCostStd, Mathf.Max(std, 1e-4f), adaptRate);
                _smoothMaxMag = Mathf.Lerp(_smoothMaxMag, Mathf.Max(maxMag, 1e-4f), adaptRate);
                _dbgCostMean = _smoothCostMean;
                _dbgCostStd = _smoothCostStd;
                _dbgMaxMag = _smoothMaxMag;
            }
        }

        private void RespawnParticle(ref ParticleState p)
        {
            Vector3 dir = Random.onUnitSphere;
            float rFrac = Mathf.Pow(Random.Range(0.05f, 1f), 1f / 3f);
            p.currentOffset = dir * rFrac * radius;
            p.age = 0f;
            p.lifetime = respawnInterval * Random.Range(0.7f, 1.3f);
            p.phaseOffset = Random.Range(0f, Mathf.PI * 2f);
        }

        private void OnRenderObject()
        {
            if (config == null || field == null || _mat == null || !config.showVolumetricFog) return;
            Camera cam = Camera.current;
            if (cam == null || cam != _cam) return;

            Vector3 bpos = balloon != null ? balloon.transform.position : Vector3.one * 8f;
            _mat.SetPass(0);

            int targetCount = config != null ? (int)config.fogParticleCount : particleCount;
            if (_particles.Length != targetCount) { RegenerateParticles(); }

            for (int i = 0; i < _particles.Length; i++)
            {
                ref ParticleState p = ref _particles[i];
                if (p.currentOffset.magnitude < innerMaskRadius) continue;

                Color targetColor;
                float targetSize;
                ComputeParticleVisuals(p.lastCost, p.lastMag, p.currentOffset.magnitude, p.age, p.lifetime, out targetColor, out targetSize);
                if (targetColor.a < cutoffAlpha) continue;

                p.smoothedColor = Color.Lerp(p.smoothedColor, targetColor, 1f - smoothing);
                p.smoothedSize = Mathf.Lerp(p.smoothedSize, targetSize, 1f - smoothing);
                if (p.smoothedColor.a < cutoffAlpha) continue;

                Vector3 worldPos = bpos + p.currentOffset;
                Matrix4x4 mtx = Matrix4x4.TRS(worldPos, Quaternion.LookRotation(cam.transform.forward, cam.transform.up), Vector3.one * p.smoothedSize);
                _mat.SetColor("_Color", p.smoothedColor);
                Graphics.DrawMeshNow(_quadMesh, mtx);
            }
        }

        private void ComputeParticleVisuals(float cost, float mag, float dist, float age, float lifetime, out Color color, out float size)
        {
            float normCost = (_smoothCostStd > 1e-6f) ? (cost - _smoothCostMean) / _smoothCostStd : 0f;
            float intensity = Mathf.Clamp01(mag / Mathf.Max(_smoothMaxMag, 1e-4f));

            Color baseColor;
            if (normCost < -0.4f)
            {
                float t = Mathf.Clamp01((-normCost - 0.4f) / 1.5f);
                baseColor = Color.Lerp(new Color(0.2f, 0.85f, 0.7f), new Color(0.1f, 0.3f, 1.0f), t);
            }
            else if (normCost > 0.4f)
            {
                float t = Mathf.Clamp01((normCost - 0.4f) / 1.5f);
                baseColor = Color.Lerp(new Color(1.0f, 0.6f, 0.2f), new Color(1.0f, 0.1f, 0.05f), t);
            }
            else
            {
                float greenIntensity = 1f - Mathf.Abs(normCost) / 0.4f;
                baseColor = Color.Lerp(new Color(0.4f, 0.7f, 0.3f), new Color(0.15f, 0.95f, 0.35f), greenIntensity);
            }

            float saturation = Mathf.Max(0.2f, Mathf.Clamp01(Mathf.Sqrt(intensity) * colorIntensityScale));
            color = Color.Lerp(calmColor, baseColor, saturation);

            float alpha = baseAlpha + (maxAlpha - baseAlpha) * intensity;
            alpha *= (config != null ? config.fogAlpha : 0.25f) * 2f;
            alpha *= Mathf.Clamp01(1f - (dist / radius) * 0.25f);
            float lifeFrac = age / lifetime;
            if (lifeFrac < 0.1f) alpha *= lifeFrac / 0.1f;
            else if (lifeFrac > 0.85f) alpha *= (1f - lifeFrac) / 0.15f;
            color.a = alpha;

            float sizeScale = config != null ? config.fogParticleSize : 1.4f;
            size = Mathf.Min(baseParticleSize * sizeScale + intensity * sizeByIntensity, maxParticleSize);
            size *= 0.95f + 0.05f * Mathf.Sin(age * 2f);
        }

        private static Mesh BuildQuad()
        {
            Mesh m = new Mesh();
            m.vertices = new[] { new Vector3(-0.5f, -0.5f, 0), new Vector3(0.5f, -0.5f, 0), new Vector3(0.5f, 0.5f, 0), new Vector3(-0.5f, 0.5f, 0) };
            m.uv = new[] { new Vector2(0, 0), new Vector2(1, 0), new Vector2(1, 1), new Vector2(0, 1) };
            m.triangles = new[] { 0, 2, 1, 0, 3, 2 };
            m.RecalculateBounds();
            return m;
        }

        private void OnValidate()
        {
            if (Application.isPlaying && _particles != null && config != null && _particles.Length != (int)config.fogParticleCount)
            {
                RegenerateParticles();
            }
        }

        private void OnDisable()
        {
            if (_mat != null) DestroyImmediate(_mat);
        }
    }
}
