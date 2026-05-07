using UnityEngine;

namespace BalloonSim.Sim
{
    public class FlowHeatmapRenderer : MonoBehaviour
    {
        public SimulationConfig config;
        public BlockWorld world;
        public TurbulenceField field;
        public BalloonState balloon;

        [Header("Semantic Grid")]
        [Range(1, 8)] public int gridStep = 4;
        [Range(0.03f, 0.2f)] public float pointSize = 0.06f;
        [Range(0f, 1f)] public float minAlpha = 0.08f;
        [Range(0f, 1f)] public float maxAlpha = 0.45f;
        [Range(1f, 80f)] public float fadeDistance = 24f;

        private Material _mat;
        private Mesh _quad;

        private void Awake()
        {
            if (_mat == null)
                _mat = new Material(Shader.Find("BalloonSim/FlowHeatmap"));
            _quad = BuildQuad();
        }

        private static Mesh BuildQuad()
        {
            var m = new Mesh();
            m.vertices = new[]
            {
                new Vector3(-0.5f, -0.5f, 0f),
                new Vector3( 0.5f, -0.5f, 0f),
                new Vector3( 0.5f,  0.5f, 0f),
                new Vector3(-0.5f,  0.5f, 0f),
            };
            m.uv = new[] { new Vector2(0,0), new Vector2(1,0), new Vector2(1,1), new Vector2(0,1) };
            m.triangles = new[] { 0, 1, 2, 0, 2, 3 };
            m.RecalculateBounds();
            return m;
        }

        private void OnRenderObject()
        {
            if (config == null || world == null || field == null || _mat == null || _quad == null) return;
            if (!config.showHeatmapPoints) return;

            Camera cam = Camera.current;
            if (cam == null) return;

            Vector3 bpos = balloon != null ? balloon.transform.position : Vector3.zero;
            Transform frame = Camera.main != null ? Camera.main.transform : null;
            Quaternion rot = frame != null ? frame.rotation : Quaternion.identity;
            Vector3 bestDir = ResolveBestKeyDirection(bpos, rot);

            float s = config.boxSize;
            float cell = s / 16f;

            _mat.SetPass(0);

            foreach (var block in world.UnlockedBlocks)
            {
                Vector3 blockBase = new Vector3(block.x * s, block.y * s, block.z * s);

                for (int x = 0; x < 16; x += gridStep)
                for (int y = 0; y < 16; y += gridStep)
                for (int z = 0; z < 16; z += gridStep)
                {
                    Vector3 p = blockBase + new Vector3(x * cell, y * cell, z * cell);

                    Vector3 u = field.Sample(p);
                    if (float.IsNaN(u.x) || float.IsNaN(u.y) || float.IsNaN(u.z)) continue;

                    float d = (p - bpos).magnitude;
                    float fade = Mathf.Clamp01(d / fadeDistance);
                    float alpha = Mathf.Lerp(maxAlpha, minAlpha, fade);

                    Color c = SemanticColor(u, p, bpos, bestDir, alpha);
                    _mat.SetColor("_Color", c);

                    var mtx = Matrix4x4.TRS(p, cam.transform.rotation, Vector3.one * pointSize);
                    Graphics.DrawMeshNow(_quad, mtx);
                }
            }
        }

        private static Color SemanticColor(Vector3 u, Vector3 p, Vector3 bpos, Vector3 bestDir, float alpha)
        {
            if (u.sqrMagnitude < 1e-8f) return new Color(0.2f, 0.9f, 0.9f, alpha * 0.6f);

            Vector3 un = u.normalized;
            float assist = Vector3.Dot(un, bestDir); // + best path
            Vector3 toBalloon = (bpos - p).normalized;
            float attract = Vector3.Dot(un, toBalloon); // + toward balloon

            if (assist > 0.35f)
            {
                float t = Mathf.Clamp01((assist - 0.35f) / 0.65f);
                return Color.Lerp(new Color(0.2f, 0.75f, 0.35f, alpha), new Color(0.05f, 1f, 0.15f, alpha), t); // green
            }

            if (attract > 0.25f)
            {
                float t = Mathf.Clamp01((attract - 0.25f) / 0.75f);
                return Color.Lerp(new Color(0.15f, 0.45f, 1f, alpha), new Color(0.05f, 0.25f, 1f, alpha), t); // blue
            }

            float repel = Mathf.Clamp01(-assist);
            return Color.Lerp(new Color(0.85f, 0.45f, 0.1f, alpha), new Color(1f, 0.08f, 0.08f, alpha), repel); // red/orange
        }

        private Vector3 ResolveBestKeyDirection(Vector3 pos, Quaternion rot)
        {
            Vector3[] locals = { Vector3.forward, Vector3.back, Vector3.left, Vector3.right, Vector3.down, Vector3.up };

            float best = float.MaxValue;
            Vector3 bestDir = Vector3.forward;

            for (int i = 0; i < locals.Length; i++)
            {
                Vector3 d = rot * locals[i];
                Vector3 u = field.Sample(pos + d * 0.8f);
                float resist = -Vector3.Dot(u, d);
                if (resist < best)
                {
                    best = resist;
                    bestDir = d;
                }
            }

            return bestDir.normalized;
        }
    }
}
