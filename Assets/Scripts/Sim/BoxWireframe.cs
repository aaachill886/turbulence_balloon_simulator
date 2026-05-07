using UnityEngine;

namespace BalloonSim.Sim
{
    public class BoxWireframe : MonoBehaviour
    {
        public SimulationConfig config;
        public BlockWorld world;

        [Range(0.001f, 0.1f)] public float width = 0.015f;
        public Color color = new(0.55f, 0.78f, 1f, 0.85f);

        private LineRenderer _lr;

        private void Awake()
        {
            _lr = gameObject.GetComponent<LineRenderer>();
            if (_lr == null) _lr = gameObject.AddComponent<LineRenderer>();

            _lr.useWorldSpace = true;
            _lr.loop = false;
            _lr.widthMultiplier = width;
            _lr.material = new Material(Shader.Find("Sprites/Default"));
            _lr.startColor = color;
            _lr.endColor = color;
        }

        private void LateUpdate()
        {
            Rebuild();
        }

        public void Rebuild()
        {
            if (config == null) return;

            float s = config.boxSize;
            Vector3 origin = Vector3.zero;

            if (world != null)
                origin = new Vector3(world.CurrentBlock.x * s, world.CurrentBlock.y * s, world.CurrentBlock.z * s);

            float min = config.BoxMin;
            float max = config.BoxMax;

            Vector3[] c =
            {
                origin + new Vector3(min, min, min),
                origin + new Vector3(max, min, min),
                origin + new Vector3(max, max, min),
                origin + new Vector3(min, max, min),
                origin + new Vector3(min, min, max),
                origin + new Vector3(max, min, max),
                origin + new Vector3(max, max, max),
                origin + new Vector3(min, max, max),
            };

            // Base cube edges
            int[][] e =
            {
                new[] {0, 1}, new[] {1, 2}, new[] {2, 3}, new[] {3, 0},
                new[] {4, 5}, new[] {5, 6}, new[] {6, 7}, new[] {7, 4},
                new[] {0, 4}, new[] {1, 5}, new[] {2, 6}, new[] {3, 7},
            };

            var pts = new System.Collections.Generic.List<Vector3>(128);

            void AddEdge(int a, int b)
            {
                pts.Add(c[a]);
                pts.Add(c[b]);
                pts.Add(c[b]);
            }

            // For each face, skip edges that border an unlocked neighbor block.
            // Faces: -X, +X, -Y, +Y, -Z, +Z
            bool hasNX = world != null && world.UnlockedBlocks.Contains(world.CurrentBlock + new Int3(-1, 0, 0));
            bool hasPX = world != null && world.UnlockedBlocks.Contains(world.CurrentBlock + new Int3(1, 0, 0));
            bool hasNY = world != null && world.UnlockedBlocks.Contains(world.CurrentBlock + new Int3(0, -1, 0));
            bool hasPY = world != null && world.UnlockedBlocks.Contains(world.CurrentBlock + new Int3(0, 1, 0));
            bool hasNZ = world != null && world.UnlockedBlocks.Contains(world.CurrentBlock + new Int3(0, 0, -1));
            bool hasPZ = world != null && world.UnlockedBlocks.Contains(world.CurrentBlock + new Int3(0, 0, 1));

            // Edges per face (by corner indices)
            // -X face uses corners 0,3,7,4
            if (!hasNX) { AddEdge(0, 3); AddEdge(3, 7); AddEdge(7, 4); AddEdge(4, 0); }
            // +X face uses corners 1,2,6,5
            if (!hasPX) { AddEdge(1, 2); AddEdge(2, 6); AddEdge(6, 5); AddEdge(5, 1); }
            // -Y face uses corners 0,1,5,4
            if (!hasNY) { AddEdge(0, 1); AddEdge(1, 5); AddEdge(5, 4); AddEdge(4, 0); }
            // +Y face uses corners 3,2,6,7
            if (!hasPY) { AddEdge(3, 2); AddEdge(2, 6); AddEdge(6, 7); AddEdge(7, 3); }
            // -Z face uses corners 0,1,2,3
            if (!hasNZ) { AddEdge(0, 1); AddEdge(1, 2); AddEdge(2, 3); AddEdge(3, 0); }
            // +Z face uses corners 4,5,6,7
            if (!hasPZ) { AddEdge(4, 5); AddEdge(5, 6); AddEdge(6, 7); AddEdge(7, 4); }

            if (pts.Count == 0)
            {
                // If everything is open, draw nothing.
                _lr.positionCount = 0;
                return;
            }

            _lr.positionCount = pts.Count;
            _lr.SetPositions(pts.ToArray());
        }
    }
}
