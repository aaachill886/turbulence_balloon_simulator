using System;
using System.Collections.Generic;
using UnityEngine;

namespace BalloonSim.Sim
{
    public enum WaypointSpawnMode
    {
        UnlockedOnly = 0,
        NewOnly = 1,
        Both = 2,
    }

    [Serializable]
    public struct Int3 : IEquatable<Int3>
    {
        public int x;
        public int y;
        public int z;

        public Int3(int x, int y, int z) { this.x = x; this.y = y; this.z = z; }

        public bool Equals(Int3 other) => x == other.x && y == other.y && z == other.z;
        public override bool Equals(object obj) => obj is Int3 other && Equals(other);
        public override int GetHashCode() => HashCode.Combine(x, y, z);
        public static Int3 operator +(Int3 a, Int3 b) => new Int3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    public class BlockWorld : MonoBehaviour
    {
        public SimulationConfig config;

        [Header("Blocks")]
        public bool enableBlockGeneration = true;
        public WaypointSpawnMode spawnMode = WaypointSpawnMode.Both;

        [Tooltip("How close counts as reaching the waypoint.")]
        public float waypointReachDist = 1.2f;

        public Int3 CurrentBlock { get; private set; }
        public HashSet<Int3> UnlockedBlocks => _unlocked;

        private readonly HashSet<Int3> _unlocked = new();
        private readonly List<Int3> _candidateUnlocked = new();

        private static readonly Int3[] SixNeighbors =
        {
            new Int3(1, 0, 0),
            new Int3(-1, 0, 0),
            new Int3(0, 1, 0),
            new Int3(0, -1, 0),
            new Int3(0, 0, 1),
            new Int3(0, 0, -1),
        };

        private bool GenEnabled => config != null ? config.enableBlockGeneration : enableBlockGeneration;

        public void ResetWorld()
        {
            _unlocked.Clear();
            CurrentBlock = new Int3(0, 0, 0);
            _unlocked.Add(CurrentBlock);
        }

        private void Awake()
        {
            if (_unlocked.Count == 0) ResetWorld();
        }

        public bool IsUnlocked(Int3 b) => _unlocked.Contains(b);

        public Int3 GetBlockOf(Vector3 worldPos)
        {
            float s = config != null ? config.boxSize : 16f;
            return new Int3(
                Mathf.FloorToInt(worldPos.x / s),
                Mathf.FloorToInt(worldPos.y / s),
                Mathf.FloorToInt(worldPos.z / s)
            );
        }

        public void UpdateCurrentBlock(Vector3 worldPos)
        {
            CurrentBlock = GetBlockOf(worldPos);
        }

        public bool CheckReachWaypoint(Vector3 balloonPos, Vector3 waypointPos)
            => (balloonPos - waypointPos).magnitude <= waypointReachDist;

        public void UnlockNeighborsOfWaypoint(Vector3 waypointPos)
        {
            if (!GenEnabled) return;

            var wb = GetBlockOf(waypointPos);

            int pick = UnityEngine.Random.Range(0, 6);
            _unlocked.Add(wb + SixNeighbors[pick]);
        }

        public Vector3 SpawnWaypoint(System.Random rng, Vector3 aroundWorld)
        {
            float s = config != null ? config.boxSize : 16f;

            _candidateUnlocked.Clear();

            foreach (var b in _unlocked)
                _candidateUnlocked.Add(b);

            Int3 chosen = _candidateUnlocked.Count > 0 
                ? _candidateUnlocked[rng.Next(_candidateUnlocked.Count)] 
                : CurrentBlock;

            float min = config != null ? config.BoxMin : 0.6f;
            float max = config != null ? config.BoxMax : 15.4f;

            float x = (float)rng.NextDouble() * (max - min) + min;
            float y = (float)rng.NextDouble() * (max - min) + min;
            float z = (float)rng.NextDouble() * (max - min) + min;

            return new Vector3(chosen.x * s + x, chosen.y * s + y, chosen.z * s + z);
        }

        public Vector3 ClampToAllowed(Vector3 pos)
        {
            if (config == null) return pos;

            float s = config.boxSize;
            float margin = config.boxMinMargin;

            Int3 b = GetBlockOf(pos);

            if (!IsUnlocked(b))
                b = CurrentBlock;

            Vector3 origin = new Vector3(b.x * s, b.y * s, b.z * s);

            bool openNX = IsUnlocked(b + new Int3(-1, 0, 0));
            bool openPX = IsUnlocked(b + new Int3(1, 0, 0));
            bool openNY = IsUnlocked(b + new Int3(0, -1, 0));
            bool openPY = IsUnlocked(b + new Int3(0, 1, 0));
            bool openNZ = IsUnlocked(b + new Int3(0, 0, -1));
            bool openPZ = IsUnlocked(b + new Int3(0, 0, 1));

            float minX = origin.x + (openNX ? 0f : margin);
            float maxX = origin.x + s - (openPX ? 0f : margin);
            float minY = origin.y + (openNY ? 0f : margin);
            float maxY = origin.y + s - (openPY ? 0f : margin);
            float minZ = origin.z + (openNZ ? 0f : margin);
            float maxZ = origin.z + s - (openPZ ? 0f : margin);

            pos.x = Mathf.Clamp(pos.x, minX, maxX);
            pos.y = Mathf.Clamp(pos.y, minY, maxY);
            pos.z = Mathf.Clamp(pos.z, minZ, maxZ);

            return pos;
        }
    }
}
