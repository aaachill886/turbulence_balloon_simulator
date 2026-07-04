using UnityEngine;

namespace BalloonSim.Sim
{
    public class WaypointController : MonoBehaviour
    {
        public BlockWorld world;
        public Transform balloon;
        public Transform waypoint;

        [Header("Random")]
        public int seed = 12345;

        private System.Random _rng;

        private void Awake()
        {
            _rng = new System.Random(seed);

            if (world != null)
                world.ResetWorld();

            RespawnWaypoint();
        }

        public void RespawnWaypoint()
        {
            if (world == null || waypoint == null) return;

            Vector3 around = balloon != null ? balloon.position : Vector3.zero;
            waypoint.position = world.SpawnWaypoint(_rng, around);
        }

        private void Update()
        {
            if (world == null || balloon == null || waypoint == null) return;

            world.UpdateCurrentBlock(balloon.position);

            if (world.CheckReachWaypoint(balloon.position, waypoint.position))
            {
                world.UnlockNeighborsOfWaypoint(waypoint.position);
                RespawnWaypoint();
            }
        }
    }
}
