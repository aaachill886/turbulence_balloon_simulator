using UnityEngine;

namespace BalloonSim.Sim
{
    public class BalloonState : MonoBehaviour
    {
        public Vector3 velocity;

        [Header("Attitude State")]
        public Vector3 forward = Vector3.forward;
        public float yawDeg;

        private void Awake()
        {
            forward = transform.forward;
            yawDeg = transform.eulerAngles.y;
        }
    }
}
