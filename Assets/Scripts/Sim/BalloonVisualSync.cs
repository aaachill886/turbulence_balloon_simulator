using UnityEngine;

namespace BalloonSim.Sim
{
    public class BalloonVisualSync : MonoBehaviour
    {
        public SimulationConfig config;

        private void LateUpdate()
        {
            if (config == null) return;
            float d = Mathf.Max(0.1f, config.balloonRadius * 2f);
            if (Mathf.Abs(transform.localScale.x - d) > 1e-4f)
                transform.localScale = new Vector3(d, d, d);
        }
    }
}
