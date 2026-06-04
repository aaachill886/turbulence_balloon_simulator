using System.Collections.Generic;
using UnityEngine;

namespace BalloonSim.Sim
{
    public class ObservationBuffer : MonoBehaviour
    {
        public int historyLength = 8;
        public float sampleSpacing = 0.5f;

        [System.Serializable]
        public struct ObservationFrame
        {
            public float t;
            public Vector3 pos;
            public Vector3 vel;
            public Vector3[] windSamples; // 27 samples
        }

        private readonly Queue<ObservationFrame> _hist = new();

        public int Count => _hist.Count;

        public void Push(Vector3 pos, Vector3 vel, TurbulenceField field)
        {
            if (field == null) return;

            Vector3[] samples = new Vector3[27];
            int idx = 0;
            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)
            {
                Vector3 off = new Vector3(dx, dy, dz) * sampleSpacing;
                samples[idx++] = field.Sample(pos + off);
            }

            _hist.Enqueue(new ObservationFrame
            {
                t = Time.time,
                pos = pos,
                vel = vel,
                windSamples = samples
            });

            while (_hist.Count > historyLength) _hist.Dequeue();
        }

        public ObservationFrame[] Snapshot()
        {
            var arr = new ObservationFrame[_hist.Count];
            _hist.CopyTo(arr, 0);
            return arr;
        }
    }
}
