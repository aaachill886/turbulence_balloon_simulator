using UnityEngine;

namespace BalloonSim.Sim
{
    public static class MathUtil
    {
        public static float Clamp(float v, float a, float b) => Mathf.Max(a, Mathf.Min(b, v));
        public static float Wrap(float v, float period)
        {
            if (period <= 0f) return v;
            v %= period;
            if (v < 0f) v += period;
            return v;
        }

        public static Vector3 Wrap3(Vector3 v, float period)
            => new Vector3(Wrap(v.x, period), Wrap(v.y, period), Wrap(v.z, period));

        public static bool IsFinite(float f) => !(float.IsNaN(f) || float.IsInfinity(f));
        public static bool IsFinite(Vector3 v) => IsFinite(v.x) && IsFinite(v.y) && IsFinite(v.z);
    }
}
