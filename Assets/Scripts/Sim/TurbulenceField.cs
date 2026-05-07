using System;
using UnityEngine;

namespace BalloonSim.Sim
{
    public class TurbulenceField : MonoBehaviour
    {
        public SimulationConfig config;

        [Header("Physical Grid")]
        [SerializeField] private int grid = 16;

        [Header("Von Karman")]
        public float reTau = 1000f;
        public float nuMultiplier = 1.15f;
        public float slopeMin = 1.65f;
        public float slopeMax = 1.90f;
        public float anisotropyU = 1.25f;
        public float anisotropyV = 0.80f;

        [Header("Advection-Diffusion")]
        public float diffusionDt = 0.02f;
        public float advectionCoeffX = 0.10f;
        public float advectionCoeffY = 0.08f;
        public float advectionCoeffZ = 0.06f;
        public float rollCoupling01 = 0.07f;
        public float rollCoupling10 = 0.06f;
        public float rollCoupling20 = 0.05f;
        public float crossProduct01 = 0.04f;
        public float crossProduct12 = 0.03f;
        public float crossTanh02 = 0.03f;
        public float cubicDrag = 0.015f;
        public float wallDriveStrength = 0.0f;

        private Vector3[,,] _f;
        private Vector3[,,] _fn;
        private float _evolveTimer;

        private float Cell => config.boxSize / grid;
        private float Nu => (1f / Mathf.Max(1f, reTau)) * nuMultiplier;

        public void Initialize()
        {
            _f = new Vector3[grid, grid, grid];
            _fn = new Vector3[grid, grid, grid];
            Generate();
        }

        public void Generate()
        {
            if (_f == null) Initialize();
            Array.Clear(_f, 0, _f.Length);

            int maxModes = grid / 2;
            float slope = UnityEngine.Random.Range(slopeMin, slopeMax);
            var rng = new System.Random(UnityEngine.Random.Range(0, int.MaxValue));

            for (int kxi = -maxModes; kxi <= maxModes; kxi++)
            for (int kyi = -maxModes; kyi <= maxModes; kyi++)
            for (int kzi = -maxModes; kzi <= maxModes; kzi++)
            {
                float kx = kxi * 2f * Mathf.PI / grid;
                float ky = kyi * 2f * Mathf.PI / grid;
                float kz = kzi * 2f * Mathf.PI / grid;
                float kMag = Mathf.Sqrt(kx * kx + ky * ky + kz * kz);
                if (kMag < 0.1f * 2f * Mathf.PI / grid) continue;

                float Ek = Mathf.Pow(kMag, -slope);
                float amp = Mathf.Sqrt(Ek + 1e-8f) / (grid * grid * grid);

                float[] re = new float[3];
                float[] im = new float[3];
                for (int c = 0; c < 3; c++)
                {
                    re[c] = (float)NextGaussian(rng) * amp;
                    im[c] = (float)NextGaussian(rng) * amp;
                }

                re[0] *= anisotropyU; im[0] *= anisotropyU;
                re[1] *= anisotropyV; im[1] *= anisotropyV;

                float k2 = kMag * kMag + 1e-8f;
                float kdotR = kx * re[0] + ky * re[1] + kz * re[2];
                float kdotI = kx * im[0] + ky * im[1] + kz * im[2];
                float[] kv = { kx, ky, kz };
                for (int c = 0; c < 3; c++)
                {
                    re[c] -= kv[c] * kdotR / k2;
                    im[c] -= kv[c] * kdotI / k2;
                }

                for (int ix = 0; ix < grid; ix++)
                for (int iy = 0; iy < grid; iy++)
                for (int iz = 0; iz < grid; iz++)
                {
                    float ph = kx * ix + ky * iy + kz * iz;
                    float cph = Mathf.Cos(ph);
                    float sph = Mathf.Sin(ph);

                    _f[ix, iy, iz].x += re[0] * cph - im[0] * sph;
                    _f[ix, iy, iz].y += re[1] * cph - im[1] * sph;
                    _f[ix, iy, iz].z += re[2] * cph - im[2] * sph;
                }
            }
        }

        public void Step(float dt, Vector3 balloonPos, Vector3 balloonVel)
        {
            if (_f == null) Initialize();

            _evolveTimer += dt;
            float evolveInterval = 1f / 12.5f;
            while (_evolveTimer >= evolveInterval)
            {
                AdvectDiffuse();
                _evolveTimer -= evolveInterval;
            }
        }

        private void AdvectDiffuse()
        {
            float nuDt = Nu * diffusionDt;

            for (int ix = 0; ix < grid; ix++)
            for (int iy = 0; iy < grid; iy++)
            for (int iz = 0; iz < grid; iz++)
            {
                int xp = (ix + 1) % grid, xm = (ix - 1 + grid) % grid;
                int yp = (iy + 1) % grid, ym = (iy - 1 + grid) % grid;
                int zp = (iz + 1) % grid, zm = (iz - 1 + grid) % grid;

                Vector3 c = _f[ix, iy, iz];
                Vector3 lap =
                    _f[xp, iy, iz] + _f[xm, iy, iz] +
                    _f[ix, yp, iz] + _f[ix, ym, iz] +
                    _f[ix, iy, zp] + _f[ix, iy, zm] - 6f * c;

                Vector3 n = c + nuDt * lap;

                float gx = (_f[xp, iy, iz].x - _f[xm, iy, iz].x) * 0.5f;
                float gy = (_f[ix, yp, iz].y - _f[ix, ym, iz].y) * 0.5f;
                float gz = (_f[ix, iy, zp].z - _f[ix, iy, zm].z) * 0.5f;

                n.x -= advectionCoeffX * c.x * gx;
                n.y -= advectionCoeffY * c.y * gy;
                n.z -= advectionCoeffZ * c.z * gz;

                n.x += rollCoupling01 * _f[ix, ym, iz].y;
                n.y += rollCoupling10 * _f[ix, iy, zm].x;
                n.z += rollCoupling20 * _f[xm, iy, iz].x;

                n.x += crossProduct01 * c.x * c.y;
                n.y += crossProduct12 * c.y * c.z;
                n.z += crossTanh02 * TanhApprox(c.x * c.z);

                n.x -= cubicDrag * c.x * Mathf.Abs(c.x);
                n.y -= cubicDrag * c.y * Mathf.Abs(c.y);
                n.z -= cubicDrag * c.z * Mathf.Abs(c.z);

                float z01 = (float)iz / (grid - 1);
                n.x += wallDriveStrength * z01;

                n.x = Mathf.Clamp(n.x, -100f, 100f);
                n.y = Mathf.Clamp(n.y, -100f, 100f);
                n.z = Mathf.Clamp(n.z, -100f, 100f);

                _fn[ix, iy, iz] = n;
            }

            (_f, _fn) = (_fn, _f);
        }

        public Vector3 ExternalAt(Vector3 world)
        {
            float a = config.gustDirDeg * Mathf.Deg2Rad;
            Vector3 gu = new Vector3(Mathf.Sin(a), 0f, Mathf.Cos(a)) * (config.gustStrength * 0.09f);

            Vector3 d = world - new Vector3(11f, 8f, 8f);
            float dist = d.magnitude + 1e-6f;
            float loc = Mathf.Exp(-(dist * dist) / 10f) * config.convectionStrength * 0.14f;
            Vector3 sw = d / dist * loc;

            if (config.tornado)
            {
                float rx = world.x - 8f;
                float rz = world.z - 8f;
                float rr = Mathf.Sqrt(rx * rx + rz * rz) + 1e-6f;
                float v = 1.3f * Mathf.Exp(-(rr * rr) / 24f);
                sw.x += -rz / rr * v;
                sw.z += rx / rr * v;
                sw.y += 0.4f * Mathf.Exp(-(rr * rr) / 20f);
            }

            return gu + sw;
        }

        public float WakeDot(Vector3 world, Vector3 balloonPos, Vector3 balloonVel)
        {
            Vector3 d = world - balloonPos;
            float dist = d.magnitude + 1e-6f;
            float vr = balloonVel.magnitude;
            float dot = Vector3.Dot(balloonVel, d / dist);

            float area = config.balloonRadius * config.balloonRadius;
            float vis = Mathf.Clamp(config.viscosity, 0f, 8f);
            float visRange = 1f + vis * config.visRangeK;
            float visAmp = 1f / (1f + vis * config.visAmpK);

            float denom = (config.wakeBaseR + area * config.wakeSizeR) * visRange;
            return Mathf.Exp(-(dist * dist) / denom) * vr * config.wakeStrength * area * dot * visAmp;
        }

        public Vector3 Sample(Vector3 world)
        {
            float period = config.boxSize;
            world = MathUtil.Wrap3(world, period);

            float gx = world.x / Cell;
            float gy = world.y / Cell;
            float gz = world.z / Cell;

            int x0 = Mathf.FloorToInt(gx) % grid;
            int y0 = Mathf.FloorToInt(gy) % grid;
            int z0 = Mathf.FloorToInt(gz) % grid;
            int x1 = (x0 + 1) % grid;
            int y1 = (y0 + 1) % grid;
            int z1 = (z0 + 1) % grid;

            float fx = gx - Mathf.Floor(gx);
            float fy = gy - Mathf.Floor(gy);
            float fz = gz - Mathf.Floor(gz);

            Vector3 c000 = _f[x0, y0, z0];
            Vector3 c100 = _f[x1, y0, z0];
            Vector3 c010 = _f[x0, y1, z0];
            Vector3 c110 = _f[x1, y1, z0];
            Vector3 c001 = _f[x0, y0, z1];
            Vector3 c101 = _f[x1, y0, z1];
            Vector3 c011 = _f[x0, y1, z1];
            Vector3 c111 = _f[x1, y1, z1];

            Vector3 c00 = Vector3.Lerp(c000, c100, fx);
            Vector3 c10 = Vector3.Lerp(c010, c110, fx);
            Vector3 c01 = Vector3.Lerp(c001, c101, fx);
            Vector3 c11 = Vector3.Lerp(c011, c111, fx);

            Vector3 c0 = Vector3.Lerp(c00, c10, fy);
            Vector3 c1 = Vector3.Lerp(c01, c11, fy);

            return Vector3.Lerp(c0, c1, fz) + ExternalAt(world);
        }

        public float[,] SampleGradient(Vector3 worldPos, float delta = 0.3f)
        {
            float[,] J = new float[3, 3];
            for (int j = 0; j < 3; j++)
            {
                Vector3 off = Vector3.zero;
                off[j] = delta;
                Vector3 vp = Sample(worldPos + off);
                Vector3 vm = Sample(worldPos - off);
                for (int i = 0; i < 3; i++) J[i, j] = (vp[i] - vm[i]) / (2f * delta);
            }
            return J;
        }

        private static float TanhApprox(float x)
        {
            if (x > 3f) return 1f;
            if (x < -3f) return -1f;
            float x2 = x * x;
            return x * (27f + x2) / (27f + 9f * x2);
        }

        private static double NextGaussian(System.Random rng)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
    }
}
