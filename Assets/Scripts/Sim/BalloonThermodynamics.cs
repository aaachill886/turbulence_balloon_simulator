using UnityEngine;

namespace BalloonSim.Sim
{
    /// <summary>
    /// 热力学气球模型：考虑温度、气体泄漏、大气密度分层
    /// 基于理想气体定律和静力学平衡
    /// </summary>
    public class BalloonThermodynamics : MonoBehaviour
    {
        [Header("Gas Properties")]
        [Tooltip("气体类型：氢气 M=2.016 g/mol, 氦气 M=4.003 g/mol")]
        public float gasMolarMass = 2.016f;
        [Tooltip("初始气体质量 (kg)")]
        public float initialGasMass = 0.5f;
        [Tooltip("气体泄漏率 (kg/s)，0表示无泄漏")]
        public float leakRate = 0.0f;

        [Header("Balloon Envelope")]
        [Tooltip("气球体积 (m³)")]
        public float volume = 10f;
        [Tooltip("气球外壳质量 (kg)")]
        public float envelopeMass = 2f;
        [Tooltip("有效载荷质量 (kg)")]
        public float payloadMass = 1f;

        [Header("Temperature")]
        [Tooltip("气体温度 (K)")]
        public float gasTemperature = 288f;
        [Tooltip("环境温度 (K)")]
        public float ambientTemperature = 288f;
        [Tooltip("太阳辐射加热率 (W/m²)")]
        public float solarRadiation = 0f;
        [Tooltip("对流散热系数 (W/(m²·K))")]
        public float convectionCoeff = 5f;
        [Tooltip("气球表面积 (m²)，0表示自动计算")]
        public float surfaceArea = 0f;

        [Header("Atmosphere Model")]
        [Tooltip("海平面大气密度 (kg/m³)")]
        public float rho0 = 1.225f;
        [Tooltip("大气标高 (m)")]
        public float scaleHeight = 8500f;
        [Tooltip("重力加速度 (m/s²)")]
        public float gravity = 9.81f;

        [Header("Runtime State")]
        [Tooltip("当前气体质量 (kg)")]
        public float currentGasMass;
        [Tooltip("当前浮力 (N)")]
        public float currentBuoyancy;
        [Tooltip("当前重力 (N)")]
        public float currentWeight;
        [Tooltip("净升力 (N)")]
        public float netLift;

        // 物理常数
        private const float R = 8.314f; // 气体常数 J/(mol·K)

        private void Awake()
        {
            currentGasMass = initialGasMass;
            if (surfaceArea <= 0f)
                surfaceArea = 4f * Mathf.PI * Mathf.Pow(Mathf.Pow(volume * 3f / (4f * Mathf.PI), 1f / 3f), 2f);
        }

        private void FixedUpdate()
        {
            float dt = Time.fixedDeltaTime;
            UpdateThermodynamics(dt);
        }

        private void UpdateThermodynamics(float dt)
        {
            // 1. 气体泄漏
            currentGasMass = Mathf.Max(0f, currentGasMass - leakRate * dt);

            // 2. 热传导
            float solarHeat = solarRadiation * surfaceArea;
            float convectionHeat = convectionCoeff * surfaceArea * (ambientTemperature - gasTemperature);
            float totalHeat = solarHeat + convectionHeat;

            // 气体比热容 (J/(kg·K))，氢气约14300，氦气约5193
            float specificHeat = gasMolarMass < 3f ? 14300f : 5193f;
            float dT = totalHeat * dt / (currentGasMass * specificHeat + 1e-6f);
            gasTemperature = Mathf.Clamp(gasTemperature + dT, 200f, 400f);

            // 3. 计算气体密度 (理想气体定律)
            // ρ = PM/(RT), P = ρ_air * g * h (简化为海平面压强)
            float P = 101325f; // Pa
            float gasDensity = (P * gasMolarMass * 0.001f) / (R * gasTemperature);

            // 4. 计算大气密度 (随高度变化)
            float altitude = transform.position.y;
            float airDensity = rho0 * Mathf.Exp(-altitude / scaleHeight);

            // 5. 浮力 (阿基米德原理)
            currentBuoyancy = airDensity * volume * gravity;

            // 6. 重力
            float totalMass = currentGasMass + envelopeMass + payloadMass;
            currentWeight = totalMass * gravity;

            // 7. 净升力
            netLift = currentBuoyancy - currentWeight;
        }

        /// <summary>
        /// 获取归一化浮力加速度 (m/s²)
        /// </summary>
        public float GetBuoyancyAcceleration()
        {
            float totalMass = currentGasMass + envelopeMass + payloadMass;
            if (totalMass < 1e-6f) return 0f;
            return netLift / totalMass;
        }

        /// <summary>
        /// 获取等效密度比 (兼容旧接口)
        /// </summary>
        public float GetEffectiveDensityRatio()
        {
            float altitude = transform.position.y;
            float airDensity = rho0 * Mathf.Exp(-altitude / scaleHeight);
            float P = 101325f;
            float gasDensity = (P * gasMolarMass * 0.001f) / (R * gasTemperature);
            return Mathf.Clamp01(gasDensity / (airDensity + 1e-6f));
        }

        /// <summary>
        /// 手动设置气体质量 (用于测试)
        /// </summary>
        public void SetGasMass(float mass)
        {
            currentGasMass = Mathf.Max(0f, mass);
        }

        /// <summary>
        /// 手动设置温度 (用于测试)
        /// </summary>
        public void SetTemperature(float temp)
        {
            gasTemperature = Mathf.Clamp(temp, 200f, 400f);
        }

        private void OnGUI()
        {
            if (!Application.isPlaying) return;

            GUILayout.BeginArea(new Rect(Screen.width - 310, 10, 300, 200), GUI.skin.box);
            GUILayout.Label("=== Balloon Thermodynamics ===");
            GUILayout.Label($"Gas Mass: {currentGasMass:F3} kg");
            GUILayout.Label($"Gas Temp: {gasTemperature:F1} K ({gasTemperature - 273.15f:F1} °C)");
            GUILayout.Label($"Altitude: {transform.position.y:F1} m");
            GUILayout.Label($"Buoyancy: {currentBuoyancy:F2} N");
            GUILayout.Label($"Weight: {currentWeight:F2} N");
            GUILayout.Label($"Net Lift: {netLift:F2} N ({GetBuoyancyAcceleration():F2} m/s²)");
            GUILayout.Label($"Density Ratio: {GetEffectiveDensityRatio():F3}");
            GUILayout.EndArea();
        }
    }
}
