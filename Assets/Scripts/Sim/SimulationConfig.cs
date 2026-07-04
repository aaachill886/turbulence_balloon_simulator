using UnityEngine;

namespace BalloonSim.Sim
{
    [CreateAssetMenu(menuName = "BalloonSim/Simulation Config")]
    public class SimulationConfig : ScriptableObject
    {
        [Header("World")]
        public float boxSize = 16f;
        public float boxMinMargin = 0.6f;

        [Header("Balloon")]
        [Min(0.1f)] public float balloonRadius = 1f;
        [Tooltip("Density ratio ρ_balloon / ρ_air. <1 rises, >1 sinks")]
        [Range(0.05f, 3f)] public float densityRatio = 1f;

        [Header("Control")]
        [Min(0f)] public float throttle = 1f;
        [Min(0f)] public float manualSpeedScale = 0.22f;
        public bool enableAIPredictor = false;

        [Header("Assist/Hold")]
        [Range(0f, 4f)] public float holdPosK = 1.2f;
        [Range(0f, 4f)] public float holdVelK = 1.0f;
        [Range(0.5f, 8f)] public float holdMaxSpeed = 3.0f;
        [Range(0f, 2f)] public float holdForwardK = 0.25f;
        [Range(0f, 2f)] public float holdReleaseVelK = 0.20f;
        [Range(0f, 1f)] public float holdEnvComp = 0.0f;
        [Range(0f, 2f)] public float holdAltIK = 0.35f;
        [Tooltip("If ON, hold ignores environment and acts like hard lock (benchmark only)")]
        public bool strictHoldNoDrift = false;
        [Tooltip("Only for debug comparison; OFF means no oracle cancellation")]
        public bool assistUseOracleEnvCancellation = false;

        [Header("Flow - primary")]
        [Min(0f)] public float beaufort = 3.5f;
        [Min(0f)] public float viscosity = 0.2f;
        [Min(20f)] public float reynolds = 5000f;
        [Min(0f)] public float randomStrength = 1f;

        [Header("Flow - gust/convection")]
        [Min(0f)] public float gustStrength = 0f;
        [Range(-180f, 180f)] public float gustDirDeg = 0f;
        [Min(0f)] public float convectionStrength = 0f;
        public bool tornado = false;

        [Header("Wake")]
        [Min(0f)] public float wakeStrength = 1f;

        [Header("Autopilot")]
        public bool aiEnabled = false;
        [Min(0f)] public float aiGain = 0.8f;
        [Min(0f)] public float ecoKeep = 0.35f;
        [Tooltip("If ON, prediction uncertainty softly attenuates AI assistance")]
        public bool usePredSigmaConfidence = true;

        [Header("Attitude")]
        [Range(30f, 360f)] public float attitudeMaxDegPerSec = 140f;
        [Range(1f, 30f)] public float attitudeResponsiveness = 10f;

        [Header("Visualization")]
        public bool showHeatmapPoints = false;
        public bool showVolumetricFog = true;
        public bool showMinimap = true;
        [Range(0f, 3f)] public float semanticAlignW = 1.0f;
        [Range(0f, 3f)] public float semanticMagW = 0.7f;
        [Range(0f, 3f)] public float semanticGradW = 0.5f;
        [Range(0f, 1f)] public float semanticCalmBand = 0.15f;

        [Header("Volumetric Fog Tuning")]
        [Range(100, 3000)] public int fogParticleCount = 1400;
        [Range(0.1f, 3f)] public float fogParticleSize = 1.4f;
        [Range(0f, 1f)] public float fogAlpha = 0.25f;

        [Header("Exploration")]
        public bool enableBlockGeneration = true;
        [Range(0f, 2f)]
        public float waypointSpawnMode = 2f;

        [Header("Advanced coefficients")]
        public float visRangeK = 0.75f;
        public float visAmpK = 0.18f;
        public float wakeBaseR = 5f;
        public float wakeSizeR = 3f;
        public float buoyK = 0.45f;
        public float aiSafePosK = 0.45f;
        public float aiSafeMaxK = 0.55f;
        public float aiSafeDistK = 0.35f;
        public float aiSafeVelK = 0.7f;
        public float aiNeedK = 0.35f;

        public float BoxMin => boxMinMargin;
        public float BoxMax => boxSize - boxMinMargin;
    }
}
