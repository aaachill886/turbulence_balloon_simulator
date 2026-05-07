using BalloonSim.UI;
using UnityEngine;

namespace BalloonSim.Sim
{
    public static class Bootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Init()
        {
            if (Object.FindObjectOfType<GameController>() != null)
                return;

            var cfg = ScriptableObject.CreateInstance<SimulationConfig>();
            cfg.aiEnabled = true; // default to control-assist mode

            var root = new GameObject("BalloonSim");

            var balloonGo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            balloonGo.name = "Balloon";
            balloonGo.transform.SetParent(root.transform);
            balloonGo.transform.position = new Vector3(8f, 8f, 8f);
            balloonGo.transform.localScale = Vector3.one * (cfg.balloonRadius * 2f);
            var balloon = balloonGo.AddComponent<BalloonState>();
            var balloonSync = balloonGo.AddComponent<BalloonVisualSync>();
            balloonSync.config = cfg;
            var thermo = balloonGo.AddComponent<BalloonThermodynamics>();

            var waypointGo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            waypointGo.name = "Waypoint";
            waypointGo.transform.SetParent(root.transform);
            waypointGo.transform.position = new Vector3(4f, 11f, 12f);
            waypointGo.transform.localScale = Vector3.one * 0.55f;
            var waypointRenderer = waypointGo.GetComponent<Renderer>();
            if (waypointRenderer != null) waypointRenderer.material.color = new Color(1f, 0.88f, 0.35f, 1f);

            var fieldGo = new GameObject("TurbulenceField");
            fieldGo.transform.SetParent(root.transform);
            var field = fieldGo.AddComponent<TurbulenceField>();
            field.config = cfg;
            field.Initialize();

            var apGo = new GameObject("Autopilot");
            apGo.transform.SetParent(root.transform);
            var ap = apGo.AddComponent<AutopilotController>();
            ap.config = cfg;
            ap.balloon = balloon;
            ap.field = field;

            var worldGo = new GameObject("BlockWorld");
            worldGo.transform.SetParent(root.transform);
            var world = worldGo.AddComponent<BlockWorld>();
            world.config = cfg;
            world.enableBlockGeneration = true;
            world.spawnMode = WaypointSpawnMode.Both;
            world.ResetWorld();

            var boxGo = new GameObject("BoxWireframe");
            boxGo.transform.SetParent(root.transform);
            var box = boxGo.AddComponent<BoxWireframe>();
            box.config = cfg;
            box.world = world;

            var camGo = new GameObject("Main Camera");
            camGo.tag = "MainCamera";
            var cam = camGo.AddComponent<Camera>();
            cam.nearClipPlane = 0.01f;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.02f, 0.03f, 0.08f, 1f);
            var rig = camGo.AddComponent<SimpleCameraRig>();
            rig.follow = balloonGo.transform;
            rig.distance = 16f;

            var lightGo = new GameObject("Directional Light");
            var light = lightGo.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.1f;
            lightGo.transform.rotation = Quaternion.Euler(55f, 35f, 0f);

            var wpGo = new GameObject("WaypointController");
            wpGo.transform.SetParent(root.transform);
            var wp = wpGo.AddComponent<WaypointController>();
            wp.world = world;
            wp.balloon = balloonGo.transform;
            wp.waypoint = waypointGo.transform;

            var obsGo = new GameObject("ObservationBuffer");
            obsGo.transform.SetParent(root.transform);
            var obs = obsGo.AddComponent<ObservationBuffer>();

            var onnxGo = new GameObject("ONNXPredictor");
            onnxGo.transform.SetParent(root.transform);
            var onnx = onnxGo.AddComponent<ONNXPredictor>();
            onnx.enableONNX = false;

            ap.observationBuffer = obs;
            ap.onnxPredictor = onnx;
            ap.thermodynamics = thermo;
            ap.ResetState();

            var gameGo = new GameObject("GameController");
            gameGo.transform.SetParent(root.transform);
            var game = gameGo.AddComponent<GameController>();
            game.config = cfg;
            game.field = field;
            game.balloon = balloon;
            game.autopilot = ap;
            game.world = world;
            game.inputFrame = camGo.transform;
            game.observationBuffer = obs;

            var heatGo = new GameObject("FlowHeatmap");
            heatGo.transform.SetParent(root.transform);
            var heat = heatGo.AddComponent<FlowHeatmapRenderer>();
            heat.config = cfg;
            heat.world = world;
            heat.field = field;
            heat.balloon = balloon;

            var fog = camGo.AddComponent<VolumetricFogPhysical>();
            fog.config = cfg;
            fog.field = field;
            fog.balloon = balloon;

            var miniGo = new GameObject("MinimapCamera");
            miniGo.transform.SetParent(root.transform);
            miniGo.AddComponent<Camera>();
            var mini = miniGo.AddComponent<MinimapCubeRenderer>();
            mini.config = cfg;
            mini.world = world;
            mini.balloon = balloonGo.transform;
            mini.waypoint = waypointGo.transform;

            var uiGo = new GameObject("SimUIOnGUI");
            uiGo.transform.SetParent(root.transform);
            var ui = uiGo.AddComponent<SimUIOnGUI>();
            ui.game = game;
            ui.field = field;
            ui.autopilot = ap;
            ui.balloon = balloon;
            ui.config = cfg;

            var logGo = new GameObject("DataLogger");
            logGo.transform.SetParent(root.transform);
            var log = logGo.AddComponent<DataLogger>();
            log.config = cfg;
            log.balloon = balloon;
            log.field = field;
            log.autopilot = ap;
            log.game = game;
            log.world = world;
            log.waypoint = waypointGo.transform;
            log.observationBuffer = obs;

            ui.logger = log;
        }
    }
}
