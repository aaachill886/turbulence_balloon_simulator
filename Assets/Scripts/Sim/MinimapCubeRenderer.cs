using UnityEngine;

namespace BalloonSim.Sim
{
    public class MinimapCubeRenderer : MonoBehaviour
    {
        public SimulationConfig config;
        public BlockWorld world;
        public Transform balloon;
        public Transform waypoint;

        [Header("Minimap")]
        public bool showMinimap = true;
        public bool expandedMode = false;

        [Header("Viewport")]
        public Rect smallViewport = new Rect(0.78f, 0.02f, 0.20f, 0.20f);
        public Rect expandedViewport = new Rect(0.25f, 0.15f, 0.50f, 0.70f);

        [Header("Camera")]
        public float yaw = 45f;
        public float pitch = 35f;
        public float distance = 40f;
        public float rotateSpeed = 2.5f;
        public float zoomSpeed = 2.0f;

        private Camera _cam;
        private Material _cubeMat;
        private Mesh _cubeMesh;
        private bool _drag;

        public static bool IsHoveringMinimap { get; private set; }

        private void Awake()
        {
            _cam = gameObject.GetComponent<Camera>();
            if (_cam == null) _cam = gameObject.AddComponent<Camera>();

            _cam.orthographic = false;
            _cam.depth = 10;
            _cam.clearFlags = CameraClearFlags.SolidColor;
            _cam.backgroundColor = new Color(0.03f, 0.05f, 0.12f, 1f);
            _cam.cullingMask = 0;

            Shader shader = Shader.Find("BalloonSim/MinimapCube");
            if (shader == null || !shader.isSupported)
            {
                Debug.LogWarning("[MinimapCubeRenderer] Custom shader not found, using Unlit/Color");
                shader = Shader.Find("Unlit/Color");
            }
            _cubeMat = new Material(shader);
            _cubeMesh = CreateCubeMesh();
        }

        private Mesh CreateCubeMesh()
        {
            Mesh m = new Mesh();
            m.vertices = new Vector3[]
            {
                new Vector3(-0.5f, -0.5f, -0.5f), new Vector3( 0.5f, -0.5f, -0.5f),
                new Vector3( 0.5f,  0.5f, -0.5f), new Vector3(-0.5f,  0.5f, -0.5f),
                new Vector3(-0.5f, -0.5f,  0.5f), new Vector3( 0.5f, -0.5f,  0.5f),
                new Vector3( 0.5f,  0.5f,  0.5f), new Vector3(-0.5f,  0.5f,  0.5f),
            };
            m.triangles = new int[]
            {
                0,2,1, 0,3,2, 1,2,6, 1,6,5, 4,5,6, 4,6,7,
                4,7,3, 4,3,0, 3,7,6, 3,6,2, 4,0,1, 4,1,5
            };
            m.RecalculateNormals();
            return m;
        }

        public void ResetView()
        {
            yaw = 45f;
            pitch = 35f;
            distance = 40f;
        }

        public void SyncWithMainCamera()
        {
            Camera mainCam = Camera.main;
            if (mainCam != null)
            {
                Vector3 euler = mainCam.transform.rotation.eulerAngles;
                yaw = euler.y;
                pitch = euler.x;
                if (pitch > 180f) pitch -= 360f;
            }
        }

        private void Update()
        {
            if (config == null) return;

            if (Input.GetKeyDown(KeyCode.M))
            {
                config.showMinimap = !config.showMinimap;
                if (!config.showMinimap)
                {
                    expandedMode = false;
                    SyncWithMainCamera();
                }
            }

            showMinimap = config.showMinimap;
            _cam.enabled = showMinimap;

            if (!showMinimap)
            {
                IsHoveringMinimap = false;
                return;
            }

            Rect vp = expandedMode ? expandedViewport : smallViewport;
            _cam.rect = vp;

            Vector2 mp = Input.mousePosition;
            Rect pr = _cam.pixelRect;
            bool hover = mp.x >= pr.xMin && mp.x <= pr.xMax && mp.y >= pr.yMin && mp.y <= pr.yMax;
            IsHoveringMinimap = hover;

            if (expandedMode && hover)
            {
                if (Input.GetMouseButtonDown(0)) _drag = true;
                if (Input.GetMouseButtonUp(0)) _drag = false;

                if (_drag)
                {
                    float mx = Input.GetAxis("Mouse X");
                    float my = Input.GetAxis("Mouse Y");
                    yaw += mx * rotateSpeed;
                    pitch -= my * rotateSpeed;
                    pitch = Mathf.Clamp(pitch, -80f, 80f);
                }

                float wheel = Input.mouseScrollDelta.y;
                if (Mathf.Abs(wheel) > 1e-3f)
                    distance = Mathf.Clamp(distance - wheel * zoomSpeed, 10f, 200f);
            }

            Apply();
        }

        private void Apply()
        {
            if (world == null || config == null) return;

            Vector3 center = GetUnlockedCenter();
            Quaternion rot = Quaternion.Euler(pitch, yaw, 0f);
            Vector3 back = rot * Vector3.back;
            transform.position = center + back * distance;
            transform.rotation = rot;
        }

        private Vector3 GetUnlockedCenter()
        {
            if (world == null || world.UnlockedBlocks.Count == 0) return Vector3.zero;

            float s = config != null ? config.boxSize : 16f;
            Vector3 sum = Vector3.zero;
            foreach (var b in world.UnlockedBlocks)
                sum += new Vector3(b.x * s + s * 0.5f, b.y * s + s * 0.5f, b.z * s + s * 0.5f);

            return sum / world.UnlockedBlocks.Count;
        }

        private void OnPostRender()
        {
            if (!showMinimap || world == null || config == null || _cubeMat == null || _cubeMesh == null) return;
            if (Camera.current != _cam) return;

            float s = config.boxSize;
            Int3 playerBlock = world.CurrentBlock;
            Int3 waypointBlock = waypoint != null ? world.GetBlockOf(waypoint.position) : new Int3(int.MaxValue, int.MaxValue, int.MaxValue);

            foreach (Int3 b in world.UnlockedBlocks)
            {
                Vector3 center = new Vector3(b.x * s + s * 0.5f, b.y * s + s * 0.5f, b.z * s + s * 0.5f);

                bool isPlayer = b.Equals(playerBlock);
                bool isWaypoint = b.Equals(waypointBlock);

                Color c;
                if (isPlayer && isWaypoint)
                {
                    c = new Color(0.3f, 0.85f, 0.4f, 0.65f);
                }
                else if (isPlayer)
                {
                    c = new Color(0.95f, 0.85f, 0.3f, 0.65f);
                }
                else if (isWaypoint)
                {
                    c = new Color(0.3f, 0.55f, 0.95f, 0.65f);
                }
                else
                {
                    c = new Color(0.25f, 0.45f, 0.75f, 0.20f);
                }

                _cubeMat.SetColor("_Color", c);
                _cubeMat.SetPass(0);
                Matrix4x4 mtx = Matrix4x4.TRS(center, Quaternion.identity, Vector3.one * s);
                Graphics.DrawMeshNow(_cubeMesh, mtx);
            }

            DrawMarkers();
        }

        private void DrawMarkers()
        {
            if (balloon != null)
            {
                _cubeMat.SetColor("_Color", Color.green);
                _cubeMat.SetPass(0);
                Matrix4x4 m = Matrix4x4.TRS(balloon.position, Quaternion.identity, Vector3.one * 0.6f);
                Graphics.DrawMeshNow(_cubeMesh, m);
            }

            if (waypoint != null)
            {
                _cubeMat.SetColor("_Color", new Color(1f, 0.85f, 0.35f, 1f));
                _cubeMat.SetPass(0);
                Matrix4x4 m = Matrix4x4.TRS(waypoint.position, Quaternion.identity, Vector3.one * 0.5f);
                Graphics.DrawMeshNow(_cubeMesh, m);
            }
        }

        private void OnGUI()
        {
            if (!showMinimap || config == null) return;

            Rect pr = _cam.pixelRect;

            if (!expandedMode)
            {
                GUI.Box(new Rect(pr.xMin - 4, Screen.height - pr.yMax - 18, pr.width + 8, pr.height + 24), "minimap (M hide)");

                if (GUI.Button(new Rect(pr.xMax - 24, Screen.height - pr.yMax - 18, 20, 20), "+"))
                {
                    expandedMode = true;
                    ResetView();
                }
            }
            else
            {
                GUI.Box(new Rect(pr.xMin - 4, Screen.height - pr.yMax - 18, pr.width + 8, pr.height + 24), "expanded map (drag rotate, wheel zoom, M hide)");

                if (GUI.Button(new Rect(pr.xMax - 24, Screen.height - pr.yMax - 18, 20, 20), "-"))
                {
                    expandedMode = false;
                }
            }
        }
    }
}
