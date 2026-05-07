using UnityEngine;

namespace BalloonSim.Sim
{
    public class SimpleCameraRig : MonoBehaviour
    {
        public Transform follow;
        public float distance = 12f;
        public float minDistance = 4f;
        public float maxDistance = 40f;

        public bool firstPerson;
        public float yaw;
        public float pitch;

        [Header("Sensitivity")]
        public float mouseYaw = 0.35f;
        public float mousePitch = 0.35f;
        public float wheelZoom = 2.0f;

        private void Start()
        {
            ResetView();
        }

        public void ResetView()
        {
            yaw = 25f;
            pitch = 18f;
            distance = Mathf.Clamp(distance, minDistance, maxDistance);
        }

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.Alpha1))
                firstPerson = !firstPerson;

            if (Input.GetKeyDown(KeyCode.Space))
                ResetView();

            bool minimapActive = MinimapCubeRenderer.IsHoveringMinimap;

            if (!minimapActive)
            {
                float mx = Input.GetAxis("Mouse X");
                float my = Input.GetAxis("Mouse Y");

                bool dragging = firstPerson || Input.GetMouseButton(0);
                if (dragging)
                {
                    yaw += mx * mouseYaw;
                    pitch -= my * mousePitch;
                    pitch = Mathf.Clamp(pitch, -80f, 80f);
                }

                float wheel = Input.mouseScrollDelta.y;
                if (Mathf.Abs(wheel) > 1e-3f)
                    distance = Mathf.Clamp(distance - wheel * wheelZoom, minDistance, maxDistance);
            }

            Apply();
        }

        private void Apply()
        {
            Vector3 targetPos = follow != null ? follow.position : Vector3.zero;

            Quaternion rot = Quaternion.Euler(pitch, yaw, 0f);

            if (firstPerson)
            {
                transform.position = targetPos;
                transform.rotation = rot;
            }
            else
            {
                Vector3 back = rot * Vector3.back;
                transform.position = targetPos + back * distance;
                transform.rotation = rot;
            }
        }
    }
}
