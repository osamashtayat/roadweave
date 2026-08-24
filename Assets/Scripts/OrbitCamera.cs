using UnityEngine;

public class OrbitCamera : MonoBehaviour
{
    [Tooltip("Legacy free-camera controller. Keep disabled while VehicleFollowCamera is active.")]
    public float moveSpeed = 5f;
    public float rotateSpeed = 80f;

    private void Update()
    {
        Vector3 movement = Vector3.zero;

        if (Input.GetKey(KeyCode.W))
            movement += transform.forward;

        if (Input.GetKey(KeyCode.S))
            movement -= transform.forward;

        if (Input.GetKey(KeyCode.A))
            movement -= transform.right;

        if (Input.GetKey(KeyCode.D))
            movement += transform.right;

        if (Input.GetKey(KeyCode.E))
            movement += Vector3.up;          // Up

        if (Input.GetKey(KeyCode.Q))
            movement += Vector3.down;        // Down

        transform.position += movement.normalized * moveSpeed * Time.deltaTime;

        // Rotation
        if (Input.GetKey(KeyCode.LeftArrow))
            transform.Rotate(0, -rotateSpeed * Time.deltaTime, 0);

        if (Input.GetKey(KeyCode.RightArrow))
            transform.Rotate(0, rotateSpeed * Time.deltaTime, 0);
    }
}
