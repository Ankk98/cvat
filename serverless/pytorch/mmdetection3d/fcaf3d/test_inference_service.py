
import requests
import numpy as np
import base64
import json
import time

def create_dummy_pcd():
    # properties of a simple float32 point cloud
    num_points = 1000
    # x, y, z, r, g, b (6 channels required by FCAF3D config)
    points = np.random.rand(num_points, 6).astype(np.float32)
    return points.tobytes()

def test_inference():
    url = "http://localhost:8001/detect"
    print(f"Testing inference at {url}...")

    # Create dummy data
    pcd_bytes = create_dummy_pcd()
    pcd_b64 = base64.b64encode(pcd_bytes).decode('utf-8')

    payload = {
        "image": pcd_b64,
        "frame_number": 999,
        "threshold": 0.1
    }

    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=30)
        end_time = time.time()

        print(f"Status Code: {response.status_code}")
        print(f"Time Taken: {end_time - start_time:.2f}s")

        if response.status_code == 200:
            print("Response JSON:")
            print(json.dumps(response.json(), indent=2))
        else:
            print("Error Response:")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    # Wait a bit to ensure service might be up if run immediately after deploy
    test_inference()
