import cv2
import numpy as np
import json

# Load intrinsic parameters from JSON file
with open("0010-calibration_instrinic.json", "r") as f:
    intrinsic_data = json.load(f)
camera1_intrinsics = intrinsic_data["camera1"]
camera2_intrinsics = intrinsic_data["camera2"]

# Extract camera matrices and distortion coefficients
camera1_matrix = np.array(camera1_intrinsics["camera_matrix"])
camera1_dist_coeffs = np.array(camera1_intrinsics["dist_coeffs"])

camera2_matrix = np.array(camera2_intrinsics["camera_matrix"])
camera2_dist_coeffs = np.array(camera2_intrinsics["dist_coeffs"])

# Load calibration points (2D and 3D) from JSON file
with open("0010-calibration_key_points.json", "r") as f:
    key_points_data = json.load(f)
camera1_points = np.array(key_points_data["camera1_points"], dtype=np.float32)
camera2_points = np.array(key_points_data["camera2_points"], dtype=np.float32)
object_points_camera1 = np.array(key_points_data["3d_coordinates"][:8], dtype=np.float32)  # First 8 for camera1
object_points_camera2 = np.array(key_points_data["3d_coordinates"][8:], dtype=np.float32)  # Last 8 for camera2

# Calculate extrinsics for camera1
success1, rvec1, tvec1 = cv2.solvePnP(object_points_camera1, camera1_points, camera1_matrix, camera1_dist_coeffs)
if success1:
    print(f"Camera1 Extrinsics:\nRotation Vector:\n{rvec1}\nTranslation Vector:\n{tvec1}")
else:
    print("Failed to calculate extrinsics for camera1.")

# Calculate extrinsics for camera2
success2, rvec2, tvec2 = cv2.solvePnP(object_points_camera2, camera2_points, camera2_matrix, camera2_dist_coeffs)
if success2:
    print(f"Camera2 Extrinsics:\nRotation Vector:\n{rvec2}\nTranslation Vector:\n{tvec2}")
else:
    print("Failed to calculate extrinsics for camera2.")

# Prepare data to save to JSON file
calibration_data = {
    "camera1": {
        "intrinsics": {
            "camera_matrix": camera1_matrix.tolist(),
            "dist_coeffs": camera1_dist_coeffs.tolist()
        },
        "extrinsics": {
            "rotation_vector": rvec1.tolist(),
            "translation_vector": tvec1.tolist()
        }
    },
    "camera2": {
        "intrinsics": {
            "camera_matrix": camera2_matrix.tolist(),
            "dist_coeffs": camera2_dist_coeffs.tolist()
        },
        "extrinsics": {
            "rotation_vector": rvec2.tolist(),
            "translation_vector": tvec2.tolist()
        }
    }
}

# Save to JSON file
with open("0012-calibration_in_ex_trinsic.json", "w") as f:
    json.dump(calibration_data, f, indent=4)

print("Calibration data (intrinsics and extrinsics) saved to 0012-calibration_in_ex_trinsic.json")
