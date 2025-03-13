import json
import numpy as np
import cv2
from pyswarm import pso


class CameraTuner:
    def __init__(self, initial_cam1_params, initial_cam2_params, points_2d_cam1, points_2d_cam2, points_3d):
        self.initial_cam1_params = initial_cam1_params
        self.initial_cam2_params = initial_cam2_params
        self.points_2d_cam1 = points_2d_cam1
        self.points_2d_cam2 = points_2d_cam2
        self.points_3d = points_3d
        self.iteration = 0  # Track the iteration number
        self.optimized_cam1_rot_vec = None
        self.optimized_cam1_trans_vec = None
        self.optimized_cam2_rot_vec = None
        self.optimized_cam2_trans_vec = None

    def objective_function(self, params):
        # Split parameters for camera 1 and camera 2
        cam1_params = params[:9]  # 4 intrinsic parameters and 5 distortion coefficients
        cam2_params = params[9:]

        self.iteration += 1
        if self.iteration % 100 == 0:  # Print progress every 100 iterations
            print(f"PSO Iteration: {self.iteration}")

        cam1_intrinsic, cam1_dist_coeffs = self.unpack_intrinsics(cam1_params)
        cam2_intrinsic, cam2_dist_coeffs = self.unpack_intrinsics(cam2_params)

        # Calculate rvec and tvec using solvePnP
        success_cam1, cam1_rot_vec, cam1_trans_vec = cv2.solvePnP(
            self.points_3d[:8], self.points_2d_cam1, cam1_intrinsic, cam1_dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        success_cam2, cam2_rot_vec, cam2_trans_vec = cv2.solvePnP(
            self.points_3d[8:], self.points_2d_cam2, cam2_intrinsic, cam2_dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not (success_cam1 and success_cam2):
            # If solvePnP fails, return a large error to indicate an invalid solution
            print("solvePnP failed for one or both cameras.")
            return 1e6

        # Store the optimized rotation and translation vectors for saving later
        self.optimized_cam1_rot_vec = cam1_rot_vec
        self.optimized_cam1_trans_vec = cam1_trans_vec
        self.optimized_cam2_rot_vec = cam2_rot_vec
        self.optimized_cam2_trans_vec = cam2_trans_vec

        # Project 3D points onto 2D using calculated rvec and tvec
        projected_2d_cam1 = self.project_points(cam1_intrinsic, cam1_dist_coeffs, cam1_rot_vec, cam1_trans_vec, self.points_3d[:8])
        projected_2d_cam2 = self.project_points(cam2_intrinsic, cam2_dist_coeffs, cam2_rot_vec, cam2_trans_vec, self.points_3d[8:])

        # Calculate reprojection error
        error_cam1 = np.linalg.norm(self.points_2d_cam1 - projected_2d_cam1, axis=1).mean()
        error_cam2 = np.linalg.norm(self.points_2d_cam2 - projected_2d_cam2, axis=1).mean()

        return error_cam1 + error_cam2

    def unpack_intrinsics(self, params):
        intrinsic = np.array([[params[0], 0, params[1]],  # f_x and c_x
                              [0, params[2], params[3]],  # f_y and c_y
                              [0, 0, 1]])                 # Fixed elements
        dist_coeffs = np.array(params[4:9])  # 5 distortion coefficients
        return intrinsic, dist_coeffs

    def project_points(self, intrinsic, dist_coeffs, rot_vec, trans_vec, points_3d):
        projected_2d, _ = cv2.projectPoints(points_3d, rot_vec, trans_vec, intrinsic, dist_coeffs)
        return projected_2d.reshape(-1, 2)


def load_calibration_data():
    with open('0012-calibration_in_ex_trinsic.json', 'r') as f:
        calibration_data = json.load(f)

    # Load only camera_matrix and dist_coeffs
    camera1_intrinsic = np.array(calibration_data['camera1']['intrinsics']['camera_matrix']).flatten()
    camera1_dist_coeffs = np.array(calibration_data['camera1']['intrinsics']['dist_coeffs']).flatten()
    initial_cam1_params = np.hstack([camera1_intrinsic[[0, 2, 4, 5]], camera1_dist_coeffs])

    camera2_intrinsic = np.array(calibration_data['camera2']['intrinsics']['camera_matrix']).flatten()
    camera2_dist_coeffs = np.array(calibration_data['camera2']['intrinsics']['dist_coeffs']).flatten()
    initial_cam2_params = np.hstack([camera2_intrinsic[[0, 2, 4, 5]], camera2_dist_coeffs])

    return initial_cam1_params, initial_cam2_params


def load_key_points():
    with open('0010-calibration_key_points.json', 'r') as f:
        key_points_data = json.load(f)

    points_2d_cam1 = np.array(key_points_data["camera1_points"], dtype=np.float32)
    points_2d_cam2 = np.array(key_points_data["camera2_points"], dtype=np.float32)
    points_3d = np.array(key_points_data["3d_coordinates"][:8] + key_points_data["3d_coordinates"][8:], dtype=np.float32)

    if len(points_2d_cam1) < 4 or len(points_2d_cam2) < 4:
        raise ValueError("Each camera must have at least 4 corresponding 2D points for solvePnP.")

    if len(points_3d) < 4:
        raise ValueError("At least 4 corresponding 3D points are required for solvePnP.")

    return points_2d_cam1, points_2d_cam2, points_3d


def save_optimized_parameters(cam1_params, cam2_params, tuner, filename="0012-calibration_in_ex_trinsic_new.json"):
    def format_intrinsics(params):
        return [
            [params[0], 0, params[1]],
            [0, params[2], params[3]],
            [0, 0, 1]
        ]

    def format_extrinsics(vec):
        return [[float(v)] for v in vec.flatten()]  # Ensure format matches JSON structure

    optimized_data = {
        "camera1": {
            "intrinsics": {
                "camera_matrix": format_intrinsics(cam1_params[:4]),
                "dist_coeffs": [cam1_params[4:9].tolist()]
            },
            "extrinsics": {
                "rotation_vector": format_extrinsics(tuner.optimized_cam1_rot_vec),
                "translation_vector": format_extrinsics(tuner.optimized_cam1_trans_vec)
            }
        },
        "camera2": {
            "intrinsics": {
                "camera_matrix": format_intrinsics(cam2_params[:4]),
                "dist_coeffs": [cam2_params[4:9].tolist()]
            },
            "extrinsics": {
                "rotation_vector": format_extrinsics(tuner.optimized_cam2_rot_vec),
                "translation_vector": format_extrinsics(tuner.optimized_cam2_trans_vec)
            }
        }
    }

    with open(filename, 'w') as f:
        json.dump(optimized_data, f, indent=4)
    print(f"Optimized parameters saved to {filename}")


def main():
    initial_cam1_params, initial_cam2_params = load_calibration_data()
    points_2d_cam1, points_2d_cam2, points_3d = load_key_points()

    initial_params = np.hstack([initial_cam1_params, initial_cam2_params])

    lower_bounds = [1170, 850, 1100, 500, -1, -1, -1, -1, -1] * 2
    upper_bounds = [1190, 1000, 1250, 600, 1, 1, 1, 1, 1] * 2

    camera_tuner = CameraTuner(initial_cam1_params, initial_cam2_params, points_2d_cam1, points_2d_cam2, points_3d)

    best_params, _ = pso(camera_tuner.objective_function, lower_bounds, upper_bounds, maxiter=200000, swarmsize=5000)

    optimized_cam1_params = best_params[:9]
    optimized_cam2_params = best_params[9:]

    print("Optimized Camera 1 Parameters:", optimized_cam1_params)
    print("Optimized Camera 2 Parameters:", optimized_cam2_params)

    save_optimized_parameters(optimized_cam1_params, optimized_cam2_params, camera_tuner)


if __name__ == "__main__":
    main()
