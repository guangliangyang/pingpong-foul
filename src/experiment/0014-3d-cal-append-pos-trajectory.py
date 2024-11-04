import json
import cv2
import numpy as np
import pygame
import sys
import mediapipe as mp
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)


class TableTennisGame:
    def __init__(self):
        # Initialize separate MediaPipe Pose objects for each camera
        self.mp_pose = mp.solutions.pose
        self.pose_camera1 = self.mp_pose.Pose()
        self.pose_camera2 = self.mp_pose.Pose()

        # Initialize and load calibration data
        self.load_calibration_data()

        # Paths for dedicated video sources
        self.video_paths = {
            'camera1': 'C:\\workspace\\datasets\\foul-video\\c1.mp4',
            'camera2': 'C:\\workspace\\datasets\\foul-video\\c2.mp4'
        }

        # Initialize video captures for both cameras
        self.caps = {
            "camera1": cv2.VideoCapture(self.video_paths['camera1']),
            "camera2": cv2.VideoCapture(self.video_paths['camera2'])
        }

        # Dictionary to store the current 3D coordinates of the upper body
        self.current_pose_3d = {}

        # Initialize plot figure for trajectory visualization
        self.fig = plt.figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

    def load_calibration_data(self):
        # Load camera calibration data from JSON
        with open('0012-calibration_in_ex_trinsic.json', 'r') as f:
            calibration_data = json.load(f)

        # Extract parameters for camera1
        self.camera1_intrinsics = np.array(calibration_data['camera1']['intrinsics']['camera_matrix'])
        self.camera1_dist_coeffs = np.array(calibration_data['camera1']['intrinsics']['dist_coeffs'])
        self.camera1_rot_vec = np.array(calibration_data['camera1']['extrinsics']['rotation_vector'])
        self.camera1_trans_vec = np.array(calibration_data['camera1']['extrinsics']['translation_vector'])

        # Extract parameters for camera2
        self.camera2_intrinsics = np.array(calibration_data['camera2']['intrinsics']['camera_matrix'])
        self.camera2_dist_coeffs = np.array(calibration_data['camera2']['intrinsics']['dist_coeffs'])
        self.camera2_rot_vec = np.array(calibration_data['camera2']['extrinsics']['rotation_vector'])
        self.camera2_trans_vec = np.array(calibration_data['camera2']['extrinsics']['translation_vector'])

        # Convert rotation vectors to rotation matrices
        self.camera1_rot_matrix, _ = cv2.Rodrigues(self.camera1_rot_vec)
        self.camera2_rot_matrix, _ = cv2.Rodrigues(self.camera2_rot_vec)

        # Define projection matrices for both cameras
        self.proj_matrix1 = np.dot(self.camera1_intrinsics,
                                   np.hstack((self.camera1_rot_matrix, self.camera1_trans_vec)))
        self.proj_matrix2 = np.dot(self.camera2_intrinsics,
                                   np.hstack((self.camera2_rot_matrix, self.camera2_trans_vec)))

        logging.debug("Calibration data loaded successfully.")

    def calculate_3d_coordinates(self, point1, point2):
        # Calculate 3D coordinates using triangulation
        points_4d_homogeneous = cv2.triangulatePoints(self.proj_matrix1, self.proj_matrix2, point1, point2)
        points_3d = points_4d_homogeneous[:3] / points_4d_homogeneous[3]
        points_3d[0] = -points_3d[0]  # Invert x-axis to correct left-right orientation
        return points_3d.flatten()

    def process_frame(self, frame, camera_key):
        if camera_key == "camera1":
            pose = self.pose_camera1
        elif camera_key == "camera2":
            pose = self.pose_camera2
        else:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Capture upper body landmarks if detected
        if results.pose_landmarks:
            upper_body_points = {}
            landmarks = {
                "right_wrist": self.mp_pose.PoseLandmark.RIGHT_WRIST,
                "right_elbow": self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                "right_shoulder": self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                "left_shoulder": self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                "left_elbow": self.mp_pose.PoseLandmark.LEFT_ELBOW,
                "left_wrist": self.mp_pose.PoseLandmark.LEFT_WRIST,
                "left_hip": self.mp_pose.PoseLandmark.LEFT_HIP,
                "right_hip": self.mp_pose.PoseLandmark.RIGHT_HIP
            }

            for name, landmark in landmarks.items():
                body_part = results.pose_landmarks.landmark[landmark]
                upper_body_points[name] = np.array([[int(body_part.x * frame.shape[1])],
                                                    [int(body_part.y * frame.shape[0])]], dtype=np.float32)
            return upper_body_points
        return None

    def update_plot_surface(self):
        self.ax.cla()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)

        # Define body connections for drawing lines between landmarks
        connections = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip")
        ]

        # Plot the current 3D pose by drawing lines between connected points
        for part1, part2 in connections:
            if part1 in self.current_pose_3d and part2 in self.current_pose_3d:
                x_vals = [self.current_pose_3d[part1][0], self.current_pose_3d[part2][0]]
                y_vals = [self.current_pose_3d[part1][1], self.current_pose_3d[part2][1]]
                z_vals = [self.current_pose_3d[part1][2], self.current_pose_3d[part2][2]]
                self.ax.plot(x_vals, y_vals, z_vals, color="blue")

        self.ax.view_init(elev=20, azim=135)
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        plot_surface = pygame.image.fromstring(canvas.tostring_rgb(), canvas.get_width_height(), "RGB")
        logging.debug("3D plot surface updated with the current pose.")
        return plot_surface

    def read_frame(self, camera):
        ret, frame = self.caps[camera].read()
        if not ret:
            logging.warning(f"End of video for {camera}. Restarting video.")
            self.caps[camera].set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.caps[camera].read()
        return frame


def main():
    pygame.init()
    screen_width, screen_height = 1200, 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Table Tennis 3D Pose")

    game = TableTennisGame()

    running = True
    while running:
        screen.fill((255, 255, 255))

        frame1 = game.read_frame("camera1")
        frame2 = game.read_frame("camera2")

        upper_body_points_cam1 = game.process_frame(frame1, "camera1")
        upper_body_points_cam2 = game.process_frame(frame2, "camera2")

        # Calculate the current 3D pose only (no trajectory)
        if upper_body_points_cam1 and upper_body_points_cam2:
            for part in upper_body_points_cam1:
                point1 = upper_body_points_cam1[part]
                point2 = upper_body_points_cam2[part]
                game.current_pose_3d[part] = game.calculate_3d_coordinates(point1, point2)

        plot_surface = game.update_plot_surface()

        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        frame1_surface = pygame.surfarray.make_surface(cv2.resize(frame1_rgb, (400, 400)).swapaxes(0, 1))
        frame2_surface = pygame.surfarray.make_surface(cv2.resize(frame2_rgb, (400, 400)).swapaxes(0, 1))

        screen.blit(frame1_surface, (0, 0))
        screen.blit(frame2_surface, (400, 0))
        screen.blit(plot_surface, (800, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        pygame.display.flip()
        pygame.time.delay(30)

    game.pose_camera1.close()
    game.pose_camera2.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
