import json
import math

import cv2
import numpy as np
import pygame
import sys
import mediapipe as mp
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import logging
import torch
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.DEBUG)


class TableTennisGame:
    def __init__(self):
        # Initialize YOLO model for ball tracking
        model_path = "C:/workspace/projects/pingpong-foul/model/best-yolo11-transfer.pt"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path, verbose=False).to(self.device)
        print("Using GPU:", self.model.device)

        # Initialize MediaPipe Pose for skeleton detection
        self.mp_pose = mp.solutions.pose
        self.pose_camera1 = self.mp_pose.Pose()
        self.pose_camera2 = self.mp_pose.Pose()

        # Load calibration data and key points for 3D triangulation
        self.load_calibration_data()
        self.load_key_points()

        # Paths for video sources
        self.video_paths = {
            'camera1': 'C:\\workspace\\datasets\\foul-video\\c1.mp4',
            'camera2': 'C:\\workspace\\datasets\\foul-video\\c2.mp4'
        }
        self.caps = {
            "camera1": cv2.VideoCapture(self.video_paths['camera1']),
            "camera2": cv2.VideoCapture(self.video_paths['camera2'])
        }

        # Store 3D ball trajectory and separate 2D trajectories for each camera
        self.ball_trajectory_3d = []
        self.trajectory_2d_camera1 = []
        self.trajectory_2d_camera2 = []

        # Initialize plot figure for trajectory visualization
        self.fig = plt.figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

    def load_calibration_data(self):
        with open('0012-calibration_in_ex_trinsic.json', 'r') as f:
            calibration_data = json.load(f)
        self.camera1_intrinsics = np.array(calibration_data['camera1']['intrinsics']['camera_matrix'])
        self.camera1_rot_vec = np.array(calibration_data['camera1']['extrinsics']['rotation_vector'])
        self.camera1_trans_vec = np.array(calibration_data['camera1']['extrinsics']['translation_vector'])
        self.camera1_rot_matrix, _ = cv2.Rodrigues(self.camera1_rot_vec)
        self.proj_matrix1 = np.dot(self.camera1_intrinsics,
                                   np.hstack((self.camera1_rot_matrix, self.camera1_trans_vec)))

        self.camera2_intrinsics = np.array(calibration_data['camera2']['intrinsics']['camera_matrix'])
        self.camera2_rot_vec = np.array(calibration_data['camera2']['extrinsics']['rotation_vector'])
        self.camera2_trans_vec = np.array(calibration_data['camera2']['extrinsics']['translation_vector'])
        self.camera2_rot_matrix, _ = cv2.Rodrigues(self.camera2_rot_vec)
        self.proj_matrix2 = np.dot(self.camera2_intrinsics,
                                   np.hstack((self.camera2_rot_matrix, self.camera2_trans_vec)))
        logging.debug("Calibration data loaded successfully.")

    def load_key_points(self):
        with open('0010-calibration_key_points.json', 'r') as f:
            key_points_data = json.load(f)
        self.camera1_points = key_points_data["camera1_points"]
        self.camera2_points = key_points_data["camera2_points"]
        self.camera1_3d_coordinates = key_points_data["3d_coordinates"][:8]  # First 8 for camera1
        self.camera2_3d_coordinates = key_points_data["3d_coordinates"][8:]  # Next 8 for camera2
        logging.debug("2D key points and corresponding 3D coordinates loaded successfully.")

    def calculate_3d_skeleton(self, landmarks1, landmarks2, video_width, video_height):
        skeleton_3d = []
        for lm1, lm2 in zip(landmarks1, landmarks2):
            point1 = np.array([[lm1.x * video_width], [lm1.y * video_height]], dtype=np.float32)
            point2 = np.array([[lm2.x * video_width], [lm2.y * video_height]], dtype=np.float32)
            skeleton_3d.append(self.calculate_3d_coordinates(point1, point2))
        return skeleton_3d

    def draw_skeleton_3d(self, skeleton_3d):
        # Draw lines connecting the skeleton parts based on their MediaPipe landmark connections
        connections = [
            (0, 1), (1, 2), (2, 3),  # Right eye to right ear
            (0, 4), (4, 5), (5, 6),  # Left eye to left ear
            (0, 9), (9, 10),  # Nose to mouth
            (10, 8), (9, 7),  # Mouth to ears
            (11, 12), (11, 13), (13, 15), (15, 17),  # Left arm
            (12, 14), (14, 16), (16, 18),  # Right arm
            (11, 23), (12, 24), (23, 24),  # Body and hips
            # (23, 25), (24, 26),  # Upper legs
            # (25, 27), (26, 28),  # Lower legs
            # (27, 29), (28, 30),  # Ankles to feet
            # (29, 31), (30, 32)  # Feet
        ]
        for start, end in connections:
            xs, ys, zs = zip(skeleton_3d[start], skeleton_3d[end])
            self.ax.plot(xs, ys, zs, color='green')  # Set skeleton color to green

    def draw_key_points(self, frame, points, coordinates):
        for (x, y), (x3d, y3d, z3d) in zip(points, coordinates):
            cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
            text = f"({x3d:.2f}, {y3d:.2f}, {z3d:.2f})"
            #cv2.putText(frame, text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def calculate_3d_coordinates(self, point1, point2):
        points_4d_homogeneous = cv2.triangulatePoints(self.proj_matrix1, self.proj_matrix2, point1, point2)
        points_3d = points_4d_homogeneous[:3] / points_4d_homogeneous[3]
        return points_3d.flatten()

    def track_ball(self, frame, camera_key, trajectory_2d):
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")
        if results[0].boxes:
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            trajectory_2d.append((cx, cy))
            if len(trajectory_2d) > 100:
                trajectory_2d.pop(0)
            return np.array([[cx], [cy]], dtype=np.float32)
        return None

    def process_frame_for_skeleton(self, frame, pose, camera_key):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None

    def filter_noise_by_acceleration(self,trajectory_2d, accel_threshold=5):
        if len(trajectory_2d) < 3:
            return trajectory_2d  # 如果点数不足，直接返回

        filtered_trajectory = [trajectory_2d[0]]  # 保留第一个点
        for i in range(1, len(trajectory_2d) - 1):
            v1 = np.array(trajectory_2d[i]) - np.array(trajectory_2d[i - 1])
            v2 = np.array(trajectory_2d[i + 1]) - np.array(trajectory_2d[i])
            accel = np.linalg.norm(v2 - v1)
            if accel < accel_threshold:  # 如果加速度在阈值内，保留该点
                filtered_trajectory.append(trajectory_2d[i])

        return filtered_trajectory

    def draw_trajectory(self, frame, trajectory_2d, max_distance=20):

        #trajectory_2d = self.filter_noise_by_acceleration(trajectory_2d)
        if len(trajectory_2d) < 2:
            return  # Not enough points to draw

        for i in range(1, len(trajectory_2d)):
            cv2.line(frame, trajectory_2d[i - 1], trajectory_2d[i], (255, 0, 0), 2)
            cv2.circle(frame, trajectory_2d[i], 3, (0, 255, 255), -1)

    def draw_3d_net(self):
        # Coordinates for the table net
        net_points = [
            (-0.03, 1.52, 0.185),  # Top left of the net
            (-0.03, 1.52, 0),  # Bottom left of the net
            (1.55, 1.52, 0.185),  # Top right of the net
            (1.55, 1.52, 0)  # Bottom right of the net
        ]

        # Plot the table net
        self.ax.plot([net_points[0][0], net_points[1][0]], [net_points[0][1], net_points[1][1]],
                     [net_points[0][2], net_points[1][2]], 'r-')  # Left vertical line

        self.ax.plot([net_points[2][0], net_points[3][0]], [net_points[2][1], net_points[3][1]],
                     [net_points[2][2], net_points[3][2]], 'r-')  # Right vertical line

        self.ax.plot([net_points[0][0], net_points[2][0]], [net_points[0][1], net_points[2][1]],
                     [net_points[0][2], net_points[2][2]], 'r-')  # Top horizontal line

    def draw_3d_cube(self):
        # Define the 8 corner points of the cube
        points = np.array([
            (0, 0, 0), (1.52, 0, 0), (1.52, -1.52, 0), (0, -1.52, 0),  # Bottom surface
            (0, 0, 1), (1.52, 0, 1), (1.52, -1.52, 1), (0, -1.52, 1)  # Top surface
        ])

        # Define the edges connecting the points to form the cube
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom surface edges
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top surface edges
            (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical edges
        ]

        # Draw the edges of the cube
        for start, end in edges:
            self.ax.plot(
                [points[start][0], points[end][0]],
                [points[start][1], points[end][1]],
                [points[start][2], points[end][2]],
                'g:'
            )

    def update_plot_surface(self, skeleton_3d):
        self.ax.cla()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # Set axis limits to make the x-axis opposite and match y-axis scale
        self.ax.set_xlim(1.72, -0.2)  # Reverse x-axis
        self.ax.set_ylim(-1.52, 1.52)
        self.ax.set_zlim(-0.2, 1.52)

        # Ensure the aspect ratio is equal for x, y, and z
        self.ax.set_box_aspect([1.92, 3.04, 1.72])  # Aspect ratio equalized

        self.draw_3d_cube()
        self.draw_3d_net()

        # Draw the table tennis table
        table_corners = [
            (0, 0, 0),
            (1.52, 0, 0),
            (1.52, 1.52, 0),
            (0, 1.52, 0),
            (0, 0, 0)  # Close the loop to complete the rectangle
        ]
        x_table, y_table, z_table = zip(*table_corners)
        self.ax.plot(x_table, y_table, z_table, color="red", linewidth=2, label="Table")


        # Plot skeleton structure if available
        if skeleton_3d:
            self.draw_skeleton_3d(skeleton_3d)

        # Plot the ball trajectory with color based on speed
        if self.ball_trajectory_3d:
            xs, ys, zs = zip(*self.ball_trajectory_3d)
            num_points = len(self.ball_trajectory_3d)

            # Calculate and plot trajectory lines with varying color based on speed
            for i in range(1, num_points):
                speed = np.linalg.norm(np.array(self.ball_trajectory_3d[i]) - np.array(self.ball_trajectory_3d[i - 1]))
                color = (
                min(1.0, speed / 5), 0, 1 - min(1.0, speed / 5))  # Color transitions from blue to red based on speed
                self.ax.plot([xs[i - 1], xs[i]], [ys[i - 1], ys[i]], [zs[i - 1], zs[i]], color=color)

            # Plot individual points with transparency for a trailing effect
            for i in range(num_points):
                alpha = (i + 1) / num_points
                self.ax.scatter(xs[i], ys[i], zs[i], color=(1 - alpha, 0, alpha), s=10, alpha=alpha)

        # Set a fixed viewing angle for better depth perception
        self.ax.view_init(elev=20, azim=135)

        # Convert the Matplotlib plot to a Pygame surface
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        plot_surface = pygame.image.fromstring(canvas.tostring_rgb(), canvas.get_width_height(), "RGB")
        logging.debug("3D plot surface updated.")
        return plot_surface

    def read_frame(self, camera):
        ret, frame = self.caps[camera].read()
        if not ret:
            self.caps[camera].set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.caps[camera].read()
        return frame

    def reset_trajectories_if_needed(self, last_point, current_point):
        if last_point[1] > 0.25 and current_point[1] < 0.15:
            self.reset_trajectories()

    def reset_trajectories(self):
        self.ball_trajectory_3d.clear()
        self.trajectory_2d_camera1.clear()
        self.trajectory_2d_camera2.clear()


def main():
    pygame.init()
    # Set screen width to twice the original width and height
    screen_width = 2400  # 2 * 1200
    screen_height = 800  # 2 * 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Table Tennis 3D Ball Trajectory - Enlarged Interface")
    game = TableTennisGame()
    running = True
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = not paused

        if not paused:
            frame1 = game.read_frame("camera1")
            frame2 = game.read_frame("camera2")
            game.video_width, game.video_height = frame1.shape[1], frame1.shape[0]
            landmarks1 = game.process_frame_for_skeleton(frame1, game.pose_camera1, "camera1")
            landmarks2 = game.process_frame_for_skeleton(frame2, game.pose_camera2, "camera2")
            skeleton_3d = game.calculate_3d_skeleton(landmarks1, landmarks2, game.video_width,
                                                     game.video_height) if landmarks1 and landmarks2 else None
            plot_surface = game.update_plot_surface(skeleton_3d)

            ball_point1 = game.track_ball(frame1, "camera1", game.trajectory_2d_camera1)
            ball_point2 = game.track_ball(frame2, "camera2", game.trajectory_2d_camera2)

            if ball_point1 is not None:
                current_x = ball_point1[0][0]  # x-coordinate of the current point

                # Check if there are at least two previous points to calculate x1 and x2
                if len(game.trajectory_2d_camera1) >= 2:
                    last_x = game.trajectory_2d_camera1[-1][0]  # x-coordinate of the last point
                    second_last_x = game.trajectory_2d_camera1[-2][0]  # x-coordinate of the second-last point

                    # Calculate distances
                    x1 = abs(last_x - second_last_x)  # Distance between the last two points
                    x2 = abs(current_x - last_x)  # Distance from the last point to the current point

                    # Reset trajectory if x2 is more than three times x1
                    if x2 > 3 * x1:
                        game.reset_trajectories()

            if ball_point1 is not None and ball_point2 is not None:
                ball_3d = game.calculate_3d_coordinates(ball_point1, ball_point2)
                if game.ball_trajectory_3d:
                    game.reset_trajectories_if_needed(game.ball_trajectory_3d[-1], ball_3d)
                game.ball_trajectory_3d.append(tuple(ball_3d))
                if len(game.ball_trajectory_3d) > 100:
                    game.ball_trajectory_3d.pop(0)

            game.draw_trajectory(frame1, game.trajectory_2d_camera1)
            game.draw_trajectory(frame2, game.trajectory_2d_camera2)
            game.draw_key_points(frame1, game.camera1_points, game.camera1_3d_coordinates)
            game.draw_key_points(frame2, game.camera2_points, game.camera2_3d_coordinates)

            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            frame1_surface = pygame.surfarray.make_surface(cv2.resize(frame1_rgb, (800, 800)).swapaxes(0, 1))
            frame2_surface = pygame.surfarray.make_surface(cv2.resize(frame2_rgb, (800, 800)).swapaxes(0, 1))

            # Blit the frames and 3D plot to the screen in a single row
            screen.blit(frame1_surface, (0, 0))
            screen.blit(frame2_surface, (800, 0))
            plot_surface_resized = pygame.transform.scale(plot_surface, (800, 800))
            screen.blit(plot_surface_resized, (1600, 0))

        pygame.display.flip()
        pygame.time.delay(30)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
