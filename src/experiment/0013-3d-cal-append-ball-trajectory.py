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
        # Load camera1 and camera2 intrinsic and extrinsic parameters
        self.camera1_intrinsics = np.array(calibration_data['camera1']['intrinsics']['camera_matrix'])
        self.camera1_rot_vec = np.array(calibration_data['camera1']['extrinsics']['rotation_vector'])
        self.camera1_trans_vec = np.array(calibration_data['camera1']['extrinsics']['translation_vector'])
        self.camera1_rot_matrix, _ = cv2.Rodrigues(self.camera1_rot_vec)
        self.proj_matrix1 = np.dot(self.camera1_intrinsics, np.hstack((self.camera1_rot_matrix, self.camera1_trans_vec)))

        self.camera2_intrinsics = np.array(calibration_data['camera2']['intrinsics']['camera_matrix'])
        self.camera2_rot_vec = np.array(calibration_data['camera2']['extrinsics']['rotation_vector'])
        self.camera2_trans_vec = np.array(calibration_data['camera2']['extrinsics']['translation_vector'])
        self.camera2_rot_matrix, _ = cv2.Rodrigues(self.camera2_rot_vec)
        self.proj_matrix2 = np.dot(self.camera2_intrinsics, np.hstack((self.camera2_rot_matrix, self.camera2_trans_vec)))
        logging.debug("Calibration data loaded successfully.")

    def load_key_points(self):
        with open('0010-calibration_key_points.json', 'r') as f:
            key_points_data = json.load(f)
        self.camera1_points = key_points_data["camera1_points"]
        self.camera2_points = key_points_data["camera2_points"]
        logging.debug("2D key points loaded successfully.")

    def draw_key_points(self, frame, points):
        for (x, y) in points:
            cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

    def calculate_3d_coordinates(self, point1, point2):
        logging.debug(f"Calculating 3D coordinates from points: {point1}, {point2}")
        points_4d_homogeneous = cv2.triangulatePoints(self.proj_matrix1, self.proj_matrix2, point1, point2)
        points_3d = points_4d_homogeneous[:3] / points_4d_homogeneous[3]
        logging.debug(f"Calculated 3D coordinates: {points_3d.flatten()}")
        return points_3d.flatten()

    def project_3d_line_to_2d(self, frame, proj_matrix, rot_vec, trans_vec, intrinsics):
        # Define the 3D line endpoints
        start_3d = np.array([[0], [0], [0]], dtype=np.float32)
        end_3d = np.array([[1.52], [0], [0]], dtype=np.float32)

        # Project the 3D points to 2D
        start_2d = cv2.projectPoints(start_3d, rot_vec, trans_vec, intrinsics, None)[0][0][0]
        end_2d = cv2.projectPoints(end_3d, rot_vec, trans_vec, intrinsics, None)[0][0][0]

        # Draw the line on the frame
        cv2.line(frame, (int(start_2d[0]), int(start_2d[1])), (int(end_2d[0]), int(end_2d[1])), (0, 0, 255), 2)

    def track_ball(self, frame, camera_key, trajectory_2d):
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")
        if results[0].boxes:
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            logging.debug(f"{camera_key}: Detected ball at ({cx}, {cy})")
            trajectory_2d.append((cx, cy))

            # Limit 2D trajectory to the last 100 points
            if len(trajectory_2d) > 100:
                trajectory_2d.pop(0)

            return np.array([[cx], [cy]], dtype=np.float32)
        logging.debug(f"{camera_key}: No ball detected.")
        return None

    def draw_trajectory(self, frame, trajectory_2d):
        for i in range(1, len(trajectory_2d)):
            cv2.line(frame, trajectory_2d[i - 1], trajectory_2d[i], (255, 0, 0), 2)
            cv2.circle(frame, trajectory_2d[i], 3, (0, 255, 255), -1)

    def update_plot_surface(self):
        # Clear the plot and set up labels
        self.ax.cla()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # Set fixed axis limits to maintain a consistent cubic space
        self.ax.set_xlim(1, -1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)

        # Plot the ball trajectory with color gradient
        if self.ball_trajectory_3d:
            xs, ys, zs = zip(*self.ball_trajectory_3d)
            num_points = len(self.ball_trajectory_3d)

            # Plot points with gradient color and transparency
            for i in range(num_points):
                alpha = (i + 1) / num_points
                color = (1 - alpha, 0, alpha)
                self.ax.scatter(xs[i], ys[i], zs[i], color=color, s=10, alpha=alpha)

            # Plot connecting lines with gradient color and transparency
            for i in range(num_points - 1):
                alpha = (i + 1) / num_points
                color = (1 - alpha, 0, alpha, alpha)
                self.ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]], [zs[i], zs[i + 1]], color=color)

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
        # Check if last_point y > 0.02 and current_point y < 0, and reset trajectories if condition is met
        if last_point[1] > 0.02 and current_point[1] < 0:
            logging.info("Resetting trajectories due to y-coordinate threshold condition.")
            self.ball_trajectory_3d.clear()
            self.trajectory_2d_camera1.clear()
            self.trajectory_2d_camera2.clear()

def main():
    pygame.init()
    screen = pygame.display.set_mode((1200, 400))
    pygame.display.set_caption("Table Tennis 3D Ball Trajectory")
    game = TableTennisGame()
    running = True
    while running:
        frame1 = game.read_frame("camera1")
        frame2 = game.read_frame("camera2")

        # Detect ball position and update 2D trajectory for each camera
        ball_point1 = game.track_ball(frame1, "camera1", game.trajectory_2d_camera1)
        ball_point2 = game.track_ball(frame2, "camera2", game.trajectory_2d_camera2)

        # If both ball points are detected, calculate 3D coordinates
        if ball_point1 is not None and ball_point2 is not None:
            ball_3d = game.calculate_3d_coordinates(ball_point1, ball_point2)
            logging.info(f"3D Ball Coordinates: {ball_3d}")

            # Check if we need to reset the trajectories
            if game.ball_trajectory_3d:
                game.reset_trajectories_if_needed(game.ball_trajectory_3d[-1], ball_3d)

            game.ball_trajectory_3d.append(tuple(ball_3d))

            # Limit 3D trajectory to the last 100 points
            if len(game.ball_trajectory_3d) > 100:
                game.ball_trajectory_3d.pop(0)

        # Draw 2D trajectories on each frame
        game.draw_trajectory(frame1, game.trajectory_2d_camera1)
        game.draw_trajectory(frame2, game.trajectory_2d_camera2)

        # Draw calibration key points on each frame
        game.draw_key_points(frame1, game.camera1_points)
        game.draw_key_points(frame2, game.camera2_points)

        # Project the fixed 3D line to both frames
        game.project_3d_line_to_2d(frame1, game.proj_matrix1, game.camera1_rot_vec, game.camera1_trans_vec, game.camera1_intrinsics)
        game.project_3d_line_to_2d(frame2, game.proj_matrix2, game.camera2_rot_vec, game.camera2_trans_vec, game.camera2_intrinsics)

        # Update and retrieve the 3D plot surface
        plot_surface = game.update_plot_surface()

        # Convert frames to a format suitable for Pygame display
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # Resize frames to fit the Pygame display
        frame1_surface = pygame.surfarray.make_surface(cv2.resize(frame1_rgb, (400, 400)).swapaxes(0, 1))
        frame2_surface = pygame.surfarray.make_surface(cv2.resize(frame2_rgb, (400, 400)).swapaxes(0, 1))

        # Display frames and plot surface on the Pygame window
        screen.blit(frame1_surface, (0, 0))
        screen.blit(frame2_surface, (400, 0))
        screen.blit(plot_surface, (800, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.flip()
        pygame.time.delay(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
