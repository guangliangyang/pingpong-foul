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

        # List to store the 3D wrist trajectory
        self.wrist_trajectory_3d = []

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
        logging.debug(f"Calculating 3D coordinates from points: {point1}, {point2}")
        points_4d_homogeneous = cv2.triangulatePoints(self.proj_matrix1, self.proj_matrix2, point1, point2)
        points_3d = points_4d_homogeneous[:3] / points_4d_homogeneous[3]
        logging.debug(f"Calculated 3D coordinates: {points_3d.flatten()}")
        return points_3d.flatten()

    def process_frame(self, frame, camera_key):
        # Select the appropriate pose object based on the camera
        if camera_key == "camera1":
            pose = self.pose_camera1
        elif camera_key == "camera2":
            pose = self.pose_camera2
        else:
            return None

        # Convert the frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            logging.debug(f"{camera_key}: Landmarks detected by MediaPipe.")

            # Draw all landmarks for visual confirmation
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

            # Get the right wrist landmark
            wrist_landmark = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            wrist_x = int(wrist_landmark.x * frame.shape[1])
            wrist_y = int(wrist_landmark.y * frame.shape[0])

            logging.debug(f"{camera_key}: Detected wrist at ({wrist_x}, {wrist_y})")

            # Draw a circle around the wrist position for confirmation
            cv2.circle(frame, (wrist_x, wrist_y), 5, (0, 255, 0), -1)
            return np.array([[wrist_x], [wrist_y]], dtype=np.float32)

        logging.debug(f"{camera_key}: No landmarks detected by MediaPipe.")
        return None

    def update_plot_surface(self):
        # Clear the plot and set up labels
        self.ax.cla()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # Set fixed axis limits to maintain a consistent cubic space
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)

        # Plot the wrist trajectory with color gradient
        if self.wrist_trajectory_3d:
            xs, ys, zs = zip(*self.wrist_trajectory_3d)
            num_points = len(self.wrist_trajectory_3d)

            # Plot points with gradient color and transparency
            for i in range(num_points):
                alpha = (i + 1) / num_points  # Calculate transparency for each point
                color = (1 - alpha, 0, alpha)  # Color gradient from red (old) to blue (new)
                self.ax.scatter(xs[i], ys[i], zs[i], color=color, s=10, alpha=alpha)

            # Plot connecting lines with gradient color and transparency
            for i in range(num_points - 1):
                alpha = (i + 1) / num_points  # Calculate transparency for each line segment
                color = (1 - alpha, 0, alpha, alpha)  # Color gradient with transparency for line
                self.ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]], [zs[i], zs[i + 1]], color=color)

        # Set a fixed viewing angle for better depth perception
        self.ax.view_init(elev=20, azim=135)  # Adjust the elevation and azimuth as needed

        # Convert the Matplotlib plot to a Pygame surface
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        plot_surface = pygame.image.fromstring(canvas.tostring_rgb(), canvas.get_width_height(), "RGB")
        logging.debug("3D plot surface updated.")
        return plot_surface

    def read_frame(self, camera):
        # Read the next frame from the specified camera
        ret, frame = self.caps[camera].read()
        if not ret:
            logging.warning(f"End of video for {camera}. Restarting video.")
            self.caps[camera].set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            ret, frame = self.caps[camera].read()
        return frame


def main():
    pygame.init()
    screen_width, screen_height = 1200, 400  # Adjust width to accommodate the plot area
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Table Tennis 3D Wrist Trajectory")
    font = pygame.font.SysFont("Arial", 24)

    game = TableTennisGame()

    running = True
    while running:
        screen.fill((255, 255, 255))

        # Read frames from both cameras
        frame1 = game.read_frame("camera1")
        frame2 = game.read_frame("camera2")

        # Process each frame to get wrist 2D coordinates
        wrist_point1 = game.process_frame(frame1, "camera1")
        wrist_point2 = game.process_frame(frame2, "camera2")

        # If both wrist points are detected, calculate 3D coordinates
        if wrist_point1 is not None and wrist_point2 is not None:
            wrist_3d = game.calculate_3d_coordinates(wrist_point1, wrist_point2)
            logging.info(f"3D Wrist Coordinates: {wrist_3d}")
            game.wrist_trajectory_3d.append(tuple(wrist_3d))

            # Limit trajectory to the last 100 points
            if len(game.wrist_trajectory_3d) > 30:
                game.wrist_trajectory_3d.pop(0)

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
        screen.blit(plot_surface, (800, 0))  # Display the plot to the right of frame 2

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        pygame.display.flip()
        pygame.time.delay(30)  # Control frame update speed

    # Close MediaPipe
    game.pose_camera1.close()
    game.pose_camera2.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
