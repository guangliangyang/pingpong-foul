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
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter



# Set up logging
logging.basicConfig(level=logging.DEBUG)


class TableTennisGame:
    def __init__(self):
        # Initialize YOLO model for ball tracking
        model_path = "C:/workspace/projects/pingpong-foul/model/best-yolo11-transfer03.pt"
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

    def draw_table_points(self, frame, points, coordinates):
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

    def find_key_points0(self,positions,fps = 60.0):

        #positions = ball_trajectory_3d  # [(x0, y0, z0, f0), (x1, y1, z1, f1), ...]
        velocities = []
        accelerations = []

        # Ensure there are enough positions to compute velocities and accelerations
        if len(positions) < 3:
            # Not enough data to compute key points
            return positions[0], positions[0], positions[-1]

        # Step 1: Compute velocities
        for i in range(1, len(positions)):
            x0, y0, z0, f0 = positions[i - 1]
            x1, y1, z1, f1 = positions[i]
            dt = (f1 - f0) / fps if f1 != f0 else 1 / fps  # Avoid division by zero

            vx = (x1 - x0) / dt
            vy = (y1 - y0) / dt
            vz = (z1 - z0) / dt
            velocities.append((vx, vy, vz, f1))

        # Step 2: Compute accelerations
        for i in range(1, len(velocities)):
            vx0, vy0, vz0, _ = velocities[i - 1]
            vx1, vy1, vz1, _ = velocities[i]
            dt = 1 / fps  # Time between frames is constant

            ax = (vx1 - vx0) / dt
            ay = (vy1 - vy0) / dt
            az = (vz1 - vz0) / dt
            accelerations.append((ax, ay, az))

        # Step 3: Identify the Hit Point
        # Find the maximum acceleration in Y and get the last occurrence
        if accelerations:
            max_ay = max(accelerations, key=lambda a: a[1])[1]
            hit_indices = [i for i, a in enumerate(accelerations) if a[1] == max_ay]
            hit_acc_index = max(hit_indices)
            hit_point_index = hit_acc_index + 2  # Adjusting index to match positions
            if hit_point_index < len(positions):
                hit_point = positions[hit_point_index]
            else:
                hit_point = positions[-1]
        else:
            hit_point = positions[-1]

        # Step 4: Identify the Highest Point
        positions_before_hit = positions[:hit_point_index]
        if positions_before_hit:
            highest_z = max(positions_before_hit, key=lambda p: p[2])[2]
            highest_indices = [i for i, p in enumerate(positions_before_hit) if p[2] == highest_z]
            highest_index = highest_indices[0]
            highest_point = positions[highest_index]
        else:
            highest_point = positions[0]

        # Step 5: Identify the Throw Point
        if highest_index >= 2:
            accelerations_before_highest = accelerations[:highest_index - 1]
            if accelerations_before_highest:
                max_az = max(accelerations_before_highest, key=lambda a: a[2])[2]
                throw_indices = [i for i, a in enumerate(accelerations_before_highest) if a[2] == max_az]
                throw_acc_index = throw_indices[0]
                throw_point_index = throw_acc_index + 1  # Adjust index for positions
                throw_point = positions[throw_point_index]
            else:
                throw_point = positions[0]  # Default to first point if not enough data
        else:
            throw_point = positions[0]  # Default to first point if not enough data

        return throw_point, highest_point, hit_point

    import numpy as np
    from scipy.signal import savgol_filter

    def find_key_points(self, positions, fps=60.0):
        # positions = ball_trajectory_3d  # [(x0, y0, z0, f0), (x1, y1, z1, f1), ...]

        # Ensure positions is a NumPy array for easier manipulation
        positions = np.array(positions)  # shape: (N, 4)
        if positions.shape[0] < 5:
            # Not enough data to compute key points
            return positions[0], positions[0], positions[-1]

        # Extract x, y, z, and frame indices
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        frame_indices = positions[:, 3]
        time = frame_indices / fps  # Convert frame indices to time

        # Step 1: Apply Savitzky-Golay filter to smooth the data
        window_size = 11  # Must be odd and less than the length of the data
        poly_order = 3  # Polynomial order less than window_size

        # Adjust window size if necessary
        if window_size >= positions.shape[0]:
            window_size = positions.shape[0] - 1 if positions.shape[0] % 2 == 1 else positions.shape[0] - 2
            if window_size < 3:
                # Not enough data for smoothing
                return positions[0], positions[0], positions[-1]

        x_smooth = savgol_filter(x, window_size, poly_order)
        y_smooth = savgol_filter(y, window_size, poly_order)
        z_smooth = savgol_filter(z, window_size, poly_order)

        # Step 2: Compute velocities using smoothed data
        vx = np.gradient(x_smooth, time)
        vy = np.gradient(y_smooth, time)
        vz = np.gradient(z_smooth, time)

        # Step 3: Identify the Highest Point (maximum Z value)
        highest_point_index = np.argmax(z_smooth)
        highest_point = positions[highest_point_index]

        # Step 4: Identify the Throw Point
        # Before the Highest Point, find where vz increases significantly
        if highest_point_index >= 2:
            vz_before_highest = vz[:highest_point_index]
            time_before_highest = time[:highest_point_index]
            # Compute acceleration in Z
            az = np.gradient(vz_before_highest, time_before_highest)
            # Find the index where az is maximum
            throw_point_index = np.argmax(az)
            throw_point = positions[throw_point_index]
        else:
            throw_point = positions[0]

        # Step 5: Identify the Hit Point
        # After the Highest Point, find where the speed increases abruptly
        if highest_point_index < len(vx) - 2:
            vx_after_highest = vx[highest_point_index:]
            vy_after_highest = vy[highest_point_index:]
            vz_after_highest = vz[highest_point_index:]
            time_after_highest = time[highest_point_index:]

            # Compute speed magnitude
            speed_after_highest = np.sqrt(vx_after_highest ** 2 + vy_after_highest ** 2 + vz_after_highest ** 2)
            # Compute speed difference
            speed_diff = np.diff(speed_after_highest)
            # Find the index where speed increases abruptly (maximum positive difference)
            hit_point_relative_index = np.argmax(speed_diff) + 1  # +1 to correct the index after diff
            hit_point_index = highest_point_index + hit_point_relative_index
            if hit_point_index < len(positions):
                hit_point = positions[hit_point_index]
            else:
                hit_point = positions[-1]
        else:
            hit_point = positions[-1]

        return throw_point, highest_point, hit_point

    def update_plot_surface(self, skeleton_3d, rotation_angle=135):
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

        # Draw 3D elements on the plot
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

        # Smooth 3D trajectory plotting using interpolation
        if len(self.ball_trajectory_3d) > 3:  # Check for sufficient points
            xs, ys, zs, frame_index = zip(*self.ball_trajectory_3d)

            # Ensure unique points to prevent interpolation errors
            if len(set(zip(xs, ys, zs))) > 3:
                try:
                    # Interpolate to create a smooth curve
                    tck, _ = splprep([xs, ys, zs], s=0)
                    smooth_points = splev(np.linspace(0, 1, 100), tck)

                    # Plot smooth trajectory
                    self.ax.plot(smooth_points[0], smooth_points[1], smooth_points[2], color='blue', linewidth=2)

                    # Plot points with a trailing effect
                    for i in range(len(xs)):
                        alpha = (i + 1) / len(xs)
                        self.ax.scatter(xs[i], ys[i], zs[i], color=(1 - alpha, 0, alpha), s=2, alpha=alpha)
                except ValueError as e:
                    logging.warning(f"Spline interpolation failed: {e}")
                    # Plot trajectory without interpolation if there's an error
                    self.ax.plot(xs, ys, zs, color="blue", linewidth=1, linestyle="--")

            # Step: Find and plot key points
            throw_point, highest_point, hit_point = self.find_key_points(self.ball_trajectory_3d)

            # Plot each key point with a different color
            self.ax.scatter(*throw_point[:3], color='yellow', s=2, label='Throw Point')
            self.ax.scatter(*highest_point[:3], color='red', s=2, label='Highest Point')
            self.ax.scatter(*hit_point[:3], color='red', s=2, label='Hit Point')

        # Set a fixed viewing angle for better depth perception
        self.ax.view_init(elev=20, azim=rotation_angle)

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
    # Set screen to fit a 2x2 grid layout with 640x480 for each video, keeping aspect ratio
    video_width, video_height = 640, 480
    screen_width = video_width * 2  # 1280
    screen_height = video_height * 2  # 960
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Table Tennis 3D Ball Trajectory - Enlarged Interface")
    game = TableTennisGame()
    running = True
    paused = False
    rotation_angle = 0  # Initial angle for rotating the 3D plot
    last_skeleton_3d = None  # Store the last skeleton data

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
            skeleton_3d = game.calculate_3d_skeleton(landmarks1, landmarks2, game.video_width, game.video_height) if landmarks1 and landmarks2 else None
            last_skeleton_3d = skeleton_3d  # Update the last known skeleton data
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
                frame_index = int(game.caps["camera1"].get(cv2.CAP_PROP_POS_FRAMES))  # Get current frame index
                print("frame_index=",frame_index)
                if game.ball_trajectory_3d:
                    game.reset_trajectories_if_needed(game.ball_trajectory_3d[-1], ball_3d)
                #game.ball_trajectory_3d.append(tuple(ball_3d))
                game.ball_trajectory_3d.append((*tuple(ball_3d),frame_index))

                if len(game.ball_trajectory_3d) > 100:
                    game.ball_trajectory_3d.pop(0)



            game.draw_trajectory(frame1, game.trajectory_2d_camera1)
            game.draw_trajectory(frame2, game.trajectory_2d_camera2)
            game.draw_table_points(frame1, game.camera1_points, game.camera1_3d_coordinates)
            game.draw_table_points(frame2, game.camera2_points, game.camera2_3d_coordinates)

            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            frame1_surface = pygame.surfarray.make_surface(cv2.resize(frame1_rgb, (video_width, video_height)).swapaxes(0, 1))
            frame2_surface = pygame.surfarray.make_surface(cv2.resize(frame2_rgb, (video_width, video_height)).swapaxes(0, 1))
            plot_surface_resized = pygame.transform.scale(plot_surface, (video_width, video_height))

            # Define the layout positions
            screen.blit(frame1_surface, (0, 0))  # Top-left: Camera 1
            screen.blit(frame2_surface, (video_width, 0))  # Top-right: Camera 2
            screen.blit(plot_surface_resized, (0, video_height))  # Bottom-left: 3D plot

            # Placeholder for future data panel (Bottom-right)
            data_panel = pygame.Surface((video_width, video_height))
            data_panel.fill((50, 50, 50))  # Dark gray placeholder color
            screen.blit(data_panel, (video_width, video_height))

        else:
            # Rotate the 3D plot while paused
            rotation_angle = (rotation_angle + 2) % 360
            plot_surface = game.update_plot_surface(last_skeleton_3d, rotation_angle=rotation_angle)
            plot_surface_resized = pygame.transform.scale(plot_surface, (video_width, video_height))
            screen.blit(plot_surface_resized, (0, video_height))

            # Draw static elements
            screen.blit(frame1_surface, (0, 0))
            screen.blit(frame2_surface, (video_width, 0))
            screen.blit(data_panel, (video_width, video_height))
        pygame.display.flip()
        pygame.time.delay(30)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
