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

VIDEO_WIDTH = 640.0
VIDEO_HEIGHT = 480.0
ACTION_3D_Y_SPLIT = 0.4

SYS_TITLE = "Table Tennis Foul Detection System"

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
        self.action_segments = []

        # Load calibration data and key points for 3D triangulation
        self.load_calibration_data()
        self.load_key_points_for_calibration()

        # Paths for video sources
        self.video_paths = {
            'camera1': 'C:\\workspace\\datasets\\foul-video\\c1.mp4',
            'camera2': 'C:\\workspace\\datasets\\foul-video\\c2.mp4'
        }
        self.caps = {
            "camera1": cv2.VideoCapture(self.video_paths['camera1']),
            "camera2": cv2.VideoCapture(self.video_paths['camera2'])
        }

        self.foul_stats = {
            'In Front of the End Line': 0,
            'Beyond the sideline extension': 0,

            'Tossed from Below Table Surface': 0,
            'Backward Angle More Than 30°': 0,
            'Tossed Upward Less Than 16 cm': 0
        }
        self.serve_count = 0  # 记录球发球动作的次数
        self.foul_serve_count = 0
        self.last_frame_index = None  # 记录上一发球动作的最后帧索引
        self.foul_checked = False    # 标志位，表示当前轨迹是否已进行犯规统计

        # Store 3D ball trajectory and separate 2D trajectories for each camera
        self.ball_trajectory_3d = []
        self.trajectory_2d_camera1 = []
        self.trajectory_2d_camera2 = []

        self.all_trajectories = []

        # Initialize plot figure for trajectory visualization
        self.fig = plt.figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # Data panel text
        self.data_panel_text = ""

        self.frame_cache = []
        self.frame_cache_limit = 200

    def save_trajectory_to_file(self, filename="all_trajectories.json"):
        with open(filename, "w") as file:
            json.dump(self.all_trajectories, file, indent=4)

        print(f"All trajectories saved to {filename}")

    def record_trajectory(self):
        """
        Records the current trajectory with labeled key points for transformer training.
        Converts all values to standard Python data types for JSON serialization.
        """
        # Identify key points
        throw_point, highest_point, hit_point = self.find_serve_key_points(self.ball_trajectory_3d)

        # Store trajectory with labeled points
        trajectory_entry = {
            "trajectory": [
                {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "frame_index": int(point[3])
                }
                for point in self.ball_trajectory_3d
            ],
            "key_points": {
                "throw_point": {
                    "x": float(throw_point[0]),
                    "y": float(throw_point[1]),
                    "z": float(throw_point[2]),
                    "frame_index": int(throw_point[3])
                },
                "highest_point": {
                    "x": float(highest_point[0]),
                    "y": float(highest_point[1]),
                    "z": float(highest_point[2]),
                    "frame_index": int(highest_point[3])
                },
                "hit_point": {
                    "x": float(hit_point[0]),
                    "y": float(hit_point[1]),
                    "z": float(hit_point[2]),
                    "frame_index": int(hit_point[3])
                },
            }
        }

        # Append to all trajectories
        self.all_trajectories.append(trajectory_entry)
        print("Current trajectory recorded.")

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

    def load_key_points_for_calibration(self):
        with open('0010-calibration_key_points.json', 'r') as f:
            key_points_data = json.load(f)
        self.camera1_points = key_points_data["camera1_points"]
        self.camera2_points = key_points_data["camera2_points"]
        self.camera1_3d_coordinates = key_points_data["3d_coordinates"][:8]  # First 8 for camera1
        self.camera2_3d_coordinates = key_points_data["3d_coordinates"][8:]  # Next 8 for camera2
        logging.debug("2D key points and corresponding 3D coordinates loaded successfully.")

    def calculate_3d_skeleton(self, landmarks1, landmarks2, VIDEO_WIDTH, VIDEO_HEIGHT):
        skeleton_3d = []
        for lm1, lm2 in zip(landmarks1, landmarks2):
            point1 = np.array([[lm1.x * VIDEO_WIDTH], [lm1.y * VIDEO_HEIGHT]], dtype=np.float32)
            point2 = np.array([[lm2.x * VIDEO_WIDTH], [lm2.y * VIDEO_HEIGHT]], dtype=np.float32)
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

    def draw_table_points_calibration(self, frame, points, coordinates):
        for (x, y), (x3d, y3d, z3d) in zip(points, coordinates):
            cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
            text = f"({x3d:.2f}, {y3d:.2f}, {z3d:.2f})"
            #cv2.putText(frame, text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def calculate_3d_coordinates(self, point1, point2):
        points_4d_homogeneous = cv2.triangulatePoints(self.proj_matrix1, self.proj_matrix2, point1, point2)
        points_3d = points_4d_homogeneous[:3] / points_4d_homogeneous[3]
        return points_3d.flatten()

    def track_ball_2d(self, frame, camera_key, trajectory_2d):
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")
        if results[0].boxes:
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            frame_index = int(self.caps[camera_key].get(cv2.CAP_PROP_POS_FRAMES))  # Get current frame index

            # Append (x, y, frame_index) to trajectory_2d
            trajectory_2d.append((cx, cy, frame_index))

            # Limit the length of trajectory_2d to avoid overflow
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

    def draw_2d_trajectory(self, frame, trajectory_2d, max_distance=20):
        # Find the frame index where y > ACTION_3D_Y_SPLIT
        cutoff_index = None
        for i, point in enumerate(self.ball_trajectory_3d):
            _, y, _, frame_index = point
            if y > ACTION_3D_Y_SPLIT:
                cutoff_index = frame_index
                break

        # Filter the trajectory points to only include those before the cutoff frame index
        if cutoff_index is not None:
            filtered_trajectory = [pt for pt in trajectory_2d if pt[2] < cutoff_index]

        else:
            filtered_trajectory = trajectory_2d  # If no cutoff, draw the full trajectory

        # Now draw the filtered trajectory
        if len(filtered_trajectory) < 2:
            return  # Not enough points to draw

        for i in range(1, len(filtered_trajectory)):
            cv2.line(frame, filtered_trajectory[i - 1][:2], filtered_trajectory[i][:2], (255, 0, 0), 2)
            cv2.circle(frame, filtered_trajectory[i][:2], 3, (0, 255, 255), -1)

    def draw_3d_table_net(self):
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

    def draw_serve_area_3d_cube(self):
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


    def find_serve_key_points(self, positions, fps=60.0):
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

        # Step 3: Identify the Highest Point (maximum Z value with y < 0.2)
        sorted_indices = np.argsort(z_smooth)[::-1]  # Indices sorted by z value in descending order
        highest_point_index = None
        for index in sorted_indices:
            if y_smooth[index] < 0.2:  # Check if y < 0.2
                highest_point_index = index
                break
        if highest_point_index is None:
            highest_point_index = np.argmax(z_smooth)  # Fall back to max z without y < 0.2 condition
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
            throw_point = positions[throw_point_index + 1]
        else:
            throw_point = positions[0]

        # Step 5: Identify the Hit Point
        # After the Highest Point, find where vy (y-axis velocity) increases abruptly
        if highest_point_index < len(vy) - 2:
            vy_after_highest = vy[highest_point_index:]
            time_after_highest = time[highest_point_index:]

            # Compute the difference in vy to find abrupt changes
            vy_diff = np.diff(vy_after_highest)
            # Find the index where vy increases abruptly (maximum positive difference)
            hit_point_relative_index = np.argmax(vy_diff) + 1  # +1 to correct the index after diff
            hit_point_index = highest_point_index + hit_point_relative_index

            if hit_point_index < len(positions):
                hit_point = positions[hit_point_index]
            else:
                hit_point = positions[-1]
        else:
            hit_point = positions[-1]

        return throw_point, highest_point, hit_point

    def calculate_angle_with_vertical(self,throw_point, highest_point):
        # Define the vector from throw point to highest point
        vector_throw_to_highest = np.array([
            highest_point[0] - throw_point[0],
            highest_point[1] - throw_point[1],
            highest_point[2] - throw_point[2]
        ])

        # Define the vertical vector along the z-axis from the throw point
        vertical_vector = np.array([0, 0, 1])  # Vertical in z direction

        # Calculate the dot product and magnitudes
        dot_product = np.dot(vector_throw_to_highest, vertical_vector)
        magnitude_throw_to_highest = np.linalg.norm(vector_throw_to_highest)
        magnitude_vertical = np.linalg.norm(vertical_vector)

        # Calculate the angle in radians
        angle_rad = np.arccos(dot_product / (magnitude_throw_to_highest * magnitude_vertical))

        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def update_3D_plot_surface(self, skeleton_3d, rotation_angle=135):
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
        self.draw_serve_area_3d_cube()
        self.draw_3d_table_net()

        # Draw the table tennis table
        table_corners = [
            (0, 0, 0),
            (1.52, 0, 0),
            (1.52, 1.52, 0),
            (0, 1.52, 0),
            (0, 0, 0)  # Close the loop to complete the rectangle
        ]
        x_table, y_table, z_table = zip(*table_corners)
        self.ax.plot(x_table, y_table, z_table, color="red", linewidth=2)

        # Plot skeleton structure if available
        if skeleton_3d:
            self.draw_skeleton_3d(skeleton_3d)

        # Only draw the 3D trajectory between the throw point and hit point
        if len(self.ball_trajectory_3d) > 3:
            throw_point, highest_point, hit_point = self.find_serve_key_points(self.ball_trajectory_3d)

            try:
                throw_index = self.ball_trajectory_3d.index(tuple(throw_point))
                hit_index = self.ball_trajectory_3d.index(tuple(hit_point))
            except ValueError:
                logging.warning("Throw or hit point not found in trajectory data.")
                throw_index, hit_index = 0, len(self.ball_trajectory_3d) - 1

            trajectory_segment = self.ball_trajectory_3d[throw_index:hit_index + 1]

            if len(trajectory_segment) > 3:
                xs, ys, zs, frame_index = zip(*trajectory_segment)
                if len(set(zip(xs, ys, zs))) > 3:
                    try:
                        tck, _ = splprep([xs, ys, zs], s=0)
                        smooth_points = splev(np.linspace(0, 1, 100), tck)
                        self.ax.plot(smooth_points[0], smooth_points[1], smooth_points[2], color='blue', linewidth=2)
                    except ValueError as e:
                        logging.warning(f"Spline interpolation failed: {e}")
                        self.ax.plot(xs, ys, zs, color="blue", linewidth=1, linestyle="--")

            # Plot each key point with a different color and add labels
            self.ax.scatter(*throw_point[:3], color='yellow', s=2)
            self.ax.scatter(*highest_point[:3], color='red', s=2)
            self.ax.scatter(*hit_point[:3], color='green', s=2)

            self.check_and_perform_foul_statistics()
            self.save_trajectory_to_file("trajectories_for_training.json")

        self.draw_serve_area_3d_cube()

        # Set a fixed viewing angle for better depth perception
        self.ax.view_init(elev=20, azim=rotation_angle)

        # Manually create fixed legend entries
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Throw Point', markerfacecolor='yellow', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Highest Point', markerfacecolor='red', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Hit Point', markerfacecolor='green', markersize=8),
            Line2D([0], [0], linestyle='--', color='green', label='Serve Area')
        ]

        # Add the fixed-position legend
        self.ax.legend(handles=legend_elements, loc='upper left')

        # Convert the Matplotlib plot to a Pygame surface
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        plot_surface = pygame.image.fromstring(canvas.tostring_rgb(), canvas.get_width_height(), "RGB")
        logging.debug("3D plot surface updated.")
        return plot_surface

    def create_label_surface(self, text, font, bg, fg):
        pygame_font = pygame.font.SysFont(font[0], font[1])
        label_surface = pygame_font.render(text, True, pygame.Color(fg), pygame.Color(bg))
        return label_surface

    def draw_data_panel(self, data_panel_surface, screen, font, x_loc, y_loc, color=(255, 255, 255)):
        # Split `data_panel_text` by newlines for rendering line by line
        lines = self.data_panel_text.split("\n")

        # Define a larger font for the first line
        larger_font = pygame.font.Font(None, 40)  # Adjust size as needed for visibility

        # Define layout variables
        row_height = font.get_linesize() + 10
        column_width = data_panel_surface.get_width() // 2

        # Draw header with larger font
        summary_text = lines[0]  # The first line is the summary
        summary_surface = larger_font.render(summary_text, True, color)
        data_panel_surface.blit(summary_surface, (10, 10))

        # Find the "Current Fouls" section and extract the fouls listed under it
        if "Current Fouls:" in lines:
            current_fouls_start = lines.index("Current Fouls:") + 1
            current_fouls = [line.strip(" -") for line in lines[current_fouls_start:] if line.startswith(" -")]
        else:
            current_fouls = []

        # Render each line in the "Foul Statistics" section
        for i, line in enumerate(lines[2:8]):  # Lines 2-7 are Foul Statistics
            # Highlight in yellow if the line corresponds to a foul in current_fouls
            foul_color = (255, 255, 0) if any(foul in line for foul in current_fouls) else color
            foul_text_surface = font.render(line, True, foul_color)
            data_panel_surface.blit(foul_text_surface, (10, i * row_height + 70))  # Left column

        # Render each line in the "Current Action Statistics" section
        for i, line in enumerate(lines[9:15]):  # Lines 9-14 are Current Action Statistics
            action_text_surface = font.render(line, True, color)
            data_panel_surface.blit(action_text_surface, (column_width + 10, i * row_height + 70))  # Right column

        # Draw column and row lines
        for row in range(7):  # 6 rows, so 7 horizontal lines
            pygame.draw.line(data_panel_surface, (100, 100, 100), (0, row * row_height + 60),
                             (data_panel_surface.get_width(), row * row_height + 60), 1)

        pygame.draw.line(data_panel_surface, (100, 100, 100), (column_width, 60),
                         (column_width, row_height * 8), 1)

        # Display the data panel on the screen
        screen.blit(data_panel_surface, (x_loc, y_loc))

    def read_frame(self, camera):
        ret, frame = self.caps[camera].read()
        if not ret:
            self.caps[camera].set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.caps[camera].read()
        return frame

    def check_and_perform_foul_statistics(self):
        if len(self.ball_trajectory_3d) > 1 and not self.foul_checked:
            last_point = self.ball_trajectory_3d[-1]
            if last_point[1] > ACTION_3D_Y_SPLIT:
                self.perform_foul_check_and_statistics()
                self.foul_checked = True  # 设置标志，避免重复统计

    def reset_trajectories_if_needed(self, last_point, current_point):
        if last_point[1] > ACTION_3D_Y_SPLIT and current_point[1] < 0.2:
            self.reset_trajectories()

    def reset_trajectories(self):
        self.ball_trajectory_3d.clear()
        self.trajectory_2d_camera1.clear()
        self.trajectory_2d_camera2.clear()
        self.foul_checked = False

    def perform_foul_check_and_statistics(self):
        """In the current 3D trajectory ending, perform foul checks and update statistics."""
        if len(self.ball_trajectory_3d) < 5:
            return  # Skip if trajectory data is insufficient

        # Calculate key points
        throw_point, highest_point, hit_point = self.find_serve_key_points(self.ball_trajectory_3d)

        # Calculate Tossed Upward Distance
        tossed_upward_distance = (highest_point[2] - throw_point[2])*100

        # Current fouls for this trajectory
        current_fouls = []

        # Foul check conditions
        if throw_point[2] < 0:
            self.foul_stats['Tossed from Below Table Surface'] += 1
            current_fouls.append('Tossed from Below Table Surface')
        if throw_point[1] > 0 or highest_point[1] > 0 or hit_point[1] > 0:
            self.foul_stats['In Front of the End Line'] += 1
            current_fouls.append('In Front of the End Line')

        if (throw_point[0] < 0 or highest_point[0] < 0 or hit_point[0] < 0
                or throw_point[0] > 1.52 or highest_point[0] > 1.52 or hit_point[0] > 1.52):
            self.foul_stats['Beyond the sideline extension'] += 1
            current_fouls.append('Beyond the sideline extension')

        angle_with_vertical = self.calculate_angle_with_vertical(throw_point, highest_point)
        if angle_with_vertical > 30:
            self.foul_stats['Backward Angle More Than 30°'] += 1
            current_fouls.append('Backward Angle More Than 30°')
        if tossed_upward_distance < 16:
            self.foul_stats['Tossed Upward Less Than 16 cm'] += 1
            current_fouls.append('Tossed Upward Less Than 16 cm')

        # Update serve statistics
        self.serve_count += 1
        if current_fouls:
            self.foul_serve_count += 1

        start_frame_index = self.ball_trajectory_3d[0][3]  # First frame index
        end_frame_index = self.ball_trajectory_3d[-1][3]  # Last frame index
        action_segment = {
            'start_frame': start_frame_index,
            'throw_frame': int(throw_point[3]),
            'highest_frame': int(highest_point[3]),
            'hit_frame': int(hit_point[3]),
            'end_frame': end_frame_index,
            'current_fouls': current_fouls.copy()  # Store fouls for this action
        }
        self.action_segments.append(action_segment)

        # Update data panel text
        self.data_panel_text = (
            f"Foul Serves/Total Serve Actions:  {self.foul_serve_count} / {self.serve_count}\n\n"

            "Foul Statistics\n"
            f"In Front of the End Line: {self.foul_stats['In Front of the End Line']}\n"
            f"Beyond the sideline extension: {self.foul_stats['Beyond the sideline extension']}\n"
            f"Tossed from Below Table Surface: {self.foul_stats['Tossed from Below Table Surface']}\n"
            f"Backward Angle More Than 30°: {self.foul_stats['Backward Angle More Than 30°']}\n"
            f"Tossed Upward Less Than 16 cm: {self.foul_stats['Tossed Upward Less Than 16 cm']}\n\n"

            "Current Action Statistics\n"
            f"Throw Point: ({throw_point[0]:.2f}, {throw_point[1]:.2f}, {throw_point[2]:.2f})\n"
            f"Highest Point: ({highest_point[0]:.2f}, {highest_point[1]:.2f}, {highest_point[2]:.2f})\n"
            f"Hit Point: ({hit_point[0]:.2f}, {hit_point[1]:.2f}, {hit_point[2]:.2f})\n"
            f"Angle with Vertical: {angle_with_vertical:.2f} °\n"
            f"Tossed Upward Distance: {tossed_upward_distance:.2f} cm\n\n"

            "Current Fouls:\n" +
            "\n".join(f" - {foul}" for foul in current_fouls) if current_fouls else "None"
        )
        self.record_trajectory()

    def update_frame_cache(self, frame, frame_index):
        if len(self.frame_cache) >= self.frame_cache_limit:
            self.frame_cache.pop(0)
        self.frame_cache.append((frame_index, frame))

    def get_cached_frame_image(self, frame_index, frame_cache):
        for cached_index, frame in frame_cache:
            if cached_index == frame_index:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return pygame.surfarray.make_surface(cv2.resize(frame_rgb, (213, 160)).swapaxes(0, 1))
        return None

def main():
    pygame.init()

    screen_width, screen_height = 1530, 930

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Table Tennis 3D Ball Trajectory - Enlarged Interface")
    game = TableTennisGame()
    font = pygame.font.Font(None, 24)
    running, paused = True, False
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
            frame_index = int(game.caps["camera1"].get(cv2.CAP_PROP_POS_FRAMES))  # Get current frame index
            game.update_frame_cache(frame1, frame_index)
            game.VIDEO_WIDTH, game.VIDEO_HEIGHT = frame1.shape[1], frame1.shape[0]
            landmarks1 = game.process_frame_for_skeleton(frame1, game.pose_camera1, "camera1")
            landmarks2 = game.process_frame_for_skeleton(frame2, game.pose_camera2, "camera2")
            skeleton_3d = game.calculate_3d_skeleton(landmarks1, landmarks2, game.VIDEO_WIDTH, game.VIDEO_HEIGHT) if landmarks1 and landmarks2 else None
            last_skeleton_3d = skeleton_3d  # Update the last known skeleton data
            plot_surface = game.update_3D_plot_surface(skeleton_3d)

            ball_point1 = game.track_ball_2d(frame1, "camera1", game.trajectory_2d_camera1)
            ball_point2 = game.track_ball_2d(frame2, "camera2", game.trajectory_2d_camera2)

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
                print("frame_index=",frame_index)
                if game.ball_trajectory_3d:
                    game.reset_trajectories_if_needed(game.ball_trajectory_3d[-1], ball_3d)
                #game.ball_trajectory_3d.append(tuple(ball_3d))
                game.ball_trajectory_3d.append((*tuple(ball_3d),frame_index))

                if len(game.ball_trajectory_3d) > 100:
                    game.ball_trajectory_3d.pop(0)



            game.draw_2d_trajectory(frame1, game.trajectory_2d_camera1)
            game.draw_2d_trajectory(frame2, game.trajectory_2d_camera2)
            game.draw_table_points_calibration(frame1, game.camera1_points, game.camera1_3d_coordinates)
            game.draw_table_points_calibration(frame2, game.camera2_points, game.camera2_3d_coordinates)

            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            frame1_surface = pygame.surfarray.make_surface(cv2.resize(frame1_rgb, (int(VIDEO_WIDTH/2), int(VIDEO_HEIGHT/2))).swapaxes(0, 1))
            frame2_surface = pygame.surfarray.make_surface(cv2.resize(frame2_rgb, (int(VIDEO_WIDTH/2), int(VIDEO_HEIGHT/2))).swapaxes(0, 1))

            plot_surface_resized = pygame.transform.scale(plot_surface, (screen_width - VIDEO_WIDTH, VIDEO_HEIGHT*2))




        else:
            # Rotate the 3D plot while paused
            rotation_angle = (rotation_angle + 1) % 360
            plot_surface = game.update_3D_plot_surface(last_skeleton_3d, rotation_angle=rotation_angle)
            plot_surface_resized = pygame.transform.scale(plot_surface, (screen_width - VIDEO_WIDTH, VIDEO_HEIGHT*2))
            screen.blit(plot_surface_resized, (0, 0))

        # Define the layout positions
        screen.blit(frame1_surface, (screen_width - VIDEO_WIDTH, 0))  # Top-left: Camera 1
        screen.blit(frame2_surface, (screen_width - VIDEO_WIDTH / 2, 0))  # Top-right: Camera 2
        screen.blit(plot_surface_resized, (0, 0))  # Bottom-left: 3D plot

        draw_camera_title(game, screen, screen_width)
        bar_height, bar_y ,bar_x= draw_action_segmentation(frame_index, game, screen, screen_width)
        data_panel = pygame.Surface((VIDEO_WIDTH, VIDEO_HEIGHT))
        data_panel.fill((50, 50, 50))  # Dark gray placeholder color
        data_panel_y = bar_y + bar_height + 210
        game.draw_data_panel(data_panel, screen, font, screen_width - VIDEO_WIDTH, data_panel_y)
        draw_title(game, screen, screen_width)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


def draw_camera_title(game, screen, screen_width):
    # Add labels to the camera panels
    label_font = pygame.font.SysFont("Arial", 20)
    left_label = label_font.render("Left Camera", True, (255, 255, 255))
    right_label = label_font.render("Right Camera", True, (255, 255, 255))
    # Position labels on the video panels
    screen.blit(left_label, (screen_width - game.VIDEO_WIDTH + 10, 10))  # Left top corner of left camera
    right_label_width = right_label.get_width()
    screen.blit(right_label, (screen_width - right_label_width - 10, 10))  # Right top corner of right camera


def draw_action_segmentation(frame_index, game, screen, screen_width):
    # Draw the horizontal bar for action segments
    bar_height = 20
    bar_width = VIDEO_WIDTH
    bar_x = screen_width - VIDEO_WIDTH
    bar_y = VIDEO_HEIGHT / 2  # Position the bar under the video feeds
    # Draw the bar background
    pygame.draw.rect(screen, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height))  # Light gray background
    # Calculate min and max frame indices
    if game.action_segments:
        min_frame_index = min(segment['start_frame'] for segment in game.action_segments)
        max_frame_index = max(segment['end_frame'] for segment in game.action_segments)
    else:
        min_frame_index = 0
        max_frame_index = frame_index  # Use the current frame index
    # Avoid division by zero
    if max_frame_index == min_frame_index:
        max_frame_index += 1

    print("game.action_segments=",game.action_segments)
    # For each action segment, draw its representation on the bar
    for segment in game.action_segments:
        start_frame = segment['start_frame']
        end_frame = segment['end_frame']

        start_x = bar_x + ((start_frame - min_frame_index) / (max_frame_index - min_frame_index)) * bar_width
        end_x = bar_x + ((end_frame - min_frame_index) / (max_frame_index - min_frame_index)) * bar_width

        segment_width = end_x - start_x

        # Determine color based on fouls
        if segment['current_fouls']:
            color = (255, 0, 0)  # Red for fouls
        else:
            color = (0, 255, 0)  # Green for no fouls

        # Draw the rectangle for the action segment
        pygame.draw.rect(screen, color, (start_x, bar_y, segment_width, bar_height))

        # Draw markers for throw_frame, highest_frame, hit_frame
        throw_frame = segment['throw_frame']
        highest_frame = segment['highest_frame']
        hit_frame = segment['hit_frame']

        # Calculate positions
        throw_x = bar_x + ((throw_frame - min_frame_index) / (max_frame_index - min_frame_index)) * bar_width
        highest_x = bar_x + (
                (highest_frame - min_frame_index) / (max_frame_index - min_frame_index)) * bar_width
        hit_x = bar_x + ((hit_frame - min_frame_index) / (max_frame_index - min_frame_index)) * bar_width

        # Draw markers
        pygame.draw.line(screen, (0, 0, 255), (throw_x, bar_y), (throw_x, bar_y + bar_height),
                         2)  # Blue line for throw point
        pygame.draw.line(screen, (255, 255, 0), (highest_x, bar_y), (highest_x, bar_y + bar_height),
                         2)  # Yellow line for highest point
        pygame.draw.line(screen, (255, 165, 0), (hit_x, bar_y), (hit_x, bar_y + bar_height),
                         2)  # Orange line for hit point
    # Optionally, draw current frame marker
    current_x = bar_x + ((frame_index - min_frame_index) / (max_frame_index - min_frame_index)) * bar_width
    pygame.draw.line(screen, (255, 255, 255), (current_x, bar_y), (current_x, bar_y + bar_height),
                     1)  # White line for current frame

    # Add legend under the bar
    legend_font = pygame.font.SysFont("Arial", 16)
    legend_x = bar_x
    legend_y = bar_y + bar_height + 10  # Position directly under the bar

    # Draw colored rectangles with labels
    # Green rectangle for actions without fouls
    pygame.draw.rect(screen, (0, 255, 0), (legend_x, legend_y, 20, 10))
    legend_text_no_foul = legend_font.render("No Foul", True, (255, 255, 255))
    screen.blit(legend_text_no_foul, (legend_x + 25, legend_y - 5))

    # Red rectangle for actions with fouls
    pygame.draw.rect(screen, (255, 0, 0), (legend_x + 100, legend_y, 20, 10))
    legend_text_foul = legend_font.render("Foul", True, (255, 255, 255))
    screen.blit(legend_text_foul, (legend_x + 125, legend_y - 5))

    # Lines for key points
    # Blue line for throw point
    pygame.draw.line(screen, (0, 0, 255), (legend_x + 200, legend_y + 5), (legend_x + 220, legend_y + 5), 2)
    legend_text_throw = legend_font.render("Throw Point", True, (255, 255, 255))
    screen.blit(legend_text_throw, (legend_x + 225, legend_y - 5))

    # Yellow line for highest point
    pygame.draw.line(screen, (255, 255, 0), (legend_x + 350, legend_y + 5), (legend_x + 370, legend_y + 5), 2)
    legend_text_highest = legend_font.render("Highest Point", True, (255, 255, 255))
    screen.blit(legend_text_highest, (legend_x + 375, legend_y - 5))

    # Orange line for hit point
    pygame.draw.line(screen, (255, 165, 0), (legend_x + 500, legend_y + 5), (legend_x + 520, legend_y + 5), 2)
    legend_text_hit = legend_font.render("Hit Point", True, (255, 255, 255))
    screen.blit(legend_text_hit, (legend_x + 525, legend_y - 5))

    if game.action_segments:
        latest_segment = game.action_segments[-1]
        latest_segment_index = len(game.action_segments)
        throw_frame = latest_segment['throw_frame']
        highest_frame = latest_segment['highest_frame']
        hit_frame = latest_segment['hit_frame']

        # 从缓存中获取帧图像
        throw_img = game.get_cached_frame_image(throw_frame, game.frame_cache)
        highest_img = game.get_cached_frame_image(highest_frame, game.frame_cache)
        hit_img = game.get_cached_frame_image(hit_frame, game.frame_cache)

        image_y = legend_y + 30
        label_font = pygame.font.SysFont("Arial", 14)

        if throw_img:
            throw_frame_index = latest_segment['throw_frame']
            screen.blit(throw_img, (legend_x, image_y))
            throw_text = label_font.render(f"{latest_segment_index} Throw (Frame: {throw_frame_index})", True,
                                           (255, 255, 255))
            screen.blit(throw_text, (legend_x, image_y))

        if highest_img:
            highest_frame_index = latest_segment['highest_frame']
            screen.blit(highest_img, (legend_x + 220, image_y))
            highest_text = label_font.render(f"{latest_segment_index} Highest (Frame: {highest_frame_index})", True,
                                             (255, 255, 255))
            screen.blit(highest_text, (legend_x + 220, image_y))

        if hit_img:
            hit_frame_index = latest_segment['hit_frame']
            screen.blit(hit_img, (legend_x + 440, image_y))
            hit_text = label_font.render(f"{latest_segment_index} Hit (Frame: {hit_frame_index})", True,
                                         (255, 255, 255))
            screen.blit(hit_text, (legend_x + 440, image_y))

    return bar_height, bar_y,bar_x


def draw_title(game, screen, screen_width):
    # title
    title_text = f"{SYS_TITLE}"
    title_surface = game.create_label_surface(title_text, ("Arial", 28), "blue", "white")
    region1_x = 0
    region1_y = 0
    region1_width = screen_width - VIDEO_WIDTH
    region1_height = 60
    title_surface_width = title_surface.get_width()
    title_surface_height = title_surface.get_height()
    centered_x = region1_x + (region1_width - title_surface_width) // 2
    centered_y = region1_y + (region1_height - title_surface_height) // 2
    screen.fill((0, 0, 255), rect=[region1_x, region1_y, region1_width, region1_height])
    screen.blit(title_surface, (centered_x, centered_y))


if __name__ == "__main__":
    main()
