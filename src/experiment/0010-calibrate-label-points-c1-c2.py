import os
import sys
import threading
import pygame
import queue
import cv2
import numpy as np
import time
import logging
import mediapipe as mp
import json

SYS_TITLE = "Table Tennis Foul Detection System"
GOLDEN_RATIO = 1.618
DEBUG = True
CALIBRATION_FILE_PATH = "0010-calibration_data.json"

UPDATE_INTERVAL_SKELETON_SURFACE_MS = 100
UPDATE_INTERVAL_STATISTICS_TABLE_MS = 100
UPDATE_INTERVAL_DATA_PANEL_S = 5

def calculate_area(quad):
    return 0.5 * abs(
        quad[0][0]*quad[1][1] + quad[1][0]*quad[2][1] + quad[2][0]*quad[3][1] + quad[3][0]*quad[0][1] -
        (quad[1][0]*quad[0][1] + quad[2][0]*quad[1][1] + quad[3][0]*quad[2][1] + quad[0][0]*quad[3][1])
    )

def calculate_layout(total_width, total_height, title_label_height, left_ratio):
    left_width = int(total_width * left_ratio)
    right_width = total_width - left_width

    def region(x, y, width, height):
        return {"x": x, "y": y, "width": width, "height": height}

    video_height = total_height - title_label_height
    video_panel_width = left_width // 2
    video_panel_height = video_height // 2

    regions = {
        "title_region": region(0, 0, total_width, title_label_height),
        "video_region_1": region(0, title_label_height, video_panel_width, video_panel_height),
        "video_region_2": region(video_panel_width, title_label_height, video_panel_width, video_panel_height),
        "video_region_3": region(0, title_label_height + video_panel_height, video_panel_width, video_panel_height),
        "video_region_4": region(video_panel_width, title_label_height + video_panel_height, video_panel_width, video_panel_height),
        "region7": region(left_width, title_label_height, right_width, video_panel_height),
        "region6": region(left_width, title_label_height + video_panel_height, right_width, video_panel_height)
    }

    if DEBUG:
        print(regions)

    return regions

class LogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_records = []
        self.max_records = 1000

    def emit(self, record):
        log_entry = self.format(record)
        self.log_records.append(log_entry)
        if len(self.log_records) > self.max_records:
            self.log_records.pop(0)

    def get_logs(self):
        return self.log_records

log_handler = LogHandler()
log_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(log_handler)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

class TableTennisGame:
    def __init__(self):
        self.video_playing = False
        self.caps = {}
        self.video_paths = {
            'camera1': 'C:\\workspace\\datasets\\foul-video\\c1.mp4',
            'camera2': 'C:\\workspace\\datasets\\foul-video\\c2.mp4',
            'camera3': 'C:\\workspace\\datasets\\foul-video\\c3.mp4',
        }
        self.reset_variables()
        self.CV_CUDA_ENABLED = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.image_width = None
        self.image_height = None
        self.fps = None

        self.calibration_points_camera1 = []
        self.calibration_points_camera2 = []
        self.calibration_3d_coordinates = []

    def reset_variables(self):
        self.previous_time = None
        self.start_time = time.time()

    def initialize_video_captures(self):
        for index, (key, source) in enumerate(self.video_paths.items()):
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logger.error(f"Failed to open video source: {source}")
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error(f"Failed to read a frame from {source}")
                cap.release()
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                logger.warning(f"Failed to retrieve FPS from {source}, defaulting to 30 FPS")
                fps = 30

            logger.info(f"FPS for {key}: {fps}")

            self.caps[key] = {
                'cap': cap,
                'fps': fps,
                'delay': int(1000 / fps)
            }

            if index == 0:
                self.fps = fps

        if self.fps is None:
            self.fps = 30
        self.delay = int(1000 / self.fps)

    def get_fps(self):
        return self.fps

    def stop_video_analysis(self):
        self.video_playing = False
        for cap_info in self.caps.values():
            cap = cap_info['cap']
            if cap:
                cap.release()
        self.caps.clear()

    def save_calibration_data(self):
        calibration_data = {
            "camera1_points": self.calibration_points_camera1,
            "camera2_points": self.calibration_points_camera2,
            "3d_coordinates": self.calibration_3d_coordinates
        }
        with open(CALIBRATION_FILE_PATH, "w") as f:
            json.dump(calibration_data, f, indent=4)
        logger.info("Calibration data saved successfully.")

    def load_calibration_data(self):
        if os.path.exists(CALIBRATION_FILE_PATH):
            with open(CALIBRATION_FILE_PATH, "r") as f:
                data = json.load(f)
                self.calibration_points_camera1 = data["camera1_points"]
                self.calibration_points_camera2 = data["camera2_points"]
                self.calibration_3d_coordinates = data["3d_coordinates"]
            logger.info("Calibration data loaded successfully.")
        else:
            logger.warning("No calibration data found. Please run calibration first.")

    def click_event(self, event, x, y, flags, param):
        camera_id = param  # We pass camera_id through the mouse callback
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Click registered on {camera_id} at ({x}, {y})")
            x3d = float(input("Enter X coordinate in 3D space: "))
            y3d = float(input("Enter Y coordinate in 3D space: "))
            z3d = float(input("Enter Z coordinate in 3D space: "))

            if camera_id == 'camera1':
                self.calibration_points_camera1.append((x, y))
            elif camera_id == 'camera2':
                self.calibration_points_camera2.append((x, y))

            self.calibration_3d_coordinates.append((x3d, y3d, z3d))
            logger.info(f"3D coordinates for {camera_id} point added: ({x3d}, {y3d}, {z3d})")

            if len(self.calibration_points_camera1) == 8 and len(self.calibration_points_camera2) == 8:
                print("Calibration points collected. Saving data.")
                self.save_calibration_data()

    def calibrate_cameras(self):
        if len(self.calibration_points_camera1) < 8 or len(self.calibration_points_camera2) < 8:
            logger.error("Not enough calibration points. Please collect 8 points per camera.")
            return

        object_points = np.array(self.calibration_3d_coordinates, dtype=np.float32)
        image_points1 = np.array(self.calibration_points_camera1, dtype=np.float32)
        image_points2 = np.array(self.calibration_points_camera2, dtype=np.float32)

        _, rvec1, tvec1 = cv2.solvePnP(object_points, image_points1, np.eye(3), None)
        _, rvec2, tvec2 = cv2.solvePnP(object_points, image_points2, np.eye(3), None)

        logger.info(f"Camera1 Extrinsics: Rotation Vector={rvec1}, Translation Vector={tvec1}")
        logger.info(f"Camera2 Extrinsics: Rotation Vector={rvec2}, Translation Vector={tvec2}")

    def start_calibration(self):
        self.initialize_video_captures()

        for cam_name, cam_info in self.caps.items():
            cap = cam_info["cap"]
            _, frame = cap.read()

            cv2.imshow(f"Calibration - {cam_name}", frame)
            if cam_name == 'camera1':
                cv2.setMouseCallback(f"Calibration - {cam_name}", lambda event, x, y, flags, param: self.click_event(event, x, y, flags, 'camera1'))
            elif cam_name == 'camera2':
                cv2.setMouseCallback(f"Calibration - {cam_name}", lambda event, x, y, flags, param: self.click_event(event, x, y, flags, 'camera2'))

        print("Click on 8 points for each camera view and input their 3D coordinates.")
        print("Press 'q' to quit calibration mode.")

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.stop_video_analysis()

class TableTennisApp:
    def __init__(self, game):
        self.game = game
        self.game.app = self
        self.first_data_update = True
        self.mode = "video"
        self.overlay_enabled = False

        self.queue = queue.Queue(maxsize=1)
        pygame.init()
        self.screen = pygame.display.set_mode((1530, 930))
        pygame.display.set_caption(SYS_TITLE)

        self.window_width = 1530
        self.window_height = 930
        left_ratio = 1020 / 1530
        title_label_height = 60

        self.layout = calculate_layout(
            total_width=self.window_width,
            total_height=self.window_height,
            title_label_height=title_label_height,
            left_ratio=left_ratio
        )

    def main_loop(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game.close_camera()
                    pygame.quit()
                    sys.exit()

            pygame.display.update()

if __name__ == "__main__":
    game = TableTennisGame()
    app = TableTennisApp(game)
    game.start_calibration()  # Start calibration process
    app.main_loop()
