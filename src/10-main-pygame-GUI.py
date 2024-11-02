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

SYS_TITLE = "Table Tennis Foul Detection System"
GOLDEN_RATIO = 1.618
DEBUG = True

UPDATE_INTERVAL_SKELETON_SURFACE_MS = 100
UPDATE_INTERVAL_STATISTICS_TABLE_MS = 100
UPDATE_INTERVAL_DATA_PANEL_S = 5

def calculate_area(quad):
    # Calculate the area of a quadrilateral given its four corner points
    return 0.5 * abs(
        quad[0][0]*quad[1][1] + quad[1][0]*quad[2][1] + quad[2][0]*quad[3][1] + quad[3][0]*quad[0][1] -
        (quad[1][0]*quad[0][1] + quad[2][0]*quad[1][1] + quad[3][0]*quad[2][1] + quad[0][0]*quad[3][1])
    )

def calculate_layout(total_width, total_height, title_label_height, left_ratio):
    left_width = int(total_width * left_ratio)
    right_width = total_width - left_width

    def region(x, y, width, height):
        return {"x": x, "y": y, "width": width, "height": height}

    # Use the remaining height after the title region for videos
    video_height = total_height - title_label_height

    # Divide the video area into 2 rows and 2 columns
    video_panel_width = left_width // 2
    video_panel_height = video_height // 2

    # Swap positions of region6 and region7
    regions = {
        "title_region": region(0, 0, total_width, title_label_height),
        "video_region_1": region(0, title_label_height, video_panel_width, video_panel_height),
        "video_region_2": region(video_panel_width, title_label_height, video_panel_width, video_panel_height),
        "video_region_3": region(0, title_label_height + video_panel_height, video_panel_width, video_panel_height),
        "video_region_4": region(video_panel_width, title_label_height + video_panel_height, video_panel_width, video_panel_height),
        "region7": region(left_width, title_label_height, right_width, video_panel_height),  # Swapped
        "region6": region(left_width, title_label_height + video_panel_height, right_width, video_panel_height)  # Swapped
    }

    if DEBUG:
        print(regions)

    return regions

# Custom logging handler to store logs in a list
class LogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_records = []
        self.max_records = 1000  # Limit the number of records to prevent memory issues

    def emit(self, record):
        log_entry = self.format(record)
        self.log_records.append(log_entry)
        if len(self.log_records) > self.max_records:
            self.log_records.pop(0)

    def get_logs(self):
        return self.log_records

# Set up the logging system
log_handler = LogHandler()
log_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(log_handler)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

class TableTennisGame:
    def __init__(self):
        self.video_playing = False
        self.video_length = 0
        self.current_frame = 0
        self.caps = {}  # Dictionary to hold multiple video captures
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

        # Initialize foul counts
        self.foul_counts = {
            'Tossed from Below Table Surface': 0,
            'In Front of the End Line': 0,
            'Backward Angle More Than 30 Degrees': 0,
            'Tossed Upward Less Than 16 cm': 0
        }

        # Variables to track previous positions
        self.previous_ball_positions = {}
        self.previous_racket_positions = {}

        # Define constants (adjust these values based on your setup)
        self.table_surface_y = None
        self.end_line_x = None
        self.pixel_to_cm_ratio = None

    def reset_variables(self):
        self.previous_time = None
        self.start_time = time.time()

    def initialize_video_captures(self):
        for index, (key, source) in enumerate(self.video_paths.items()):
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logger.error(f"Failed to open video source: {source}")
                continue  # Skip to the next source

            # Read a frame to initialize the capture
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error(f"Failed to read a frame from {source}")
                cap.release()
                continue  # Skip to the next source

            # Now, get the FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                logger.warning(f"Failed to retrieve FPS from {source}, defaulting to 30 FPS")
                fps = 30  # Default FPS if not available

            logger.info(f"FPS for {key}: {fps}")

            self.caps[key] = {
                'cap': cap,
                'fps': fps,
                'delay': int(1000 / fps)
            }

            if index == 0:
                self.fps = fps

        if self.fps is None:
            self.fps = 30  # Default FPS if none of the sources provided it
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

    def process_video(self, frames):
        output_images = {}
        canvases = {}
        skeleton_image = None  # To store the skeleton image from camera1

        for source_key, frame in frames.items():
            timers = {}
            start_time = time.time()

            if self.CV_CUDA_ENABLED:
                cv2.cuda.setDevice(1)

            timers['initial_setup'] = time.time() - start_time
            start_time = time.time()

            image = frame

            if self.image_width is None or self.image_height is None:
                self.image_width = image.shape[1]
                self.image_height = image.shape[0]

            # Initialize reference lines if needed

            # Draw bounding boxes and grid on a separate canvas
            canvas = self.draw_bounding_boxes_and_grid(image)

            timers['draw_bounding_boxes_and_grid'] = time.time() - start_time

            output_image = image

            if DEBUG:
                for step, duration in timers.items():
                    logger.debug(f"{source_key} - {step}: {duration:.4f} seconds")

            output_images[source_key] = output_image
            canvases[source_key] = canvas

            # Process skeleton for camera1
            if source_key == 'camera1':
                skeleton_image = self.process_skeleton(frame)

        return output_images, canvases, skeleton_image

    def process_skeleton(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Create a blank image to draw the skeleton
        skeleton_image = np.zeros_like(frame)

        if results.pose_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                skeleton_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )

        return skeleton_image

    def analyze_fouls(self, source_key):
        # Simplified foul detection logic
        pass

    def draw_bounding_boxes_and_grid(self, frame):
        canvas = np.zeros_like(frame)
        return canvas

    def analyze_video(self, queue):
        self.video_playing = True
        self.start_time = time.time()
        self.frame_count = 0

        while self.video_playing:
            frames = {}
            for key, cap_info in self.caps.items():
                cap = cap_info['cap']
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning(f"Failed to read frame from {key}")
                    self.video_playing = False
                    break
                frames[key] = frame

            if not self.video_playing:
                break

            output_images, canvases, skeleton_image = self.process_video(frames)
            queue.put((output_images, canvases, skeleton_image))

            self.frame_count += 1
            time.sleep(self.delay / 1000.0)

        self.stop_video_analysis()

    def close_camera(self):
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

        self.data_panel_update_interval = UPDATE_INTERVAL_DATA_PANEL_S
        self.speed_update_interval = 1.0
        self.data_panel_last_update_time = None
        self.speed_last_update_time = None
        self.skeleton_last_update_time = None
        self.statistics_table_last_update_time = None
        self.console_last_update_time = None
        self.console_update_interval = 0.5  # Update console every 0.5 seconds

        self.setup_ui()

        # Initialize video captures before getting FPS
        self.game.initialize_video_captures()

        self.fps = self.game.get_fps()
        if self.fps is None:
            self.fps = 30

        self.video_thread = threading.Thread(target=self.game.analyze_video, args=(self.queue,))
        self.video_thread.daemon = True
        self.video_thread.start()

    def update_statistics_table(self):
        region = self.layout['region7']
        stats_surface = pygame.Surface((region['width'], region['height']))
        stats_surface.fill((255, 255, 255))

        font = pygame.font.SysFont('Arial', 24)
        y_offset = 20

        title_text = "Foul Counts"
        title_surface = font.render(title_text, True, (0, 0, 0))
        stats_surface.blit(title_surface, (10, y_offset))
        y_offset += 40

        for rule, count in self.game.foul_counts.items():
            text = f"{rule}: {count}"
            text_surface = font.render(text, True, (0, 0, 0))
            stats_surface.blit(text_surface, (10, y_offset))
            y_offset += 30

        self.screen.blit(stats_surface, (region['x'], region['y']))

    def update_console(self):
        region = self.layout['region6']
        console_surface = pygame.Surface((region['width'], region['height']))
        console_surface.fill((0, 0, 0))

        font = pygame.font.SysFont('Consolas', 16)
        log_lines = log_handler.get_logs()
        max_lines = region['height'] // (font.get_height() + 2)
        log_lines_to_display = log_lines[-max_lines:]

        y_offset = 5
        for line in log_lines_to_display:
            text_surface = font.render(line, True, (0, 255, 0))
            console_surface.blit(text_surface, (5, y_offset))
            y_offset += font.get_height() + 2

        self.screen.blit(console_surface, (region['x'], region['y']))

    def update_region6(self):
        pass

    def stop_video_analysis_thread(self):
        if self.video_thread is not None:
            self.game.stop_video_analysis()
            self.video_thread.join(timeout=5)
            self.video_thread = None

    def update_title_surface(self):
        title_text = f"{SYS_TITLE}"
        self.title_surface = self.create_label_surface(title_text, ("Arial", 28), "blue", "white")

        region1_x = self.layout['title_region']['x']
        region1_y = self.layout['title_region']['y']
        region1_width = self.layout['title_region']['width']
        region1_height = self.layout['title_region']['height']

        title_surface_width = self.title_surface.get_width()
        title_surface_height = self.title_surface.get_height()

        centered_x = region1_x + (region1_width - title_surface_width) // 2
        centered_y = region1_y + (region1_height - title_surface_height) // 2

        self.screen.fill((0, 0, 255), rect=[region1_x, region1_y, region1_width, region1_height])

        self.screen.blit(self.title_surface, (centered_x, centered_y))
        pygame.display.update()

    def create_label_surface(self, text, font, bg, fg):
        pygame_font = pygame.font.SysFont(font[0], font[1])
        label_surface = pygame_font.render(text, True, pygame.Color(fg), pygame.Color(bg))
        return label_surface

    def update_video_panel(self, images, canvases, skeleton_image):
        for source_key in images.keys():
            try:
                frame = images[source_key]

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.overlay_enabled:
                    canvas = canvases[source_key]
                    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                    overlay = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
                else:
                    overlay = frame

                region_key = self.get_region_key_from_source(source_key)
                display_region = self.layout[region_key]
                video_surface = self.video_surfaces[region_key]

                frame_height, frame_width = overlay.shape[:2]

                scale = min(display_region['width'] / frame_width,
                            display_region['height'] / frame_height)

                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)

                overlay_resized = cv2.resize(overlay, (new_width, new_height))

                video_surface.fill((0, 0, 0))

                overlay_resized = np.transpose(overlay_resized, (1, 0, 2))

                overlay_surface = pygame.surfarray.make_surface(overlay_resized)

                video_surface.blit(overlay_surface, (
                    (display_region['width'] - new_width) // 2,
                    (display_region['height'] - new_height) // 2))
                self.screen.blit(video_surface,
                                 (display_region['x'], display_region['y']))
            except Exception as e:
                logger.error(f"Error updating video panel for {source_key}: {e}")

        if skeleton_image is not None:
            try:
                skeleton_image_rgb = cv2.cvtColor(skeleton_image, cv2.COLOR_BGR2RGB)
                region_key = 'video_region_4'
                display_region = self.layout[region_key]
                video_surface = self.video_surfaces[region_key]

                frame_height, frame_width = skeleton_image_rgb.shape[:2]

                scale = min(display_region['width'] / frame_width,
                            display_region['height'] / frame_height)

                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)

                skeleton_resized = cv2.resize(skeleton_image_rgb, (new_width, new_height))

                video_surface.fill((0, 0, 0))

                skeleton_resized = np.transpose(skeleton_resized, (1, 0, 2))

                skeleton_surface = pygame.surfarray.make_surface(skeleton_resized)

                video_surface.blit(skeleton_surface, (
                    (display_region['width'] - new_width) // 2,
                    (display_region['height'] - new_height) // 2))
                self.screen.blit(video_surface,
                                 (display_region['x'], display_region['y']))
            except Exception as e:
                logger.error(f"Error updating skeleton display: {e}")

        pygame.display.update()

    def update_skeleton_surface(self, skeleton_canvas):
        pass

    def update_data_panel(self, *args):
        pass

    def update_data(self):
        pass

    def setup_ui(self):
        self.title_surface = pygame.Surface((self.layout['title_region']['width'], self.layout['title_region']['height']))
        self.title_surface.fill((255, 255, 255))

        self.video_surfaces = {}
        for i in range(1, 5):
            region_key = f'video_region_{i}'
            region = self.layout[region_key]
            video_surface = pygame.Surface((region['width'], region['height']))
            video_surface.fill((0, 0, 0))
            self.video_surfaces[region_key] = video_surface

        border_color = (0, 0, 255)
        border_width = 4

        pygame.draw.rect(self.screen, border_color, pygame.Rect(self.layout['title_region']['x'],
                                                                self.layout['title_region']['y'],
                                                                self.layout['title_region']['width'],
                                                                self.layout['title_region']['height']), border_width)
        for i in range(1, 5):
            region_key = f'video_region_{i}'
            region = self.layout[region_key]
            pygame.draw.rect(self.screen, border_color, pygame.Rect(region['x'],
                                                                    region['y'],
                                                                    region['width'],
                                                                    region['height']), border_width)

        self.region6_surface = pygame.Surface((self.layout['region6']['width'], self.layout['region6']['height']))
        self.region6_surface.fill((0, 0, 0))

        self.region7_surface = pygame.Surface((self.layout['region7']['width'], self.layout['region7']['height']))
        self.region7_surface.fill((255, 255, 255))

        self.screen.blit(self.region6_surface, (self.layout['region6']['x'], self.layout['region6']['y']))
        self.screen.blit(self.region7_surface, (self.layout['region7']['x'], self.layout['region7']['y']))

        pygame.draw.rect(self.screen, border_color, pygame.Rect(self.layout['region6']['x'],
                                                                self.layout['region6']['y'],
                                                                self.layout['region6']['width'],
                                                                self.layout['region6']['height']), border_width)
        pygame.draw.rect(self.screen, border_color, pygame.Rect(self.layout['region7']['x'],
                                                                self.layout['region7']['y'],
                                                                self.layout['region7']['width'],
                                                                self.layout['region7']['height']), border_width)

        self.update_title_surface()
        self.update_mode_label_and_reset_var()

    def update_speed_stats(self, speeds):
        pass

    def mps_to_kph(self, speed_mps):
        return speed_mps * 3.6

    def update_mode_label_and_reset_var(self):
        self.game.reset_variables()

    def on_key_press(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.game.close_camera()
                pygame.quit()
                sys.exit()
            elif event.key == pygame.K_F3:
                self.overlay_enabled = not self.overlay_enabled

    def get_region_key_from_source(self, source_key):
        mapping = {
            'camera1': 'video_region_1',
            'camera2': 'video_region_2',
            'camera3': 'video_region_3',
        }
        return mapping.get(source_key)

    def main_loop(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game.close_camera()
                    pygame.quit()
                    sys.exit()
                self.on_key_press(event)

            if not self.queue.empty():
                images, canvases, skeleton_image = self.queue.get()
                self.update_video_panel(images, canvases, skeleton_image)
                self.update_statistics_table()

            current_time = time.time()
            if self.console_last_update_time is None or (current_time - self.console_last_update_time) >= self.console_update_interval:
                self.update_console()
                self.console_last_update_time = current_time

            pygame.display.update()

if __name__ == "__main__":
    game = TableTennisGame()
    app = TableTennisApp(game)
    app.main_loop()
