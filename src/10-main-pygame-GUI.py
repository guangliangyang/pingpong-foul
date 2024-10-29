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
from ultralytics import YOLO

# Load the YOLO model
model_file_path = os.path.join('..', 'model', 'best.pt')
model = YOLO(model_file_path)

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
        # Define your video sources here
        self.video_paths = {
            'camera1': 'C:\\workspace\\datasets\\foul-video\\c1.mp4',
            'camera2': 'C:\\workspace\\datasets\\foul-video\\c2.mp4',
            'camera3': 'C:\\workspace\\datasets\\foul-video\\c3.mp4',
            # 'camera4': 'C:\\workspace\\datasets\\foul-video\\c4.mp4',  # Removed camera4
        }
        self.reset_variables()
        self.CV_CUDA_ENABLED = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.image_width = None
        self.image_height = None
        self.fps = None  # Initialize self.fps

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
        self.table_surface_y = None  # Will be set after first frame
        self.end_line_x = None       # Will be set after first frame
        self.pixel_to_cm_ratio = None  # Will be set based on known object size

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
        # Process frames from all sources together
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

            # YOLO inference
            detected_objects = self.detect_objects(frame, model)
            logger.debug(f"Detected objects for {source_key}: {detected_objects}")

            timers['yolo_detection'] = time.time() - start_time
            start_time = time.time()

            # Initialize reference lines
            if self.table_surface_y is None or self.end_line_x is None:
                self.initialize_reference_lines(detected_objects)

            # Analyze fouls
            self.analyze_fouls(detected_objects, source_key)

            # Draw bounding boxes and grid on a separate canvas
            canvas = self.draw_bounding_boxes_and_grid(image, detected_objects)

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
        # Convert the BGR image to RGB before processing.
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

    def initialize_reference_lines(self, detected_objects):
        # Find the table in the detected objects
        for (x1, y1, x2, y2, _, class_id) in detected_objects:
            if class_id == 2:  # Assuming class_id 2 is 'Table'
                # Set table_surface_y as the top y-coordinate of the table
                self.table_surface_y = y1
                # Set end_line_x as the left x-coordinate of the table
                self.end_line_x = x1
                # Optionally set pixel_to_cm_ratio based on table dimensions
                table_height_pixels = y2 - y1
                known_table_height_cm = 76  # Standard table height in cm
                self.pixel_to_cm_ratio = known_table_height_cm / table_height_pixels
                logger.info(f"Initialized reference lines: table_surface_y={self.table_surface_y}, end_line_x={self.end_line_x}, pixel_to_cm_ratio={self.pixel_to_cm_ratio}")
                break

    def analyze_fouls(self, detected_objects, source_key):
        # Simplified foul detection logic
        # Class IDs (assuming):
        # 0: Ball
        # 1: Racket
        # 2: Table
        # 3: Player

        ball_positions = []
        racket_positions = []
        player_positions = []

        for (x1, y1, x2, y2, confidence, class_id) in detected_objects:
            if class_id == 0:
                ball_positions.append((x1, y1, x2, y2))
            elif class_id == 1:
                racket_positions.append((x1, y1, x2, y2))
            elif class_id == 3:
                player_positions.append((x1, y1, x2, y2))

        # Initialize previous positions if not present
        if source_key not in self.previous_ball_positions:
            self.previous_ball_positions[source_key] = []
        if source_key not in self.previous_racket_positions:
            self.previous_racket_positions[source_key] = []

        # Rule 1: Tossed from Above Table Surface
        if ball_positions and self.table_surface_y is not None:
            # Check if the ball is below the table surface when first detected
            current_ball_y = min(y1 for (x1, y1, x2, y2) in ball_positions)
            if len(self.previous_ball_positions[source_key]) == 0:
                # Ball just appeared
                if current_ball_y > self.table_surface_y:
                    # Ball is below table surface
                    self.foul_counts['Tossed from Below Table Surface'] += 1
                    logger.info(f"Foul detected: Tossed from Below Table Surface in {source_key}")

        # Rule 2: In Front of the End Line
        if player_positions and self.end_line_x is not None:
            # Check if player is beyond the end line
            player_x = max(x2 for (x1, y1, x2, y2) in player_positions)
            if player_x > self.end_line_x:
                # Player is over the end line
                self.foul_counts['In Front of the End Line'] += 1
                logger.info(f"Foul detected: In Front of the End Line in {source_key}")

        # Rule 3: Backward Angle More Than 30 Degrees
        if len(self.previous_racket_positions[source_key]) >= 2 and racket_positions:
            prev_x1, prev_y1, prev_x2, prev_y2 = self.previous_racket_positions[source_key][-1]
            curr_x1, curr_y1, curr_x2, curr_y2 = racket_positions[0]
            dx = curr_x1 - prev_x1
            dy = curr_y1 - prev_y1
            angle = np.degrees(np.arctan2(dy, dx))
            if abs(angle) > 30:
                self.foul_counts['Backward Angle Less More 30 Degrees'] += 1
                logger.info(f"Foul detected: Backward Angle More Than 30 Degrees in {source_key}")

        # Rule 4: Tossed Upward Less Than 16 cm
        if len(self.previous_ball_positions[source_key]) >= 1 and ball_positions and self.pixel_to_cm_ratio is not None:
            prev_ball_y = self.previous_ball_positions[source_key][0][1]
            current_ball_y = ball_positions[0][1]
            vertical_displacement_pixels = prev_ball_y - current_ball_y
            vertical_displacement_cm = vertical_displacement_pixels * self.pixel_to_cm_ratio
            if vertical_displacement_cm < 16:
                self.foul_counts['Tossed Upward Less Than 16 cm'] += 1
                logger.info(f"Foul detected: Tossed Upward Less Than 16 cm in {source_key}")

        # Update previous positions
        if ball_positions:
            self.previous_ball_positions[source_key].append(ball_positions[0])
            if len(self.previous_ball_positions[source_key]) > 10:
                self.previous_ball_positions[source_key].pop(0)
        if racket_positions:
            self.previous_racket_positions[source_key].append(racket_positions[0])
            if len(self.previous_racket_positions[source_key]) > 10:
                self.previous_racket_positions[source_key].pop(0)

    def draw_bounding_boxes_and_grid(self, frame, detected_objects):
        # Create a blank canvas the same size as frame
        canvas = np.zeros_like(frame)

        # Annotate detected objects
        for (x1, y1, x2, y2, _, class_id) in detected_objects:
            color = (0, 255, 0)
            label = ''
            if class_id == 0:
                label = 'Ball'
                color = (0, 0, 255)
            elif class_id == 1:
                label = 'Racket'
                color = (255, 0, 0)
            elif class_id == 2:
                label = 'Table'
                color = (0, 255, 0)
            elif class_id == 3:
                label = 'Player'
                color = (255, 255, 0)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return canvas

    def detect_objects(self, frame, model, nms_threshold=0.4):
        results = model(frame)
        detected_objects = []

        boxes = []
        confidences = []
        class_ids = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                score = float(box.conf[0])  # Ensure the score is a float
                cls = int(box.cls[0])
                boxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to (x, y, width, height) format
                confidences.append(score)
                class_ids.append(cls)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.05, nms_threshold=nms_threshold)

        if len(indices) > 0:
            if isinstance(indices[0], list):
                indices = [i[0] for i in indices]
            for i in indices:
                x, y, w, h = boxes[i]
                detected_objects.append((x, y, x + w, y + h, confidences[i], class_ids[i]))

        return detected_objects

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
            # Synchronize frame rate
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
        # Implement the method to display foul counts in Region 7
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
        # Update the console in Region 6 with real-time logs
        region = self.layout['region6']
        console_surface = pygame.Surface((region['width'], region['height']))
        console_surface.fill((0, 0, 0))  # Black background

        font = pygame.font.SysFont('Consolas', 16)
        log_lines = log_handler.get_logs()
        max_lines = region['height'] // (font.get_height() + 2)
        log_lines_to_display = log_lines[-max_lines:]

        y_offset = 5
        for line in log_lines_to_display:
            text_surface = font.render(line, True, (0, 255, 0))  # Green text
            console_surface.blit(text_surface, (5, y_offset))
            y_offset += font.get_height() + 2

        self.screen.blit(console_surface, (region['x'], region['y']))

    def update_region6(self):
        # Now handled by update_console
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
                frame = images[source_key]  # frame is a numpy array in BGR format

                # Convert from BGR to RGB for Pygame
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

                # Transpose the image to match Pygame's format
                overlay_resized = np.transpose(overlay_resized, (1, 0, 2))

                overlay_surface = pygame.surfarray.make_surface(overlay_resized)

                video_surface.blit(overlay_surface, (
                    (display_region['width'] - new_width) // 2,
                    (display_region['height'] - new_height) // 2))
                self.screen.blit(video_surface,
                                 (display_region['x'], display_region['y']))
            except Exception as e:
                logger.error(f"Error updating video panel for {source_key}: {e}")

        # Display the skeleton image in video_region_4
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

                # Transpose the image to match Pygame's format
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
        # No longer needed, handled in update_video_panel
        pass

    def update_data_panel(self, *args):
        # Implement as needed
        pass

    def update_data(self):
        # Implement as needed
        pass

    def setup_ui(self):
        # Existing setup code
        self.title_surface = pygame.Surface((self.layout['title_region']['width'], self.layout['title_region']['height']))
        self.title_surface.fill((255, 255, 255))

        # Create video surfaces for each video feed
        self.video_surfaces = {}
        for i in range(1, 5):
            region_key = f'video_region_{i}'
            region = self.layout[region_key]
            video_surface = pygame.Surface((region['width'], region['height']))
            video_surface.fill((0, 0, 0))
            self.video_surfaces[region_key] = video_surface

        # Add borders for each region
        border_color = (0, 0, 255)
        border_width = 4  # Define the width of the border

        # Draw borders around each region
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

        # Initialize Region 6 and Region 7 surfaces with white background
        self.region6_surface = pygame.Surface((self.layout['region6']['width'], self.layout['region6']['height']))
        self.region6_surface.fill((0, 0, 0))  # Black background for console

        self.region7_surface = pygame.Surface((self.layout['region7']['width'], self.layout['region7']['height']))
        self.region7_surface.fill((255, 255, 255))

        # Blit the surfaces onto the screen
        self.screen.blit(self.region6_surface, (self.layout['region6']['x'], self.layout['region6']['y']))
        self.screen.blit(self.region7_surface, (self.layout['region7']['x'], self.layout['region7']['y']))

        # Draw borders around Region 6 and Region 7
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
        # Implement as needed
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
            # Add other key events as needed

    def get_region_key_from_source(self, source_key):
        mapping = {
            'camera1': 'video_region_1',
            'camera2': 'video_region_2',
            'camera3': 'video_region_3',
            # 'camera4': 'video_region_4',  # Removed camera4
        }
        return mapping.get(source_key)

    def main_loop(self):
        #clock = pygame.time.Clock()

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
                # Update other regions or data as needed

            # Update console at defined interval
            current_time = time.time()
            if self.console_last_update_time is None or (current_time - self.console_last_update_time) >= self.console_update_interval:
                self.update_console()
                self.console_last_update_time = current_time

            pygame.display.update()
            #clock.tick(30)  # Limit the loop to 30 FPS

if __name__ == "__main__":
    game = TableTennisGame()
    app = TableTennisApp(game)
    app.main_loop()


