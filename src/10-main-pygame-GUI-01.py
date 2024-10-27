import os
import sys
import threading
import pygame
import queue
import cv2
import numpy as np
import csv
import time
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import certifi
import itertools
import warnings
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


def get_heatmap_settings():
    # 定义自定义颜色映射，从黑色 -> 蓝色 -> 白色
    colors = [(0, 0, 0), (0, 0, 1), (1, 1, 1)]  # 黑色, 蓝色, 白色
    cmap_name = 'custom_blue_white'
    n_bins = 100  # 使用100个颜色等级
    cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    norm = mcolors.Normalize(vmin=0, vmax=100)
    return norm, cmap



def calculate_area(quad):
    # 使用四边形的四个角点计算面积
    return 0.5 * abs(
        quad[0][0]*quad[1][1] + quad[1][0]*quad[2][1] + quad[2][0]*quad[3][1] + quad[3][0]*quad[0][1] -
        (quad[1][0]*quad[0][1] + quad[2][0]*quad[1][1] + quad[3][0]*quad[2][1] + quad[0][0]*quad[3][1])
    )



def calculate_calories_burned(met, weight_kg, duration_minutes):
    calories_burned_per_minute = (met * weight_kg * 3.5) / 200
    total_calories_burned = calories_burned_per_minute * duration_minutes
    return total_calories_burned


def calculate_calories_burned_per_hour(calories_burned, total_time_minutes):
    if total_time_minutes == 0:
        return 0, "Entertainment"

    calories_burned_per_hour = (calories_burned / total_time_minutes) * 60

    if calories_burned_per_hour < 300:
        intensity = "Entertainment"
    elif 300 <= calories_burned_per_hour <= 400:
        intensity = "Moderate"
    else:
        intensity = "Competition"

    return calories_burned_per_hour, intensity


def estimate_met(average_speed, steps_count, swings_count):
    base_met = 3  # 基础活动的MET值，例如走路
    speed_factor = average_speed / 3.0  # 假设3.0 m/s 是一个高强度的运动速度
    steps_factor = steps_count / 1000  # 每1000步增加1个MET值
    swings_factor = swings_count / 100  # 每100次挥拍增加1个MET值

    estimated_met = base_met + speed_factor + steps_factor + swings_factor
    return min(estimated_met, 12)  # 限制最大MET值为12，防止过高


def calculate_layout(total_width, total_height, title_label_height, video_height, left_ratio, mode_label_height):
    left_width = int(total_width * left_ratio)
    right_width = total_width - left_width

    def region(x, y, width, height):
        return {"x": x, "y": y, "width": width, "height": height}

    regions = {
        "region1": region(0, 0, left_width, title_label_height),
        "region2": region(0, title_label_height, left_width, video_height),  # 合并后的区域
        "region6": region(left_width, 0, right_width, int(right_width * 9 / 16)),
        "region7": region(left_width, int(right_width * 9 / 16), right_width, total_height - int(right_width * 9 / 16))
    }

    if DEBUG:
        print(regions)

    return regions


def get_heatmap_settings():
    colors = [(0, 0, 0), (0, 0, 1), (1, 1, 1)]  # 黑色, 蓝色, 白色
    cmap_name = 'custom_blue_white'
    n_bins = 100  # 使用100个颜色等级
    cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    norm = mcolors.Normalize(vmin=0, vmax=100)
    return norm, cmap


class EightBallGame:
    def __init__(self):
        self.video_playing = False
        self.video_length = 0
        self.current_frame = 0
        self.cap = None
        # Update video path to the new video file
        self.video_path = 'C:\\workspace\\datasets\\foul-voideo\\c1.mp4'
        self.reset_variables()
        self.cap = None
        self.fps = 0
        self.delay = 0
        self.CV_CUDA_ENABLED = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.image_width = None
        self.image_height = None

    def reset_variables(self):
        self.previous_time = None
        self.start_time = time.time()

    def initialize_video_capture(self, source):
        self.cap = cv2.VideoCapture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print("FPS: {}".format(self.fps))
        self.delay = int(1000 / self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_fps(self):
        return self.fps

    def stop_video_analysis(self):
        self.video_playing = False
        if self.cap:
            self.cap.release()
            self.cap = None



    def process_video(self, frame):
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


        # YOLO inference for eight-ball detection
        detected_objects = self.detect_eight_ball(frame, model)
        print(detected_objects)

        timers['yolo_detection'] = time.time() - start_time
        start_time = time.time()

        # Draw bounding boxes and grid on a separate canvas
        canvas = self.draw_bounding_boxes_and_grid(image, detected_objects)

        timers['draw_bounding_boxes_and_grid'] = time.time() - start_time

        output_image = image


        if DEBUG:
            for step, duration in timers.items():
                print(f"{step}: {duration:.4f} seconds")

        return output_image, canvas

    def draw_bounding_boxes_and_grid(self, frame, detected_objects):
        # 创建一个空白的与frame大小相同的画布
        canvas = np.zeros_like(frame)

        # 标注YOLO对象点
        for (x1, y1, x2, y2, _, _) in detected_objects:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)


        return canvas

    def detect_eight_ball(self, frame, model, nms_threshold=0.4):
        results = model(frame)
        detected_objects = []

        boxes = []
        confidences = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                score = float(box.conf[0])  # Ensure the score is a float
                cls = int(box.cls[0])
                boxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to (x, y, width, height) format
                confidences.append(score)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.05, nms_threshold=nms_threshold)

        if len(indices) > 0:
            if isinstance(indices[0], list):
                indices = [i[0] for i in indices]
            for i in indices:
                x, y, w, h = boxes[i]
                detected_objects.append((x, y, x + w, y + h, confidences[i], cls))

        return detected_objects

    def analyze_video(self, queue):

        self.initialize_video_capture(self.video_path)
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.video_playing = True
        self.start_time = time.time()
        self.frame_count = 0

        while self.cap.isOpened() and self.video_playing:
            start_time = time.time()
            ret, frame = self.cap.read()
            time_read = time.time() - start_time

            if not ret:
                break

            start_time = time.time()
            image, canvas = self.process_video(frame)
            time_process_video = time.time() - start_time

            queue.put((image, canvas))

            self.frame_count += 1
            elapsed_time = (time.time() - self.start_time) * 1000
            expected_time = self.frame_count * self.delay
            wait_time = int(expected_time - elapsed_time)
            if DEBUG:
                print("int(expected_time - elapsed_time):", wait_time)
            if wait_time > 0:
                time.sleep(wait_time / 1000.0)

            if DEBUG:
                print(f"Read Frame Time: {time_read:.4f}s, Process Video Time: {time_process_video:.4f}s")

        self.video_playing = False
        self.cap.release()
        cv2.destroyAllWindows()

    def close_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class EightBallApp:

    def __init__(self, eight_ball_game):
        self.eight_ball_game = eight_ball_game
        self.eight_ball_game.app = self
        self.norm, self.cmap = get_heatmap_settings()
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
        mode_label_height = 30
        video_height = int(self.window_width * left_ratio * 9 / 16)

        self.layout = calculate_layout(self.window_width, self.window_height, title_label_height, video_height,
                                       left_ratio, mode_label_height)

        self.data_panel_update_interval = UPDATE_INTERVAL_DATA_PANEL_S
        self.speed_update_interval = 1.0
        self.data_panel_last_update_time = None
        self.speed_last_update_time = None
        self.skeleton_last_update_time = None
        self.statistics_table_last_update_time = None

        self.setup_ui()

        self.fps = self.eight_ball_game.get_fps()
        if self.fps is None:
            self.fps = 30

        self.video_thread = threading.Thread(target=self.eight_ball_game.analyze_video, args=(self.queue,))
        self.video_thread.daemon = True
        self.video_thread.start()

    def update_statistics_table(self, stats):
        pass

    def update_region6(self, canvas):
        pass

    def stop_video_analysis_thread(self):
        if self.video_thread is not None:
            self.eight_ball_game.stop_video_analysis()
            self.video_thread.join(timeout=5)
            self.video_thread = None


    def update_title_surface(self):
        title_text = f"{SYS_TITLE}"
        self.title_surface = self.create_label_surface(title_text, ("Arial", 28), "blue", "white")

        region1_x = self.layout['region1']['x']
        region1_y = self.layout['region1']['y']
        region1_width = self.layout['region1']['width']
        region1_height = self.layout['region1']['height']

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

    def update_video_panel(self, image, canvas):
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 1)

        if self.overlay_enabled:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            canvas = cv2.flip(canvas, 1)
            overlay = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        else:
            overlay = frame

        display_region = self.layout['region2']

        frame_height, frame_width = overlay.shape[:2]

        scale = min(display_region['width'] / frame_width,
                    display_region['height'] / frame_height)

        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        overlay_resized = cv2.resize(overlay, (new_width, new_height))

        self.video_surface.fill((0, 0, 0))

        overlay_resized = np.rot90(overlay_resized)
        overlay_resized = pygame.surfarray.make_surface(overlay_resized)

        self.video_surface.blit(overlay_resized, (
            (display_region['width'] - new_width) // 2,
            (display_region['height'] - new_height) // 2))
        self.screen.blit(self.video_surface,
                         (display_region['x'], display_region['y']))

        pygame.display.update()

    def update_skeleton_surface(self, skeleton_canvas):
        current_time = time.time()
        if self.skeleton_last_update_time is None or current_time - self.skeleton_last_update_time >= UPDATE_INTERVAL_SKELETON_SURFACE_MS / 1000.0:
            self.skeleton_last_update_time = current_time

            skeleton_image_np = np.rot90(skeleton_canvas, -1)
            skeleton_image_np = np.fliplr(skeleton_image_np)

            skeleton_surface = pygame.surfarray.make_surface(skeleton_image_np)

            skeleton_surface = pygame.transform.scale(skeleton_surface, (
                self.layout['region6']['width'], self.layout['region6']['height']))
            self.screen.blit(skeleton_surface, (self.layout['region6']['x'], self.layout['region6']['y']))

            pygame.display.update()


    def update_data_panel(self, corner_pocket_counts, side_pocket_counts, total_counts, calories_burned,
                          calories_burned_per_hour, intensity, duration):
        current_time = time.time()

        if self.data_panel_last_update_time is None:
            self.data_panel_last_update_time = current_time - 100000

        if current_time - self.data_panel_last_update_time < self.data_panel_update_interval:
            return

        self.data_panel_last_update_time = current_time

        panel_surface = pygame.Surface((self.layout['region7']['width'], self.layout['region7']['height']))
        panel_surface.fill((255, 255, 255))
        y_offset = 10

        norm, cmap = get_heatmap_settings()

        # 口袋进球数量统计
        pockets = [
            ("Top Left Pocket", corner_pocket_counts[0]),
            ("Top Middle Pocket", side_pocket_counts[0]),
            ("Top Right Pocket", corner_pocket_counts[1]),
            ("Bottom Left Pocket", corner_pocket_counts[2]),
            ("Bottom Middle Pocket", side_pocket_counts[1]),
            ("Bottom Right Pocket", corner_pocket_counts[3])
        ]

        for pocket, count in pockets:
            percentage = (count / total_counts) * 100 if total_counts > 0 else 0
            text = f"{pocket}: {percentage:.1f}%"
            font = pygame.font.SysFont("Arial", 22)
            text_surface = font.render(text, True, (0, 0, 0))
            panel_surface.blit(text_surface, (10, y_offset))
            y_offset += text_surface.get_height() + 5

            bar_x = 10
            bar_y = y_offset
            bar_width = self.layout['region7']['width'] - 20
            bar_height = 20
            pygame.draw.rect(panel_surface, (211, 211, 211), (bar_x, bar_y, bar_width, bar_height))

            color = cmap(norm(percentage))[:3]
            color = tuple(int(c * 255) for c in color)
            fill_width = int(bar_width * (percentage / 100))
            pygame.draw.rect(panel_surface, color, (bar_x, bar_y, fill_width, bar_height))
            y_offset += bar_height + 10  # 增加间隔

        y_offset += 20  # 空行

        # 卡路里统计信息
        font = pygame.font.SysFont("Arial", 22)
        text_surface = font.render(f"Calories Burned: {calories_burned:.1f} kcal", True, (0, 0, 0))
        panel_surface.blit(text_surface, (10, y_offset))
        y_offset += text_surface.get_height() + 5

        text_surface = font.render(f"Average Calories Burned per Hour: {calories_burned_per_hour:.1f} kcal", True,
                                   (0, 0, 0))
        panel_surface.blit(text_surface, (10, y_offset))
        y_offset += text_surface.get_height() + 5

        text_surface = font.render(f"Intensity: {intensity}", True, (0, 0, 0))
        panel_surface.blit(text_surface, (10, y_offset))
        y_offset += text_surface.get_height() + 5

        text_surface = font.render(f"Duration: {duration}", True, (0, 0, 0))
        panel_surface.blit(text_surface, (10, y_offset))
        y_offset += text_surface.get_height() + 5

        self.screen.blit(panel_surface, (self.layout['region7']['x'], self.layout['region7']['y']))
        pygame.display.update()

    def update_data(self):

        # 示例数据
        corner_pocket_counts = [5, 3, 2, 4]  # 四个角袋的进球数量
        side_pocket_counts = [6, 7]  # 两个侧袋的进球数量

        total_counts = sum(corner_pocket_counts) + sum(side_pocket_counts)

        # 卡路里统计信息
        weight_kg = 70
        total_frames = self.eight_ball_game.frame_count
        fps = self.eight_ball_game.get_fps()
        total_time_minutes = total_frames / (fps * 60)

        # 示例速度和挥拍次数
        average_speed = 1.2  # 单位: m/s
        step_count = 1000
        swing_count = 50

        estimated_met = estimate_met(average_speed, step_count, swing_count)
        calories_burned = calculate_calories_burned(estimated_met, weight_kg, total_time_minutes)
        calories_burned_per_hour, intensity = calculate_calories_burned_per_hour(calories_burned,
                                                                                 total_time_minutes)

        total_seconds = time.time() - self.eight_ball_game.start_time
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        duration = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

        self.update_data_panel(corner_pocket_counts, side_pocket_counts, total_counts, calories_burned,
                                   calories_burned_per_hour, intensity, duration)

    def setup_ui(self):
        # Existing setup code
        self.title_surface = pygame.Surface((self.layout['region1']['width'], self.layout['region1']['height']))
        self.title_surface.fill((255, 255, 255))

        # 合并后的区域
        self.video_surface = pygame.Surface(
            (self.layout['region2']['width'], self.layout['region2']['height']))
        self.video_surface.fill((0, 0, 0))

        self.skeleton_surface = pygame.Surface((self.layout['region6']['width'], self.layout['region6']['height']))
        self.skeleton_surface.fill((255, 255, 255))

        self.data_panel_surface = pygame.Surface((self.layout['region7']['width'], self.layout['region7']['height']))
        self.data_panel_surface.fill((255, 255, 255))

        # Add white borders for each region
        border_color = (0, 0, 255)
        border_width = 4  # Define the width of the border

        # Draw borders around each region
        pygame.draw.rect(self.screen, border_color, pygame.Rect(self.layout['region1']['x'],
                                                                self.layout['region1']['y'],
                                                                self.layout['region1']['width'],
                                                                self.layout['region1']['height']), border_width)
        pygame.draw.rect(self.screen, border_color, pygame.Rect(self.layout['region2']['x'],
                                                                self.layout['region2']['y'],
                                                                self.layout['region2']['width'],
                                                                self.layout['region2']['height']),
                         border_width)
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

        # Existing statistics table call
        stats = {
            'cue_ball': {
                'max_speed': 2.5,
                'avg_speed': 1.2,
                'max_energy': 1.8,
                'avg_energy': 0.9
            },
            'player': {
                'total_shots': 150,
                'potting_rate': 75.0,
                'foul_counts': 3,
                'max_consecutive_pots': 7
            }
        }
        self.update_statistics_table(stats)

    def update_speed_stats(self, speeds):
        current_time = time.time()
        if self.statistics_table_last_update_time is None or current_time - self.statistics_table_last_update_time >= UPDATE_INTERVAL_STATISTICS_TABLE_MS / 1000.0:
            self.statistics_table_last_update_time = current_time
            # 更新统计表
            stats = {
                'cue_ball': {
                    'max_speed': 2.5,
                    'avg_speed': 1.2,
                    'max_energy': 1.8,
                    'avg_energy': 0.9
                },
                'player': {
                    'total_shots': 150,
                    'potting_rate': 75.0,
                    'foul_counts': 3,
                    'max_consecutive_pots': 7
                }
            }
            self.update_statistics_table(stats)

    def mps_to_kph(self, speed_mps):
        return speed_mps * 3.6

    def update_mode_label_and_reset_var(self):
        self.eight_ball_game.reset_variables()

    def on_key_press(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.eight_ball_game.close_camera()
                pygame.quit()
                sys.exit()
            elif event.key == pygame.K_F3:
                self.overlay_enabled = not self.overlay_enabled
            elif event.key == pygame.K_F5:
                self.stop_video_analysis_thread()
                if self.mode != "real_time":
                    self.eight_ball_game.close_camera()
                    self.mode = "real_time"
                    self.update_mode_label_and_reset_var()
                    self.start_real_time_analysis()

            elif event.key == pygame.K_F1:
                self.detect_and_save_corners()

    def detect_and_save_corners(self):
        frame = self.queue.get_nowait()
        detected_objects = self.eight_ball_game.detect_eight_ball(frame, model)
        corners = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for (x1, y1, x2, y2, _, cls) in detected_objects if
                   cls == CORNER_POCKET_YOLO_CLASS_INDEX]

        best_corners = find_best_four_corners(corners)
        if best_corners is None:
            return

        save_corners_to_file(best_corners)
        self.eight_ball_game.corners = best_corners
        print("Corners saved to configuration file.")



    def start_real_time_analysis(self):
        self.stop_video_analysis_thread()
        self.eight_ball_game.reset_variables()
        self.eight_ball_game.initialize_video_capture(0)
        self.video_playing = True

        self.video_thread = threading.Thread(target=self.eight_ball_game.analyze_video, args=(self.queue,))
        self.video_thread.daemon = True
        self.video_thread.start()

    def start_video_analysis(self):
        self.stop_video_analysis_thread()
        self.eight_ball_game.reset_variables()
        self.eight_ball_game.initialize_video_capture(self.eight_ball_game.video_path)
        self.video_playing = True

        self.video_thread = threading.Thread(target=self.eight_ball_game.analyze_video, args=(self.queue,))
        self.video_thread.daemon = True
        self.video_thread.start()

    def main_loop(self):
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.eight_ball_game.close_camera()
                    pygame.quit()
                    sys.exit()
                self.on_key_press(event)

            if self.queue.empty():
                time.sleep(0.1)
                pass
            else:
                image, canvas = self.queue.get()
                self.update_video_panel(image, canvas)
                self.update_region6(canvas)
                self.update_data()
                pygame.display.update()


if __name__ == "__main__":
    eight_ball_game = EightBallGame()
    app = EightBallApp(eight_ball_game)
    app.main_loop()
