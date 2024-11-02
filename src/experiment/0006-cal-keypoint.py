import cv2
import torch
import numpy as np
from ultralytics import YOLO
import math

# 模型和视频路径
model_path = "C:/workspace/projects/pingpong-foul/model/best-yolo11-transfer02.pt"
video_paths = [
    "C:\\workspace\\datasets\\foul-video\\c2.mp4",
    # "C:\\workspace\\datasets\\foul-video\\01-5.mp4",
    # "C:\\workspace\\datasets\\foul-video\\coach01.mp4",
    # "C:\\workspace\\datasets\\foul-video\\game01.mp4",
    # "C:\\workspace\\datasets\\foul-video\\01.mov",
    # "C:\\workspace\\datasets\\foul-video\\test-pp.mp4"
]

# 加载YOLO11模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path, verbose=False)
model.to(device)
print("Using GPU:", model.device)


# 检测关键点函数
def find_key_points(trajectory, x_acceleration_threshold=5.0):
    throw_point = None
    highest_point = None
    hit_point = None

    # 找到抛球点
    for i in range(1, len(trajectory)):
        prev_x, prev_y = trajectory[i - 1]
        curr_x, curr_y = trajectory[i]
        speed = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
        if speed > 5.0:  # 假设5.0是一个小的速度阈值
            throw_point = trajectory[i - 1]
            break

    # 找到最高点 (必须在抛球点之后)
    if throw_point:
        highest_y = min([point[1] for point in trajectory[trajectory.index(throw_point):]])
        for point in trajectory[trajectory.index(throw_point):]:
            if point[1] == highest_y:
                highest_point = point
                break

    # 找到球拍击打点 (必须在最高点之后)
    if highest_point:
        for i in range(trajectory.index(highest_point) + 2, len(trajectory)):
            prev_x, _ = trajectory[i - 2]
            last_x, _ = trajectory[i - 1]
            curr_x, _ = trajectory[i]

            # 计算x方向的速度
            x_speed_prev = last_x - prev_x
            x_speed_curr = curr_x - last_x

            # 计算x方向的加速度（速度变化量）
            x_acceleration = x_speed_curr - x_speed_prev

            # 检查x方向加速度是否超过阈值，并且当前x速度大于0
            if x_speed_curr > 0 and x_acceleration > x_acceleration_threshold:
                hit_point = (last_x, trajectory[i - 1][1])  # 标记加速前的点为击打点
                break

    return throw_point, highest_point, hit_point


# 视频处理函数
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    trajectory = []
    no_detection_frames = 0  # 记录没有检测到的帧数
    paused = False  # 状态标记，是否暂停

    while cap.isOpened():
        if not paused:  # 仅当未暂停时才读取新帧
            success, frame = cap.read()
            if not success:
                break

            # 运行YOLO11跟踪并获取检测结果
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            annotated_frame = frame.copy()

            # 检查是否有检测到的对象
            if results[0].boxes:
                no_detection_frames = 0
                nearest_box = None
                min_distance = float("inf")

                if trajectory:
                    last_point = trajectory[-1]
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取检测框坐标
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 计算中心点
                        distance = math.sqrt((cx - last_point[0]) ** 2 + (cy - last_point[1]) ** 2)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_box = (cx, cy, x1, y1, x2, y2)
                else:
                    box = results[0].boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    nearest_box = ((x1 + x2) // 2, (y1 + y2) // 2, x1, y1, x2, y2)

                if nearest_box:
                    cx, cy, x1, y1, x2, y2 = nearest_box
                    trajectory.append((cx, cy))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                no_detection_frames += 1

            # 清空轨迹（如果有10帧没有检测到物体）
            if no_detection_frames > 10:
                trajectory.clear()
                no_detection_frames = 0

            # 绘制轨迹，仅从抛球点开始
            if len(trajectory) > 10:
                throw_point, highest_point, hit_point = find_key_points(trajectory)

                # Check if throw_point exists, and find its index
                if throw_point:
                    throw_index = trajectory.index(throw_point)

                    # Draw trajectory starting from the throw point
                    for i in range(throw_index + 1, len(trajectory)):
                        cv2.line(annotated_frame, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)
                        cv2.circle(annotated_frame, trajectory[i], 3, (0, 255, 255), -1)

                    # Draw key points
                    # Draw a square for the throw point
                    cv2.rectangle(annotated_frame,
                                  (throw_point[0] - 5, throw_point[1] - 5),
                                  (throw_point[0] + 5, throw_point[1] + 5),
                                  (0, 255, 255), -1)  # Yellow square

                    if highest_point:
                        # Draw a triangle for the highest point
                        triangle_points = np.array([
                            [highest_point[0], highest_point[1] - 6],  # Top vertex
                            [highest_point[0] - 5, highest_point[1] + 5],  # Bottom left vertex
                            [highest_point[0] + 5, highest_point[1] + 5]  # Bottom right vertex
                        ], np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(annotated_frame, [triangle_points], color=(0, 0, 255))  # Red triangle

                    if hit_point:
                        cv2.circle(annotated_frame, hit_point, 5, (0, 255, 0), -1)  # Green circle for hit point

            cv2.imshow("YOLO11 Tracking with Key Points", annotated_frame)

        # 键盘事件检测
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # 按 'q' 退出
            break
        elif key == ord(" "):  # 按空格键暂停或恢复
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


while True:
    for video_path in video_paths:
        print(f"Processing video: {video_path}")
        process_video(video_path)
