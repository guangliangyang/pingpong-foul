import cv2
import torch
from ultralytics import YOLO

# 模型和视频路径
model_path = "C:/workspace/projects/pingpong-foul/model/best-yolo11-transfer.pt"
video_paths = [
    "C:\\workspace\\datasets\\foul-video\\coach01.mp4",
    "C:\\workspace\\datasets\\foul-video\\game01.mp4",
    "C:\\workspace\\datasets\\foul-video\\01.mov",
    "C:\\workspace\\datasets\\foul-video\\c1.mp4",
    "C:\\workspace\\datasets\\foul-video\\01-5.mp4",
    "C:\\workspace\\datasets\\foul-video\\test-pp.mp4"
]

# 加载YOLO11模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path, verbose=False)
model.to(device)
print("Using GPU:", model.device)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    trajectory = []
    no_detection_frames = 0  # 记录没有检测到的帧数

    while cap.isOpened():
        success, frame = cap.read()
        if success:
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
                        distance = ((cx - last_point[0]) ** 2 + (cy - last_point[1]) ** 2) ** 0.5
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

            if no_detection_frames > 10:
                trajectory.clear()
                no_detection_frames = 0

            for i in range(1, len(trajectory)):
                cv2.line(annotated_frame, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)

            cv2.imshow("YOLO11 Tracking with Trajectory", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


# 循环遍历视频路径列表并处理每个视频
for video_path in video_paths:
    print(f"Processing video: {video_path}")
    process_video(video_path)
