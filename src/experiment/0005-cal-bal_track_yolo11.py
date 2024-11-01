import cv2
import torch
from ultralytics import YOLO

# 模型和视频路径
model_path = "C:/workspace/projects/pingpong-foul/model/best-yolo11-transfer.pt"
video_path = "C:\\workspace\\datasets\\foul-video\\c2.mp4"

# 加载YOLO11模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path, verbose=False)
model.to(device)
print("Using GPU:", model.device)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 初始化一个列表，用于存储检测到的对象的中心点位置
trajectory = []
no_detection_frames = 0  # 记录没有检测到的帧数

# 主循环，遍历视频帧
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # 运行YOLO11跟踪并获取检测结果
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        # 复制当前帧，用于绘制轨迹
        annotated_frame = frame.copy()

        # 检查是否有检测到的对象
        if results[0].boxes:
            no_detection_frames = 0  # 检测到对象，重置计数
            nearest_box = None
            min_distance = float("inf")

            # 只有当轨迹中有点时，才寻找最近的检测框
            if trajectory:
                last_point = trajectory[-1]
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取检测框坐标
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 计算中心点
                    distance = ((cx - last_point[0]) ** 2 + (cy - last_point[1]) ** 2) ** 0.5

                    # 更新最近的框
                    if distance < min_distance:
                        min_distance = distance
                        nearest_box = (cx, cy, x1, y1, x2, y2)
            else:
                # 如果轨迹为空，选择任意一个检测框
                box = results[0].boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                nearest_box = ((x1 + x2) // 2, (y1 + y2) // 2, x1, y1, x2, y2)

            # 如果找到最近的框，记录并绘制它
            if nearest_box:
                cx, cy, x1, y1, x2, y2 = nearest_box
                trajectory.append((cx, cy))  # 将中心点加入轨迹列表

                # 画出检测框
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            no_detection_frames += 1  # 增加没有检测的帧数

        # 如果超过10帧没有检测到，清除轨迹，开始新的轨迹
        if no_detection_frames > 10:
            trajectory.clear()  # 清除旧的轨迹
            no_detection_frames = 0  # 重置计数器

        # 画出轨迹
        for i in range(1, len(trajectory)):
            cv2.line(annotated_frame, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)  # 蓝色线条连接轨迹点

        # 显示带有轨迹的帧
        cv2.imshow("YOLO11 Tracking with Trajectory", annotated_frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果达到视频结尾，跳出循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()
