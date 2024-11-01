import cv2
import torch
from ultralytics import YOLO

# 模型和视频路径
model_path = "C:/workspace/projects/pingpong-foul/model/best-yolo11-transfer.pt"
video_path = "C:\\workspace\\datasets\\foul-video\\c1.mp4"

# 加载YOLO11模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path, verbose=False)
model.to(device)
print("Using GPU:", model.device)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 初始化一个列表，用于存储检测到的对象的中心点位置
trajectory = []

# 主循环，遍历视频帧
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # 运行YOLO11跟踪并获取检测结果
        results = model.track(frame, persist=True)

        # 复制当前帧，用于绘制轨迹
        annotated_frame = frame.copy()

        # 获取检测框的中心点并添加到轨迹
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取检测框坐标
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 计算中心点
            trajectory.append((cx, cy))  # 将中心点加入轨迹列表

            # 画出检测框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
