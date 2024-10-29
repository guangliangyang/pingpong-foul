import numpy as np
import cv2
import torch
import os
from ultralytics import YOLO
import mediapipe as mp

# Ensure the proper environment setting
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if torch.cuda.is_available():
    print("CUDA is available. GPU device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set parameters
confidence_threshold = 0.1
model_path = "C:/workspace/projects/pingpong-foul/model/best-new-transfer-yolov8m-freeze.pt"
video_path = "C:\\workspace\\datasets\\foul-video\\c2.mp4"

# Load YOLO model
model = YOLO(model_path, verbose=False)
model.to(device)
print("Using GPU:", model.device)

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Determine initial ROI using MediaPipe
def determine_roi(video_path, duration=1):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(duration * frame_rate)

    x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if idx in [11, 12, 13, 14, 15, 16]:  # Shoulders, elbows, wrists
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

    cap.release()
    pose.close()

    width = x_max - x_min
    height = y_max - y_min

    # Initial padding for width and height
    x_min = max(0, int(x_min - width))
    x_max = min(frame.shape[1], int(x_max + width))
    y_min = max(0, int(y_min - height))
    y_max = min(frame.shape[0], int(y_max + height))

    return [x_min, y_min, x_max, y_max]


# Dynamically update ROI based on player movements
def update_roi(frame, roi, pose_results):
    x_min, y_min, x_max, y_max = roi
    updated = False

    if pose_results.pose_landmarks:
        for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
            if idx in [11, 12, 13, 14, 15, 16]:  # Shoulders, elbows, wrists
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

                # Enlarge ROI if player moves outside current boundaries
                if x < x_min:
                    x_min = max(0, x - 20)  # Expand with 20 pixels buffer
                    updated = True
                if x > x_max:
                    x_max = min(frame.shape[1], x + 20)
                    updated = True
                if y < y_min:
                    y_min = max(0, y - 20)
                    updated = True
                if y > y_max:
                    y_max = min(frame.shape[0], y + 20)
                    updated = True

    if updated:
        print(f"ROI updated: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

    return [x_min, y_min, x_max, y_max]


# Extract trajectory with dynamic ROI updates
def extract_trajectory(video_path):
    cap = cv2.VideoCapture(video_path)
    trajectory = []
    last_valid_position = None
    frames_without_detection = 0

    # Initialize MediaPipe pose only once at the start of the function
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    roi = determine_roi(video_path)  # Initial ROI
    print(f"Initial ROI: x_min={roi[0]}, y_min={roi[1]}, x_max={roi[2]}, y_max={roi[3]}")

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {frame_number}. Exiting.")
            break  # Exit the loop if no more frames are read

        frame_number += 1

        # Ensure frame is correctly read and converted to RGB
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error converting frame to RGB at frame {frame_number}: {e}")
            continue

        # Detect player's pose in the current frame if pose object is valid
        if pose:
            results = pose.process(frame_rgb)
            if results.pose_landmarks is None:
                print(f"Warning: No landmarks detected at frame {frame_number}. Skipping frame.")
                continue  # Skip processing this frame if no landmarks are detected
        else:
            print("Error: MediaPipe Pose object was not initialized correctly.")
            break

        # Update ROI dynamically
        roi = update_roi(frame, roi, results)
        x, y, w, h = roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]

        # Extract the updated ROI area for further analysis
        roi_frame = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        # Background subtraction for detecting movement
        fg_mask = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False).apply(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Detect changes within ROI and update trajectory
        non_zero_count = np.count_nonzero(fg_mask)
        if non_zero_count < 50:
            frames_without_detection += 1
            if frames_without_detection >= 10:
                last_valid_position = None
                frames_without_detection = 0
            continue

        results = model(roi_frame)
        yolo_detected = False
        possible_positions = []

        for result in results:
            for detection in result.boxes:
                if detection.conf > confidence_threshold and int(detection.cls) == 3:
                    x_center = x + (detection.xyxy[0][0] + detection.xyxy[0][2]) / 2
                    y_center = y + (detection.xyxy[0][1] + detection.xyxy[0][3]) / 2
                    current_position = (x_center.item(), y_center.item())
                    possible_positions.append(current_position)

        if possible_positions:
            if last_valid_position is not None:
                possible_positions.sort(key=lambda pos: np.linalg.norm(np.array(last_valid_position) - np.array(pos)))

            best_position = possible_positions[0]
            if x <= best_position[0] <= x + w and y <= best_position[1] <= y + h:
                if last_valid_position is None or np.linalg.norm(
                        np.array(last_valid_position) - np.array(best_position)) <= 30:
                    trajectory.append((frame_number, best_position))
                    last_valid_position = best_position
                    frames_without_detection = 0
                    cv2.circle(frame, (int(best_position[0]), int(best_position[1])), 5, (0, 255, 0), 2)
                    yolo_detected = True
        else:
            last_valid_position = None
            frames_without_detection = 0

        if not yolo_detected:
            frames_without_detection += 1
            if frames_without_detection >= 10:
                last_valid_position = None
                frames_without_detection = 0
                print("超过10帧未检测到乒乓球，重置检测点")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for i in range(1, len(trajectory)):
            pt1 = (int(trajectory[i - 1][1][0]), int(trajectory[i - 1][1][1]))
            pt2 = (int(trajectory[i][1][0]), int(trajectory[i][1][1]))
            if np.linalg.norm(np.array(trajectory[i - 1][1]) - np.array(trajectory[i][1])) <= 40:
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if pose:
        pose.close()  # Ensure pose resources are properly released
    cv2.destroyAllWindows()
    return np.array(trajectory)


# Main function
trajectory = extract_trajectory(video_path)
