import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Video path
video_path = "C:\\workspace\\datasets\\foul-video\\c1.mp4"
cap = cv2.VideoCapture(video_path)

# To store angles and variance
angle1_changes = []
angle2_changes = []
prev_angle1, prev_angle2 = None, None

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB as MediaPipe expects RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates of required points
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        right_elbow = np.array(
            [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
        right_wrist = np.array(
            [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])

        # Calculate vectors for lines
        line1 = left_shoulder - right_shoulder
        line2 = right_elbow - right_shoulder
        line3 = right_wrist - right_elbow

        # Calculate angles in degrees
        angle1 = np.degrees(np.arccos(np.dot(line1, line2) / (np.linalg.norm(line1) * np.linalg.norm(line2))))
        angle2 = np.degrees(np.arccos(np.dot(line2, line3) / (np.linalg.norm(line2) * np.linalg.norm(line3))))

        # Calculate change in angle and variance if not the first frame
        if prev_angle1 is not None and prev_angle2 is not None:
            angle1_change = angle1 - prev_angle1
            angle2_change = angle2 - prev_angle2
            angle1_changes.append(angle1_change)
            angle2_changes.append(angle2_change)

        # Update previous angles
        prev_angle1, prev_angle2 = angle1, angle2

# Release resources
cap.release()
pose.close()

# Calculate variances
variance_angle1 = np.var(angle1_changes)
variance_angle2 = np.var(angle2_changes)

# Plot angle changes
plt.figure(figsize=(10, 5))
plt.plot(angle1_changes, label="Angle 1 Change (Variance: {:.2f})".format(variance_angle1))
plt.plot(angle2_changes, label="Angle 2 Change (Variance: {:.2f})".format(variance_angle2))
plt.xlabel("Frame")
plt.ylabel("Angle Change (degrees)")
plt.legend()
plt.title("Angle Changes Over Time")
plt.show()
