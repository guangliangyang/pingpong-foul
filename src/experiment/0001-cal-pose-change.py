import cv2
import pygame
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Pygame setup
pygame.init()
video_path = "C:\\workspace\\datasets\\foul-video\\01-5.mp4"
video_path = "C:\\workspace\\datasets\\foul-video\\01.mov"
video_path = "C:\\workspace\\datasets\\foul-video\\test-pp.mp4"
video_path = "C:\\workspace\\datasets\\foul-video\\c1.mp4"
cap = cv2.VideoCapture(video_path)

# Window dimensions
video_width, video_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
plot_width, plot_height = 500, 400
screen = pygame.display.set_mode((video_width + plot_width, max(video_height, plot_height)))
pygame.display.set_caption("Pose Tracking and Angle Plot")

# To store angles
angle1_changes, angle2_changes = [], []
prev_angle1, prev_angle2 = None, None

# Limit the plot to the latest 300 frames
MAX_FRAMES_DISPLAY = 300

# Initialize Matplotlib figure for plotting
fig, ax = plt.subplots(figsize=(plot_width / 100, plot_height / 100))
canvas = FigureCanvas(fig)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Calculate angles if landmarks are detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # Get coordinates of key points
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * video_width,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * video_height])
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * video_width,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * video_height])
        right_elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * video_width,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * video_height])
        right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * video_width,
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * video_height])

        # Draw lines on the frame
        cv2.line(frame, tuple(right_shoulder.astype(int)), tuple(left_shoulder.astype(int)), (255, 0, 0),
                 2)  # Line 1: Right Shoulder to Left Shoulder
        cv2.line(frame, tuple(right_shoulder.astype(int)), tuple(right_elbow.astype(int)), (0, 255, 0),
                 2)  # Line 2: Right Shoulder to Right Elbow
        cv2.line(frame, tuple(right_elbow.astype(int)), tuple(right_wrist.astype(int)), (0, 0, 255),
                 2)  # Line 3: Right Elbow to Right Wrist

        # Calculate vectors and angles
        line1 = left_shoulder - right_shoulder
        line2 = right_elbow - right_shoulder
        line3 = right_wrist - right_elbow

        angle1 = np.degrees(np.arccos(np.dot(line1, line2) / (np.linalg.norm(line1) * np.linalg.norm(line2))))
        angle2 = np.degrees(np.arccos(np.dot(line2, line3) / (np.linalg.norm(line2) * np.linalg.norm(line3))))

        # Calculate angle changes and update lists
        if prev_angle1 is not None and prev_angle2 is not None:
            angle1_change = angle1 - prev_angle1
            angle2_change = angle2 - prev_angle2
            angle1_changes.append(angle1_change)
            angle2_changes.append(angle2_change)

        prev_angle1, prev_angle2 = angle1, angle2

    # Only plot the latest 300 frames for angle1_changes and angle2_changes
    angle1_display = angle1_changes[-MAX_FRAMES_DISPLAY:]
    angle2_display = angle2_changes[-MAX_FRAMES_DISPLAY:]

    # Update plot with the latest data
    ax.clear()
    ax.plot(angle1_display, label="Angle 1 Change")
    ax.plot(angle2_display, label="Angle 2 Change")
    ax.set_ylim(-100, 100)  # Set y-axis range
    ax.set_title("Real-time Angle Changes (Last 300 Frames)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle Change (degrees)")
    ax.legend()
    canvas.draw()

    # Convert Matplotlib plot to Pygame surface
    plot_surface = pygame.image.fromstring(canvas.tostring_rgb(), canvas.get_width_height(), "RGB")

    # Convert frame to a format compatible with Pygame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(np.rot90(frame))


    # Display video frame and plot side by side
    screen.blit(frame_surface, (0, 0))
    screen.blit(plot_surface, (video_width, 0))
    pygame.display.update()

# Release resources
cap.release()
pose.close()
pygame.quit()
