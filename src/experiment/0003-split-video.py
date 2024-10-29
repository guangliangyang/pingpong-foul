import cv2
import pygame
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.signal import find_peaks
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Pygame setup
pygame.init()
video_path = "C:\\workspace\\datasets\\foul-video\\c1.mp4"
cap = cv2.VideoCapture(video_path)

# Window dimensions
video_width, video_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
plot_width, plot_height = 500, 400
screen = pygame.display.set_mode((video_width + plot_width, max(video_height, plot_height)))
pygame.display.set_caption("Pose Tracking and Right Wrist Position Change Plot with Kalman Filter")

# Kalman filter setup
kalman = cv2.KalmanFilter(2, 1)
kalman.measurementMatrix = np.array([[1, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0], [0, 1e-3]], np.float32)
kalman.measurementNoiseCov = np.array([[1e-1]], np.float32)

# Parameters for peak detection
peak_threshold = 8
min_frame_distance = 40

# To store positional changes of the right wrist
wrist_changes, filtered_changes = [], []
prev_wrist_position = None

# Initialize Matplotlib figure for plotting
fig, ax = plt.subplots(figsize=(plot_width / 100, plot_height / 100))
canvas = FigureCanvas(fig)

# Prepare directory for saving split video segments
output_dir = "C:\\workspace\\datasets\\foul-video\\split"
os.makedirs(output_dir, exist_ok=True)

# Variables for segment splitting
start_frame = 0
segment_index = 1
qualified_peaks = []

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

    # Calculate wrist position change if landmarks are detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * video_width,
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * video_height])

        # Draw right wrist position on the frame
        cv2.circle(frame, tuple(right_wrist.astype(int)), 5, (0, 255, 0), -1)

        # Calculate positional change with direction
        if prev_wrist_position is not None:
            delta_position = right_wrist - prev_wrist_position
            wrist_change_magnitude = np.linalg.norm(delta_position)
            wrist_change = wrist_change_magnitude if delta_position[1] >= 0 else -wrist_change_magnitude
            wrist_changes.append(wrist_change)

            # Apply Kalman filter to the wrist change
            kalman.correct(np.array([[np.float32(wrist_change)]]))
            filtered_change = kalman.predict()[0][0]
            filtered_changes.append(filtered_change)

        prev_wrist_position = right_wrist

    # Detect peaks in the filtered_changes
    if len(filtered_changes) > 0:
        peak_indices, _ = find_peaks(np.abs(filtered_changes), height=peak_threshold)
        qualified_peaks = [peak_indices[0]] if len(peak_indices) > 0 else []
        for i in range(1, len(peak_indices)):
            current_peak = peak_indices[i]
            previous_peak = qualified_peaks[-1]
            if current_peak - previous_peak > min_frame_distance:
                qualified_peaks.append(current_peak)

    # Update plot with the latest data
    ax.clear()
    ax.plot(filtered_changes, label="Filtered Wrist Position Change (Signed)")
    ax.axhline(y=0, color='gray', linestyle='--')
    for peak in qualified_peaks:
        ax.axvline(x=peak, color='red', linestyle='--')
    for i in range(1, len(qualified_peaks)):
        midpoint = (qualified_peaks[i - 1] + qualified_peaks[i]) // 2
        ax.axvline(x=midpoint, color='green', linestyle='--')
    ax.set_ylim(-100, 100)
    ax.set_title("Kalman Filtered Right Wrist Position Change with Direction (All Frames)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Position Change (pixels)")
    ax.legend()
    canvas.draw()

    # Convert Matplotlib plot to Pygame surface
    plot_surface = pygame.image.fromstring(canvas.tostring_rgb(), canvas.get_width_height(), "RGB")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(np.rot90(frame))

    screen.blit(frame_surface, (0, 0))
    screen.blit(plot_surface, (video_width, 0))
    pygame.display.update()

# Splitting video based on qualified peaks' midpoints (green lines)
cap.release()
cap = cv2.VideoCapture(video_path)  # Reset video capture to beginning

# Process each segment based on green line midpoints
for i in range(1, len(qualified_peaks)):
    end_frame = (qualified_peaks[i - 1] + qualified_peaks[i]) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out_path = os.path.join(output_dir, f"segment_{segment_index}.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video_width, video_height))

    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, segment_frame = cap.read()
        if not ret:
            break
        out.write(segment_frame)

    out.release()
    start_frame = end_frame
    segment_index += 1

# Save last segment if there are remaining frames
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
out_path = os.path.join(output_dir, f"segment_{segment_index}.mp4")
out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video_width, video_height))

while cap.isOpened():
    ret, segment_frame = cap.read()
    if not ret:
        break
    out.write(segment_frame)

out.release()
cap.release()
pose.close()
pygame.quit()
