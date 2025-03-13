import cv2
import pygame
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.signal import find_peaks

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
kalman = cv2.KalmanFilter(2, 1)  # 2 state variables, 1 measurement
kalman.measurementMatrix = np.array([[1, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0], [0, 1e-3]], np.float32)  # Adjust noise values as needed
kalman.measurementNoiseCov = np.array([[1e-1]], np.float32)

# Parameters for peak detection
peak_threshold = 8  # Adjust based on your data
min_frame_distance = 40  # Minimum frame distance between two peaks

# To store positional changes of the right wrist
wrist_changes, filtered_changes = [], []
prev_wrist_position = None

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

    # Calculate wrist position change if landmarks are detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # Get the right wrist position
        right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * video_width,
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * video_height])

        # Draw right wrist position on the frame
        cv2.circle(frame, tuple(right_wrist.astype(int)), 5, (0, 255, 0), -1)  # Green dot for right wrist

        # Calculate positional change with direction
        if prev_wrist_position is not None:
            # Calculate directional change
            delta_position = right_wrist - prev_wrist_position
            wrist_change_magnitude = np.linalg.norm(delta_position)

            # Assign a sign to wrist_change based on y-axis movement
            wrist_change = wrist_change_magnitude if delta_position[1] >= 0 else -wrist_change_magnitude
            wrist_changes.append(wrist_change)

            # Apply Kalman filter to the wrist change
            kalman.correct(np.array([[np.float32(wrist_change)]]))
            filtered_change = kalman.predict()[0][0]  # Predicted value after filtering
            filtered_changes.append(filtered_change)

        # Update previous wrist position
        prev_wrist_position = right_wrist

    # Show all frames in the plot
    filtered_display = filtered_changes

    # Initialize qualified_peaks to avoid NameError
    qualified_peaks = []

    # Detect peaks in the filtered_changes
    if len(filtered_changes) > 0:
        peak_indices, _ = find_peaks(np.abs(filtered_changes), height=peak_threshold)

        # Filter peaks based on minimum frame distance
        qualified_peaks = [peak_indices[0]] if len(peak_indices) > 0 else []  # Start with the first peak if it exists
        for i in range(1, len(peak_indices)):
            current_peak = peak_indices[i]
            previous_peak = qualified_peaks[-1]

            # Only add the current peak if it's more than `min_frame_distance` frames from the previous peak
            if current_peak - previous_peak > min_frame_distance:
                qualified_peaks.append(current_peak)

    # Update plot with the latest data
    ax.clear()
    ax.plot(filtered_display, label="Filtered Wrist Position Change (Signed)")
    ax.axhline(y=0, color='gray', linestyle='--')  # Add horizontal line at y=0

    # Draw vertical lines at qualified peaks
    for peak in qualified_peaks:
        ax.axvline(x=peak, color='red', linestyle='--')  # Use the actual peak index in `filtered_display`

    ax.set_ylim(-100, 100)  # Set y-axis range for positional changes
    ax.set_title("Kalman Filtered Right Wrist Position Change with Direction (All Frames)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Position Change (pixels)")
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
