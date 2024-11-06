import cv2
import torch
import numpy as np
from ultralytics import YOLO
import math
import pygame

# Initialize pygame
pygame.init()
screen_width, screen_height = 1280, 720
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("YOLO Tracking with Key Points Display")

# Define dimensions for each section
video_width, video_height = screen_width // 2, screen_height // 2

# Model and video paths
model_path = "C:/workspace/projects/pingpong-foul/model/best-yolo11-transfer.pt"
video_paths = [
    "C:\\workspace\\datasets\\foul-video\\c1.mp4"
]

# Load YOLO11 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path, verbose=False)
model.to(device)
print("Using GPU:", model.device)

# Create empty images for key frames
empty_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)


def find_key_points(trajectory, x_acceleration_threshold=5.0):
    throw_point, highest_point, hit_point = None, None, None
    throw_frame, highest_frame, hit_frame = None, None, None

    # Step 1: Initial detection of throw point based on Y speed
    for i in range(1, len(trajectory)):
        frame_no, prev_x, prev_y = trajectory[i - 1]
        curr_frame_no, curr_x, curr_y = trajectory[i]

        # Calculate the Y decrease speed
        y_speed = abs(curr_y - prev_y)

        if i > 1:
            _, _, last_y = trajectory[i - 2]
            last_y_speed = abs(prev_y - last_y)

            if y_speed > 5 * last_y_speed:
                throw_point = (curr_x, curr_y)
                throw_frame = curr_frame_no
                break

    # Step 2: Find the highest point after the throw point
    if throw_point:
        throw_index = trajectory.index((throw_frame, *throw_point))
        highest_y = min([point[2] for point in trajectory[throw_index:]])

        # Find the (frame_no, x, y) tuple with the highest point
        for frame_no, x, y in trajectory[throw_index:]:
            if y == highest_y:
                highest_point = (x, y)
                highest_frame = frame_no
                break

    # Step 3: Backtrack from the highest point to optimize the throw point
    if highest_point:
        highest_index = trajectory.index((highest_frame, *highest_point))
        for i in range(highest_index, -1, -1):
            frame_no, x, y = trajectory[i]
            if i < highest_index and y < trajectory[i + 1][2]:  # Y-axis decreases
                throw_point = (x, y)
                throw_frame = frame_no
                break

    # Step 4: Identify the hit point based on X acceleration
    if highest_point:
        for i in range(highest_index + 2, len(trajectory)):
            prev_frame_no, prev_x, prev_y = trajectory[i - 2]
            last_frame_no, last_x, last_y = trajectory[i - 1]
            curr_frame_no, curr_x, curr_y = trajectory[i]

            # Calculate X speed and acceleration
            x_speed_prev = last_x - prev_x
            x_speed_curr = curr_x - last_x
            x_acceleration = x_speed_curr - x_speed_prev

            # Calculate line distance (Euclidean distance) between last two points
            line_distance = math.sqrt((curr_x - last_x) ** 2 + (curr_y - last_y) ** 2)

            # Apply both the x_acceleration and line distance filters
            if x_speed_curr > 0 and x_acceleration > x_acceleration_threshold and line_distance < 120:
                hit_point = (last_x, last_y)
                hit_frame = last_frame_no
                break



    return (throw_point, throw_frame), (highest_point, highest_frame), (hit_point, hit_frame)


# Video processing function
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    trajectory = []
    frame_count = 0
    no_detection_frames = 0
    paused = False

    throw_frame, highest_frame, hit_frame = empty_frame, empty_frame, empty_frame

    while cap.isOpened():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    cap.release()
                    pygame.quit()
                    return
                elif event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            annotated_frame = frame.copy()

            if results[0].boxes:
                no_detection_frames = 0
                nearest_box = None
                min_distance = float("inf")
                if trajectory:
                    last_frame, last_x, last_y = trajectory[-1]
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        distance = math.sqrt((cx - last_x) ** 2 + (cy - last_y) ** 2)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_box = (frame_count, cx, cy, x1, y1, x2, y2)
                else:
                    box = results[0].boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    nearest_box = (frame_count, (x1 + x2) // 2, (y1 + y2) // 2, x1, y1, x2, y2)

                if nearest_box:
                    frame_no, cx, cy, x1, y1, x2, y2 = nearest_box
                    trajectory.append((frame_no, cx, cy))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                no_detection_frames += 1

            if no_detection_frames > 10:
                trajectory.clear()
                no_detection_frames = 0
                throw_frame, highest_frame, hit_frame = empty_frame, empty_frame, empty_frame  # Reset key frames

            # Draw the trajectory on the annotated frame
            for i in range(1, len(trajectory)):
                _, prev_x, prev_y = trajectory[i - 1]
                _, curr_x, curr_y = trajectory[i]
                cv2.line(annotated_frame, (prev_x, prev_y), (curr_x, curr_y), (255, 0, 0), 2)
                cv2.circle(annotated_frame, (curr_x, curr_y), 3, (0, 255, 255), -1)

            # Check for key points and capture frames if needed
            if len(trajectory) > 10:
                (throw_point, throw_frame_no), (highest_point, highest_frame_no), (
                hit_point, hit_frame_no) = find_key_points(trajectory)

                if throw_point:
                    throw_index = trajectory.index((throw_frame_no, *throw_point))

                    # Draw trajectory starting from the throw point
                    for i in range(throw_index + 1, len(trajectory)):
                        _, prev_x, prev_y = trajectory[i - 1]
                        _, curr_x, curr_y = trajectory[i]
                        cv2.line(annotated_frame, (prev_x, prev_y), (curr_x, curr_y), (255, 0, 0), 2)
                        cv2.circle(annotated_frame, (curr_x, curr_y), 3, (0, 255, 255), -1)

                    # Draw key points
                    cv2.rectangle(annotated_frame,
                                  (throw_point[0] - 5, throw_point[1] - 5),
                                  (throw_point[0] + 5, throw_point[1] + 5),
                                  (0, 255, 255), -1)  # Yellow square for throw point

                    if highest_point:
                        triangle_points = np.array([
                            [highest_point[0], highest_point[1] - 6],  # Top vertex
                            [highest_point[0] - 5, highest_point[1] + 5],  # Bottom left vertex
                            [highest_point[0] + 5, highest_point[1] + 5]  # Bottom right vertex
                        ], np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(annotated_frame, [triangle_points], color=(0, 0, 255))  # Red triangle

                    if hit_point:
                        cv2.circle(annotated_frame, hit_point, 5, (0, 255, 0), -1)  # Green circle for hit point

                # Capture frames at key points using frame numbers
                if throw_frame_no:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, throw_frame_no)
                    ret, throw_frame = cap.read()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)  # Return to current frame

                if highest_frame_no:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, highest_frame_no)
                    ret, highest_frame = cap.read()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

                if hit_frame_no:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, hit_frame_no)
                    ret, hit_frame = cap.read()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            # Convert frame to RGB format for pygame and resize
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            screen.blit(pygame.transform.scale(frame_surface, (video_width, video_height)), (0, 0))  # Top-left

            # Display the throw, highest, and hit frames in the corresponding quadrants
            throw_surface = pygame.surfarray.make_surface(np.rot90(cv2.cvtColor(throw_frame, cv2.COLOR_BGR2RGB)))
            screen.blit(pygame.transform.scale(throw_surface, (video_width, video_height)),
                        (video_width, 0))  # Top-right

            highest_surface = pygame.surfarray.make_surface(np.rot90(cv2.cvtColor(highest_frame, cv2.COLOR_BGR2RGB)))
            screen.blit(pygame.transform.scale(highest_surface, (video_width, video_height)),
                        (0, video_height))  # Bottom-left

            hit_surface = pygame.surfarray.make_surface(np.rot90(cv2.cvtColor(hit_frame, cv2.COLOR_BGR2RGB)))
            screen.blit(pygame.transform.scale(hit_surface, (video_width, video_height)),
                        (video_width, video_height))  # Bottom-right

            pygame.display.flip()

        pygame.time.wait(10)

    cap.release()
    pygame.quit()


# Run video processing for each video path
for video_path in video_paths:
    print(f"Processing video: {video_path}")
    process_video(video_path)
