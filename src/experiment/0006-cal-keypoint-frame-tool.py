import cv2
import torch
import numpy as np
from ultralytics import YOLO
import math
import pygame
import time

# Initialize pygame
pygame.init()
screen_width, screen_height = 960, 720
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("YOLO Tracking with Frame Index Display")

# Model and video paths
model_path = "C:/workspace/projects/pingpong-foul/model/best-yolo11-transfer03.pt"
video_paths = [
    "C:\\workspace\\datasets\\foul-video\\c1.mp4"
]

# Load YOLO11 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path, verbose=False)
model.to(device)
print("Using GPU:", model.device)

# Initialize font for frame index display
font = pygame.font.Font(None, 36)  # Use None for default font, 36 for size

# Get the user input for starting frame index
start_frame_index = int(input("Enter the starting frame index: "))
cycle_frame_count = 20  # Number of frames to cycle through

# Function to render frame index on Pygame surface
def render_frame_index(surface, text, position=(10, 10)):
    text_surface = font.render(text, True, (0, 0, 0))  # Black text
    surface.blit(text_surface, position)

# Video processing function
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = start_frame_index  # Start from the user input frame
    paused = False

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)  # Set to starting frame

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
            # Cycle play through frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            if frame_count >= start_frame_index + cycle_frame_count:
                frame_count = start_frame_index  # Reset to start frame to loop

            # Perform YOLO tracking
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            annotated_frame = frame.copy()

            # Annotate the frame with current frame index
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Convert frame to RGB format for pygame and resize to full screen
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            frame_surface = pygame.transform.scale(frame_surface, (screen_width, screen_height))

            # Render the frame index on main frame surface
            render_frame_index(frame_surface, f"Frame: {frame_count}")

            # Display the main video frame in full screen
            screen.blit(frame_surface, (0, 0))  # Top-left (full screen)

            pygame.display.flip()
            time.sleep(0.5)  # Pause for 0.5 seconds between frames

        pygame.time.wait(10)

    cap.release()
    pygame.quit()

# Run video processing for each video path
for video_path in video_paths:
    print(f"Processing video: {video_path}")
    process_video(video_path)
