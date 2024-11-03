import json
import cv2
import numpy as np
import pygame
import sys
import time


class TableTennisGame:
    def __init__(self):
        # Initialize and load calibration data
        self.load_calibration_data()
        self.click_points = {"camera1": None, "camera2": None}

        # Paths for dedicated video sources
        self.video_paths = {
            'camera1': 'C:\\workspace\\datasets\\foul-video\\c1.mp4',
            'camera2': 'C:\\workspace\\datasets\\foul-video\\c2.mp4'
        }

        # Initialize video captures for both cameras
        self.caps = {
            "camera1": cv2.VideoCapture(self.video_paths['camera1']),
            "camera2": cv2.VideoCapture(self.video_paths['camera2'])
        }

    def load_calibration_data(self):
        # Load camera calibration data from JSON
        with open('0012-calibration_in_ex_trinsic.json', 'r') as f:
            calibration_data = json.load(f)

        # Extract parameters for camera1
        self.camera1_intrinsics = np.array(calibration_data['camera1']['intrinsics']['camera_matrix'])
        self.camera1_dist_coeffs = np.array(calibration_data['camera1']['intrinsics']['dist_coeffs'])
        self.camera1_rot_vec = np.array(calibration_data['camera1']['extrinsics']['rotation_vector'])
        self.camera1_trans_vec = np.array(calibration_data['camera1']['extrinsics']['translation_vector'])

        # Extract parameters for camera2
        self.camera2_intrinsics = np.array(calibration_data['camera2']['intrinsics']['camera_matrix'])
        self.camera2_dist_coeffs = np.array(calibration_data['camera2']['intrinsics']['dist_coeffs'])
        self.camera2_rot_vec = np.array(calibration_data['camera2']['extrinsics']['rotation_vector'])
        self.camera2_trans_vec = np.array(calibration_data['camera2']['extrinsics']['translation_vector'])

        # Convert rotation vectors to rotation matrices
        self.camera1_rot_matrix, _ = cv2.Rodrigues(self.camera1_rot_vec)
        self.camera2_rot_matrix, _ = cv2.Rodrigues(self.camera2_rot_vec)

        # Define projection matrices for both cameras
        self.proj_matrix1 = np.dot(self.camera1_intrinsics,
                                   np.hstack((self.camera1_rot_matrix, self.camera1_trans_vec)))
        self.proj_matrix2 = np.dot(self.camera2_intrinsics,
                                   np.hstack((self.camera2_rot_matrix, self.camera2_trans_vec)))

    def calculate_3d_coordinates(self, point1, point2):
        # Calculate 3D coordinates using triangulation
        points_4d_homogeneous = cv2.triangulatePoints(self.proj_matrix1, self.proj_matrix2, point1, point2)
        points_3d = points_4d_homogeneous[:3] / points_4d_homogeneous[3]
        return points_3d.flatten()

    def on_user_click(self, camera, x, y):
        # Store the clicked point for each camera
        self.click_points[camera] = np.array([[x], [y]], dtype=np.float32)

        # If both points are clicked, calculate the 3D coordinate
        if self.click_points['camera1'] is not None and self.click_points['camera2'] is not None:
            point1 = self.click_points['camera1']
            point2 = self.click_points['camera2']
            coordinates_3d = self.calculate_3d_coordinates(point1, point2)
            print(f"3D Coordinates: {coordinates_3d}")

            # Reset points after calculation
            self.click_points['camera1'] = None
            self.click_points['camera2'] = None

    def read_frame(self, camera):
        # Read the next frame from the specified camera
        ret, frame = self.caps[camera].read()
        if not ret:
            print(f"End of video for {camera}. Restarting video.")
            self.caps[camera].set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            ret, frame = self.caps[camera].read()
        return frame


def main():
    pygame.init()
    screen_width, screen_height = 800, 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Table Tennis 3D Coordinate Calculation")
    font = pygame.font.SysFont("Arial", 24)

    game = TableTennisGame()
    selected_camera = "camera1"

    instructions = [
        "Click to select a point for camera1 and camera2 in sequence.",
        "Left side (0-400px) is camera1, right side (400-800px) is camera2."
    ]

    running = True
    while running:
        screen.fill((255, 255, 255))

        # Draw dividing line
        pygame.draw.line(screen, (0, 0, 0), (400, 0), (400, 400), 2)

        # Display instructions
        for i, text in enumerate(instructions):
            text_surface = font.render(text, True, (0, 0, 0))
            screen.blit(text_surface, (10, 10 + i * 30))

        # Read frames from both cameras
        frame1 = game.read_frame("camera1")
        frame2 = game.read_frame("camera2")

        # Convert frames to a format suitable for Pygame display
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # Resize frames to fit the Pygame display
        frame1_surface = pygame.surfarray.make_surface(cv2.resize(frame1_rgb, (400, 400)).swapaxes(0, 1))
        frame2_surface = pygame.surfarray.make_surface(cv2.resize(frame2_rgb, (400, 400)).swapaxes(0, 1))

        # Display frames on the Pygame window
        screen.blit(frame1_surface, (0, 0))
        screen.blit(frame2_surface, (400, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if x < 400:
                    selected_camera = "camera1"
                else:
                    selected_camera = "camera2"
                    x -= 400  # Adjust x-coordinate for camera2 to match screen coordinates

                print(f"Clicked on {selected_camera} at ({x}, {y})")
                game.on_user_click(selected_camera, x, y)

        pygame.display.flip()
        pygame.time.delay(30)  # Control frame update speed

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
