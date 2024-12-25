import mediapipe as mp
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm


class PoseDetector:
    def __init__(self, mode=False, complexity=1, smooth=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 mosaic_size=8, blur_strength=55):  # Reduced mosaic size, added blur_strength
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mosaic_size = mosaic_size
        self.blur_strength = blur_strength

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.complexity,
            smooth_landmarks=self.smooth,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def apply_enhanced_blur(self, image, x1, y1, x2, y2):
        """Apply multiple blur effects for stronger anonymization"""
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return image

        h, w = roi.shape[:2]
        if h == 0 or w == 0:
            return image

        # First apply mosaic effect
        temp = cv2.resize(roi, (self.mosaic_size, self.mosaic_size), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

        # Then apply Gaussian blur
        blurred = cv2.GaussianBlur(mosaic, (self.blur_strength, self.blur_strength), 0)

        # Additional pixel diffusion
        blurred = cv2.GaussianBlur(blurred, (5, 5), 0)

        # Blend the mosaic and blur effects
        result = cv2.addWeighted(mosaic, 0.5, blurred, 0.5, 0)

        image[y1:y2, x1:x2] = result
        return image

    def get_head_bbox(self, landmarks, img_shape):
        head_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        h, w = img_shape[:2]
        x_coords = []
        y_coords = []

        for idx in head_landmarks:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                cx = int(landmark.x * w)
                cy = int(landmark.y * h)
                x_coords.append(cx)
                y_coords.append(cy)

        if not x_coords or not y_coords:
            return None

        # Increased padding for better coverage
        padding = 20  # Increased from 30
        x1 = max(0, min(x_coords) - padding)
        y1 = max(0, min(y_coords) - padding)
        x2 = min(w, max(x_coords) + padding)
        y2 = min(h, max(y_coords) + padding)

        return (x1, y1, x2, y2)

    def find_poses(self, img, draw=False):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        poses = []

        if self.results.pose_landmarks:
            for pose_landmarks in [self.results.pose_landmarks]:
                bbox = self.get_head_bbox(pose_landmarks.landmark, img.shape)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    img = self.apply_enhanced_blur(img, x1, y1, x2, y2)

                if draw:
                    connections = [conn for conn in self.mp_pose.POSE_CONNECTIONS
                                   if not (conn[0] <= 10 and conn[1] <= 10)]
                    self.mp_draw.draw_landmarks(
                        img, pose_landmarks, connections)

                pose_points = []
                for landmark in pose_landmarks.landmark:
                    h, w, _ = img.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    pose_points.append([cx, cy])
                poses.append(pose_points)

        return img, poses


def process_images(input_folder, output_folder, mosaic_size=8, blur_strength=55):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    detector = PoseDetector(mosaic_size=mosaic_size, blur_strength=blur_strength)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
        image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))

    print(f"Found {len(image_files)} images")

    for img_path in tqdm(image_files, desc="Processing images"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        processed_img, _ = detector.find_poses(img)

        output_path = Path(output_folder) / img_path.name
        cv2.imwrite(str(output_path), processed_img)


def main():
    input_folder = "C:\\workspace\\datasets\\coco-pp\\images\\val"
    output_folder = "C:\\workspace\\datasets\\coco-pp\\images\\val_no_face"

    # Enhanced blur settings
    process_images(
        input_folder=input_folder,
        output_folder=output_folder,
        mosaic_size=8,  # Smaller size for stronger pixelation
        blur_strength=55  # Increased blur strength
    )

    print("Processing complete!")


if __name__ == "__main__":
    main()