import mediapipe as mp
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


class PersonBlurrer:
    def __init__(self, yolo_model="yolov8n.pt", mosaic_size=8, blur_strength=55):
        # Initialize YOLO
        self.yolo_model = YOLO(yolo_model)

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Blur parameters
        self.mosaic_size = mosaic_size
        self.blur_strength = blur_strength

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

        padding = 20
        x1 = max(0, min(x_coords) - padding)
        y1 = max(0, min(y_coords) - padding)
        x2 = min(w, max(x_coords) + padding)
        y2 = min(h, max(y_coords) + padding)

        return (x1, y1, x2, y2)

    def process_image(self, img):
        """Process image with YOLO person detection and MediaPipe face blur"""
        # Get original image dimensions
        img_height, img_width = img.shape[:2]

        # Run YOLO detection
        results = self.yolo_model(img, verbose=False,conf=0.2)

        # Process each person detection
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check if detection is person (class 0)
                if int(box.cls) == 0:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    # Extract person ROI
                    person_roi = img[y1:y2, x1:x2]
                    if person_roi.size == 0:
                        continue

                    # Process with MediaPipe
                    person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose.process(person_rgb)

                    if pose_results.pose_landmarks:
                        # Get head bbox in ROI coordinates
                        head_bbox = self.get_head_bbox(pose_results.pose_landmarks.landmark, person_roi.shape)

                        if head_bbox:
                            # Convert head bbox to original image coordinates
                            roi_x1, roi_y1, roi_x2, roi_y2 = head_bbox
                            abs_x1 = x1 + roi_x1
                            abs_y1 = y1 + roi_y1
                            abs_x2 = x1 + roi_x2
                            abs_y2 = y1 + roi_y2

                            # Apply blur
                            img = self.apply_enhanced_blur(img, abs_x1, abs_y1, abs_x2, abs_y2)

        return img


def process_images(input_folder, output_folder, yolo_model="yolov8n.pt", mosaic_size=8, blur_strength=55):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Initialize processor
    processor = PersonBlurrer(yolo_model=yolo_model, mosaic_size=mosaic_size, blur_strength=blur_strength)

    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
        image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))

    print(f"Found {len(image_files)} images")

    # Process images
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            # Process image
            processed_img = processor.process_image(img)

            # Save result
            output_path = Path(output_folder) / img_path.name
            cv2.imwrite(str(output_path), processed_img)

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")


def main():
    input_folder = "C:\\workspace\\datasets\\coco-pp\\images\\train"
    output_folder = "C:\\workspace\\datasets\\coco-pp\\images\\train_no_face"

    process_images(
        input_folder=input_folder,
        output_folder=output_folder,
        yolo_model="yolov8n.pt",  # You can change to yolov8s.pt, yolov8m.pt, etc.
        mosaic_size=8,
        blur_strength=55
    )

    print("Processing complete!")


if __name__ == "__main__":
    main()