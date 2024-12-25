from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model

# Predict with the model
results = model("C:\\workspace\\datasets\\coco-pp\\images\\test\\01.jpg")  # predict on an image
