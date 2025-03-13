import cv2
import os

# Initialize global variables
roi_points = []
selected_roi = None

# Mouse callback function
def select_roi(event, x, y, flags, param):
    global roi_points, selected_roi

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points = [(x, y)]  # Store starting point
    elif event == cv2.EVENT_LBUTTONUP:
        roi_points.append((x, y))  # Store end point
        x0, y0 = roi_points[0]
        x1, y1 = roi_points[1]
        selected_roi = (x0, y0, x1 - x0, y1 - y0)  # Define ROI
        print(f"Selected ROI: {selected_roi}")
        roi_points = []

# Load video and get properties
input_video = 'C:\\workspace\\datasets\\foul-voideo\\foul-01.mp4'
cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Generate output file name
input_filename = os.path.splitext(os.path.basename(input_video))[0]
output_video = f"{input_filename}_roi.mp4"

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error reading video.")
    cap.release()
    exit()

# Set up window and mouse callback for ROI selection
cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", select_roi)

# Select a single ROI
print("Please select one region of interest by clicking and dragging.")
while True:
    temp_frame = frame.copy()
    if selected_roi:
        x, y, w, h = selected_roi
        cv2.rectangle(temp_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Select ROI", temp_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or selected_roi:
        break

cv2.destroyWindow("Select ROI")

# Ensure an ROI is selected
if not selected_roi:
    print("No ROI selected.")
    cap.release()
    exit()

# Initialize video writer for the selected ROI
x, y, w, h = selected_roi
out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the frame to the selected ROI
    cropped_frame = frame[y:y + h, x:x + w]
    out.write(cropped_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video extraction complete! Saved as {output_video}.")
