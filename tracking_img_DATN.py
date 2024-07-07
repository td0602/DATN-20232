import cv2
import numpy as np
import func
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict

# Load the YOLOv8 model
model = YOLO("yolov8l.pt")
# model.to('cuda')

# Store the track history
track_history = defaultdict(lambda: [])
_vehicles = [2, 3, 5, 7]
de_scale = 1

# Load image
frame = cv2.imread("C:\\Users\\Tran Doan\\OneDrive\\DATN\\test_images\\XA_DAN_822.png")

results = model.track(frame, classes=_vehicles, persist=True, tracker="bytetrack.yaml")

boxes = results[0].boxes.xywh.cpu()
track_ids = results[0].boxes.id.int().cpu().tolist()

# Visualize the results on the frame
annotated_frame = results[0].plot()

for box, track_id in zip(boxes, track_ids):
    x, y, w, h = box
    track = track_history[track_id]
    track.append((float(x), float(y)))  # x, y center point
    if len(track) > 30:  # retain 90 tracks for 90 frames
        track.pop(0)

    # Draw the tracking lines
#     points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#     cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

# # Display the annotated frame
# height, width, _ = annotated_frame.shape 
cv2.imwrite("DATN\\detect_vehicle.jpg", annotated_frame)

# cv2.imshow("YOLOv8 Tracking", cv2.resize(annotated_frame, (int(width/de_scale), int(height/de_scale))))

# Release the video capture object and close the display window
# cap.release()
cv2.destroyAllWindows()

# =========================== ĐÁNH GIÁ TẬP TEST ==========================
# Evaluate the model's performance on the test set
# metrics = model.val(data='path/to/your/test_set_data.yaml')
# print(metrics.box.map)  # Print mAP50-95
# print(metrics.box.map50)  # Print mAP50
# print(metrics.box.map75)  # Print mAP75