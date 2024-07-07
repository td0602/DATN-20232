from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO("yolov8m.pt")

# Open the video file
video_path = "test_videos/XA_DAN_8_IPXS.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])
_vehicles = [2, 3, 5, 7]
de_scale = 6
frame_nmr = 0
sum_times = 0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        try:
            frame_nmr += 1
            print(f'================================START==={frame_nmr}')
            # Bắt đầu đo thời gian
            start_time = time.time()
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, classes=_vehicles, persist=True, tracker="bytetrack.yaml")
            # results_ = results.boxes.data.tolist()
            # for vehicle in results_:
            #     x1, y1, x2, y2, track_id, score, class_id = vehicle
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            # Display the annotated frame
            height, width, _ = annotated_frame.shape 
            cv2.imshow("YOLOv8 Tracking", cv2.resize(annotated_frame, (int(width/de_scale), int(height/de_scale))))

            end_time = time.time()
            # Tính toán thời gian thực thi và chuyển đổi sang mili giây
            execution_time = (end_time - start_time) * 1000
            if frame_nmr > 100:
                sum_times += execution_time
                print(f"Thời gian thực thi TB-{frame_nmr}: {sum_times/(frame_nmr-100)} ms")
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except Exception as e:
            print(f"Error: {e}")

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()