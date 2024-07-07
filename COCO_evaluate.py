from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

# Customize validation settings
validation_results = model.val(data="coco8.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")

# # Validate the model
# metrics = model.val()
print(f"mAP50-95: {validation_results.box.map}")
print(f"mAP50: {validation_results.box.map50}")
print(f"mAP75: {validation_results.box.map75}")