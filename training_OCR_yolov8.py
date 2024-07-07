from ultralytics import YOLO

model = YOLO('yolov8m.pt')

results = model.train(data = '/media/sanslab/Data/doantv/DATN/test_yolov8/OCR1/data.yaml', epochs=100, imgsz=640, plots=True)