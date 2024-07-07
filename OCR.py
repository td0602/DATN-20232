from PIL import Image
import cv2
from ultralytics import YOLO
from IPython.display import display
import os
import time

# load model
yolo_coco = YOLO('yolov8l.pt')
yolo_coco.to('cuda') # cấu hình để sử dụng gpu
yolo_LP_detect = YOLO('weight/detect_100epochs/best.pt')
yolo_LP_detect.to('cuda')
yolo_license_plate = YOLO('weight/ocr_100epochs/best.pt')
yolo_license_plate.to('cuda')

# Load image
img = cv2.imread("test_images/license_plate.png")

plates = yolo_license_plate.predict(img)[0]

img_result = plates.plot()
cv2.imshow("dETEct_plate",img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ======================== ĐÁNH GIÁ TẬP TEST ============================
# Validate on the test set
# metrics = yolo_license_plate.val(data='/media/sanslab/Data/doantv/DATN/dataset/OCR1_DATN/data.yaml', split='test')
# print(metrics.box.map)  # Print mAP50-95
# print(metrics.box.map50)  # Print mAP50
# print(metrics.box.map75)  # Print mAP75