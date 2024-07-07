from collections import defaultdict
import math
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict

import time

import my_util

# box đèn giao thông
tl_x1 = 2715
tl_y1 = 0
tl_x2 = 2900
tl_y2 = 160
with_tl = tl_x2 - tl_x1
height_tl = tl_y2 - tl_y1
# toa do y vach dung den do
y_line_1 = 1740
y_line_2 = 1330

# frame cắt phương tiện
vehicles_x1 = 0
vehicles_y1 = 1110
vehicles_x2 = 3465
vehicles_y2 = 2160
width_vehicles = vehicles_x2 - vehicles_x1
height_vehicles = vehicles_y2 - vehicles_y1

# load model
yolo_coco = YOLO('yolov8l.pt')
yolo_coco.to('cuda') # cấu hình để sử dụng gpu
yolo_LP_detect = YOLO('weight/detect_100epochs/best.pt')
yolo_LP_detect.to('cuda')
yolo_license_plate = YOLO('weight/ocr_100epochs/best.pt')
yolo_license_plate.to('cuda')

# Các phương tiện sẽ detect: xe máy, ô tô, xe bus, xe tải
_vehicles = [2, 3, 5, 7]
# giam ti le khung hinh
de_scale = 4
# Lấy  name classes của mô hình
CLASS_NAMES_DICT = yolo_coco.model.names
class_vehicles = {}
for id in _vehicles:
    class_vehicles[id] = CLASS_NAMES_DICT[id]
# tạo mảng color cho mỗi phương tiện
colors = np.random.randint(0, 255, size = (8, 3))
_extend = 5


cap = cv2.VideoCapture('test_videos/XA_DAN_12.mp4')

# ======================================== CHẠY VIDEO ===========================
results = {} 
frame_nmr = 0 # chứa all biển
vehicles_pass = [] # mảng chứa các phương tiện đã nhận diện được biển số OK
violated_vehicles = {} # chứa id phương tiện vi phạm
# mỗi khóa mới có giá trị là danh sách rỗng lưu trữ lịch sử theo dõi
track_history = defaultdict(lambda: [])
while True:
    frame_nmr += 1
    print(f'================================START==={frame_nmr}')

    results[frame_nmr] = {}
    ret, frame = cap.read()
    if frame is None:
        break
    if frame_nmr % 1 != 0: # số frame video đã bị giảm đi 3 lần
        continue
    if not ret:
       continue

    frame2 = frame.copy()

    # Bắt đầu đo thời gian
    start_time = time.time()

    rect = (tl_x1, tl_y1, tl_x2-tl_x1, tl_y2-tl_y1) # vùng chứa đèn giao thông
    tl_x1, tl_y1, tl_x2, tl_y2 = map(int,(tl_x1, tl_y1, tl_x2, tl_y2)) # chuyển các số về số nguyên

    cv2.rectangle(frame2, (tl_x1,tl_y1), (tl_x2,tl_y2), color = (0,0,225), thickness = 2) # vẽ khung đèn giao thông
    color_tl = my_util.detect_traffic_light_color(frame, rect) # lấy màu đèn giao thông

    # vẽ vùng quan tâm có các phương tiện
    cv2.rectangle(frame2, (vehicles_x1,vehicles_y1), (vehicles_x2,vehicles_y2), color = (0,255,0), thickness = 2)
    img_vehicles = frame[vehicles_y1:vehicles_y2, vehicles_x1:vehicles_x2]

    # =================================================================
    # ve vùng vi phạm đèn đỏ
    start_point1 = (vehicles_x1, y_line_1)
    end_point1 = (vehicles_x2, y_line_1)
    start_point2 = (vehicles_x1, y_line_2)
    end_point2 = (vehicles_x2, y_line_2)
    if color_tl == 'red' or color_tl == 'yellow':
        cv2.line(frame2, start_point1, end_point1, (0, 0, 255), 2)
        cv2.line(frame2, start_point2, end_point2, (0, 0, 255), 2)
    else:
        cv2.line(frame2, start_point1, end_point1, (0, 255, 0), 2)
        cv2.line(frame2, start_point2, end_point2, (0, 255, 0), 2)

    # Kết thúc đo thời gian

    end_time = time.time()
    # Tính toán thời gian thực thi và chuyển đổi sang mili giây
    execution_time = (end_time - start_time) * 1000
    print(f"Thời gian thực thi-{frame_nmr}: {execution_time} ms")
    # ======================================================================

    height, width, _ = frame.shape

    # nhận diện phương tiện + tracking
    # vehicles = yolo_coco.track(img_vehicles, show=False, verbose=False, conf=0.4, classes=_vehicles, persist=True, tracker="bytetrack.yaml", device=0)[0]
    # persist = True --> sử dụng khung hình trc để áp dụng vào theo dõi đối tượng khung hình tiếp theo
    vehicles = yolo_coco.track(img_vehicles, show=False, verbose=False, conf=0.2, classes=_vehicles, persist=True, tracker="bytetrack.yaml", device='cpu')[0]
    list_vehicles = vehicles.boxes.data.tolist()
    # khai báo mảng chứa box và id phương tiện
    vehicles_ = []
    # lấy ra tt box phương tiện
    count_vehicle = 0
    for vehicle in list_vehicles:
        if len(vehicle) == 7:
            print(vehicle)
            x1, y1, x2, y2, track_id, score, class_id = vehicle
            # xác định lại tọa độ với ảnh gốc
            x1, y1, x2, y2 = my_util.xac_dinh_toa_do_so_voi_anh_truoc((x1, y1, x2, y2), (vehicles_x1, vehicles_y1), frame)
            
            vehicles_ = track_history[track_id] # khởi danh sách rỗng có khóa là track_id
            vehicles_.append([[x1, y1, x2-x1, y2-y1], score, class_id]) # thêm giá trị cho phần tử có key là track_id bên trên
            # khởi tạo màu box biển số ban đầu
            color_box_plate = (0, 255, 0)
            # thêm phương tiện vi phạm
            if color_tl == "red" and y2 < y_line_1 and y2 > y_line_2 and f"{class_vehicles[class_id]}_{track_id}" not in violated_vehicles:
                violated_vehicles[f"{class_vehicles[class_id]}_{track_id}"] = {'violation': "Red light violation",
                                                                 'date_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            if (color_tl == "red" or color_tl == 'yellow') and y2 < y_line_1 and y2 > y_line_2:
                color_box_plate = (0, 0, 255)

            color = colors[int(class_id)]
            B, G, R = map(int, color)
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            # vẽ box cho vehicles
            cv2.rectangle(frame2, (x1,y1), (x2,y2), color = (B, G, R), thickness = 2)
            cv2.putText(frame2, f"{class_vehicles[int(class_id)]}_{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            # Nếu phương tiện đã nhận diện biển OK thì bỏ qua
            check_pass = f"{class_id}_{track_id}"
            # if check_pass in vehicles_pass:
            #     continue
            crop_img_vehicle = frame[y1:y2, x1:x2]

            # nhận diện ký tự
            try:
                plates = yolo_LP_detect.predict(crop_img_vehicle, conf=0.1)[0]
                list_plates = plates.boxes.data.tolist()
                if len(list_plates) == 1:
                    for plate in list_plates:
                        # count += 1
                        LP_type = "1"
                        flag = 0
                        x1_plate = int(plate[0]) - _extend # xmin
                        if x1_plate < 0:
                            x1_plate = 0
                        y1_plate = int(plate[1]) - _extend # ymin
                        if y1_plate < 0:
                            y1_plate = 0
                        x2_plate = int(plate[2]) + _extend # xmin
                        if x2_plate > width:
                            x2_plate =width
                        y2_plate = int(plate[3]) + _extend # ymin
                        if y2_plate > height:
                            y2_plate = height
                        w_plate = int(plate[2] - plate[0]) # xmax - xmin
                        h_plate = int(plate[3] - plate[1]) # ymax - ymin
                        plate_class_id_plate = int(plate[-1])
                        if plate_class_id_plate == 1:
                            LP_type = "2"
                        crop_img_plate = crop_img_vehicle[y1_plate:y2_plate, x1_plate:x2_plate]

                        # xác định box plate trong frame chính
                        x1_plate, y1_plate, x2_plate, y2_plate = my_util.xac_dinh_toa_do_so_voi_anh_truoc((x1_plate, y1_plate, x2_plate, y2_plate), (x1, y1),frame)
                        # vẽ box biển số lên frame2
                        cv2.rectangle(frame2, (x1_plate,y1_plate), (x2_plate,y2_plate), color = color_box_plate, thickness = 2)
                        # cv2.imwrite(folder_path + f"\\{file_name}_{count}.png", crop_img_plate)
                        print('======================DEBUG-1')
                        lp = ""
                        for cc in range(0,2):
                            for ct in range(0,2):
                                lp, ave_score, len_characters = my_util.read_plate_3(yolo_license_plate, my_util.deskew(crop_img_plate, cc, ct), LP_type) # utils_rotate.deskew(crop_img, cc, ct)
                                # lp = my_util.read_plate(yolo_license_plate, crop_img, LP_type)
                                # print(lp)
                                if lp != "unknown":
                                    # đánh dấu đã nhận diện được biển số OK
                                    if ave_score > 0.9:
                                        new_vehicle_pass = f"{class_id}_{track_id}"
                                        vehicles_pass.append(new_vehicle_pass)
                                    cv2.putText(frame2, lp, (x1_plate, y1_plate-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                                    count_vehicle += 1
                                    results[frame_nmr][count_vehicle] = {'vehicle': {'type': f'{class_vehicles[class_id]}_{track_id}'},
                                                                        'license_plate': {'text': lp, 'ave_score': ave_score}}
                                    # print(f"{frame_nmr}-{count_vehicle}")

                                    flag = 1
                                    # Kết thúc đo thời gian

                                    end_time = time.time()
                                    # Tính toán thời gian thực thi và chuyển đổi sang mili giây
                                    execution_time = (end_time - start_time) * 1000
                                    print(f"Thời gian thực thi-{frame_nmr}: {execution_time} ms")
                                    print('======================DEBUG-2')
                                    break
                            if flag == 1:
                                print('======================DEBUG-1')
                                break
            except ZeroDivisionError:
                print("exception")
    cv2.imshow("frame", cv2.resize(frame2, (int(width/de_scale), int(height/de_scale)))) # cv2.resize(frame2, (int(width/3), int(height/3)))
    print(f'================================END==={frame_nmr}')
    if cv2.waitKey(1) & 0xFF == 27: # 27 là esc --> nếu bấm thì thoát
        break
# ===================================== XỬ LÝ THÔNG TIN ===============================
# # print(violated_vehicles)
# print(f'======SỐ PHƯƠNG TIỆN VI PHẠM========{len(violated_vehicles)}')

# print(results)
# print(f'=============={len(results)}===========')

# # write results: ghi KQ ra FILE
# my_util.write_csv(results, 'results/license_plate.csv') # file chứa biển só all phương tiện
# print(f"===================={len(vehicles_pass)}=======================")
# my_util.write_violate_vehicles(violated_vehicles, results, 'results/violate_vehicle.csv') # file chứa biển số xe vi phạm
# print(f"===================={len(violated_vehicles)}=======================")

cap.release()
cv2.destroyAllWindows()



