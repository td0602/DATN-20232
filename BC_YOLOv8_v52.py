import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict

import time

# box đèn giao thông
tl_x1 = 2389
tl_y1 = 0
tl_x2 = 2541
tl_y2 = 153
with_tl = tl_x2 - tl_x1
height_tl = tl_y2 - tl_y1

# frame cắt phương tiện
vehicles_x1 = 0
vehicles_y1 = 1050
vehicles_x2 = 2950
vehicles_y2 = 2160
width_vehicles = vehicles_x2 - vehicles_x1
height_vehicles = vehicles_y2 - vehicles_y1

# hàm tự tạo
import my_util

# Khai báo hàm
def nhan_dien_bien_tren_xe(vehicle_track):
    # count = 0
    global frame_nmr
    for track in vehicle_track:
        if track.is_confirmed(): # ktra dc xác nhận bởi tracker    
            track_id = track.track_id
            # lấy tọa độ + class_id để vẽ lên frame
            ltrb = track.to_ltrb() # chuyển tọa độ (x1, x2, w, h) về (x1, y1) (x2, y2)
            x1, y1, x2, y2 = map(int, ltrb)
            class_id = track.get_det_class()
            color = colors[int(class_id)]
            B, G, R = map(int, color)

            crop_img_vehicle = frame[y1:y2, x1:x2]
            cv2.rectangle(frame2, (x1,y1), (x2,y2), color = (B, G, R), thickness = 2)
            cv2.putText(frame2, f"{class_vehicles[class_id]}_{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            if f'{class_id}_{track_id}' in vehicles_pass:
                continue
            # Nhận diện biển số trong box vehicles
            nhan_dien_ky_tu(crop_img_vehicle, (x1, y1, x2, y2))

def nhan_dien_ky_tu(crop_img_vehicle, box_vehicles):
    global frame_nmr
    x1, y1,x2, y2 = box_vehicles
    try:

        plates = yolo_LP_detect.predict(crop_img_vehicle, conf=0.1)[0]
        list_plates = plates.boxes.data.tolist()
        if len(list_plates) == 1:
            plate_id = 1
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

                cv2.rectangle(frame2, (x1_plate,y1_plate), (x2_plate,y2_plate), color = (0,0,225), thickness = 2)
                # cv2.imwrite(folder_path + f"\\{file_name}_{count}.png", crop_img_plate)

                lp = ""
                for cc in range(0,2):
                    for ct in range(0,2):
                        lp, score_ave, len_characters = my_util.read_plate(yolo_license_plate, my_util.deskew(crop_img_plate, cc, ct), LP_type) # utils_rotate.deskew(crop_img, cc, ct)
                        # lp = my_util.read_plate(yolo_license_plate, crop_img, LP_type)
                        # print(lp)
                        if lp != "unknown":
                            # đánh dấu đã nhận diện được biển số OK
                            if score_ave > 0.9:
                                new_vehicle_pass = f"{class_id}_{track_id}"
                                vehicles_pass.append(new_vehicle_pass)
                                print(f"++++++{len_characters}+++++")
                            cv2.putText(frame2, lp, (x1_plate, y1_plate-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            # tạo chuỗi ghi ra file
                            results[frame_nmr] = {}
                            results[frame_nmr][plate_id] = {'vehicle': {'type': f'{class_vehicles[class_id]}_{track_id}'},
                                                    'license_plate': {'text': lp,
                                                                        'total_score': score_ave},
                                                    'check_violation': {'violation': "red light violation",
                                                                        'date_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}}

                            flag = 1
                            break
                    if flag == 1:
                        break

    except ZeroDivisionError:
        print("exception")
# danh sách thông tin để ghi ra file
results = {}
# khởi tạo deepsort
# load model
yolo_coco = YOLO('yolov8m.pt')
yolo_coco.to('cuda') # cấu hình để sử dụng gpu
yolo_LP_detect = YOLO('weight/detect_100epochs/best.pt')
yolo_LP_detect.to('cuda')
yolo_license_plate = YOLO('weight/ocr_100epochs/best.pt')
yolo_license_plate.to('cuda')

# Các phương tiện sẽ detect: xe máy, ô tô, xe bus, xe tải
_vehicles = [2, 3, 5, 7]
# giam ti le khung hinh
de_scale = 4 
# Lấy tên class
CLASS_NAMES_DICT = yolo_coco.model.names
class_vehicles = {}
for id in _vehicles:
    class_vehicles[id] = CLASS_NAMES_DICT[id]
# tạo mảng color cho mỗi phương tiện
colors = np.random.randint(0, 255, size = (8, 3))
# tạo danh sách rỗng lưu trữ lịch sử theo dõi 
track_history = defaultdict(lambda: [])
# mảng chứa các phương tiện đã nhận diện được biển số OK
vehicles_pass = []
_extend = 5

cap = cv2.VideoCapture('test_videos/XA_DAN_8_IPXS.mp4')
frame_nmr = 0

while True:    
    frame_nmr += 1
    ret, frame = cap.read()
    if frame is None:
        break
    if frame_nmr % 3 != 0: # số frame video đã bị giảm đi 3 lần
        continue
    if not ret:
       continue

    frame2 = frame.copy()

    # Bắt đầu đo thời gian
    start_time = time.time()

    # Lấy vùng ROIS: vạch dừng và đèn đỏ + color đèn giao thông
    rect = (tl_x1, tl_y1, tl_x2-tl_x1, tl_y2-tl_y1) # đèn giao thông

    tl_x1, tl_y1, tl_x2, tl_y2 = map(int,(tl_x1, tl_y1, tl_x2, tl_y2))

    cv2.rectangle(frame2, (tl_x1,tl_y1), (tl_x2,tl_y2), color = (0,0,225), thickness = 2)
    color = my_util.detect_traffic_light_color(frame, rect)
    # vùng dừng đèn đỏ
    cv2.rectangle(frame2, (vehicles_x1,vehicles_y1), (vehicles_x2,vehicles_y2), color = (0,0,225), thickness = 2)
    img_vehicles = frame[vehicles_y1:vehicles_y2, vehicles_x1:vehicles_x2]

    # Kết thúc đo thời gian

    end_time = time.time()
    # Tính toán thời gian thực thi và chuyển đổi sang mili giây
    execution_time = (end_time - start_time) * 1000
    print(f"Thời gian thực thi-{frame_nmr}: {execution_time} ms")


    height, width, _ = frame.shape 

    # nhận diện phương tiện + tracking
    vehicles = yolo_coco.track(img_vehicles, show=False, verbose=False, conf=0.4, classes=_vehicles, persist=True, tracker="bytetrack.yaml", device=0)[0]

    list_vehicles = vehicles.boxes.data.tolist()
    # khai báo mảng chứa box và id phương tiện
    vehicles = []
    # # lấy ra tt box phương tiện
    # for vehicle in list_vehicles:
    #     if len(vehicle) == 7:
    #         print(vehicle)
    #         x1, y1, x2, y2, track_id, score, class_id = vehicle
    #         # xác định lại tọa độ với ảnh gốc
    #         x1, y1, x2, y2 = my_util.xac_dinh_toa_do_so_voi_anh_truoc((x1, y1, x2, y2), (vehicles_x1, vehicles_y1), frame)
    #         vehicles = track_history[track_id]
    #         vehicles.append([[x1, y1, x2-x1, y2-y1], score, class_id])

    #         color = colors[int(class_id)]
    #         B, G, R = map(int, color)
    #         x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    #         # vẽ box cho vehicles
    #         cv2.rectangle(frame2, (x1,y1), (x2,y2), color = (B, G, R), thickness = 2)
    #         cv2.putText(frame2, f"{class_vehicles[int(class_id)]}_{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    #         # Nếu phương tiện đã nhận diện biển OK thì bỏ qua
    #         check_pass = f"{class_id}_{track_id}"
    #         if check_pass in vehicles_pass:
    #             continue
    #         crop_img_vehicle = frame[y1:y2, x1:x2]

    #         nhan_dien_ky_tu(crop_img_vehicle, (x1, y1, x2, y2))

    cv2.imshow("frame", cv2.resize(frame2, (int(width/de_scale), int(height/de_scale)))) # cv2.resize(frame2, (int(width/3), int(height/3)))

    if cv2.waitKey(1) & 0xFF == 27: # 27 là esc --> nếu bấm thì thoát
        break
cap.release()    
cv2.destroyAllWindows()
# # write results: ghi KQ ra filr
# my_util.write_csv(results, './test.csv')
# print(f"===================={len(vehicles_pass)}=======================")