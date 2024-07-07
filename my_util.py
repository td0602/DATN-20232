from collections import defaultdict
import math
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict

# ================================= CẤU HÌNH ==============================================
names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "R", "S", "T", "U", "V", "X", "Y", "Z"]
# load model
yolo_coco = YOLO('yolov8l.pt')
yolo_coco.to('cuda') # cấu hình để sử dụng gpu
yolo_LP_detect = YOLO('weight/detect_100epochs/best.pt')
yolo_LP_detect.to('cuda')
yolo_license_plate = YOLO('weight/ocr_100epochs/best.pt')
yolo_license_plate.to('cuda')

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

# ==================================== CŨ ===================================

# xác định đường thẳng ax + b = y đi qua 2 điểm (x1, y1) và (x2, y2)
def linear_equation(x1, y1, x2, y2):
    b = y1 - (y2 - y1) * x1 / (x2 - x1) 
    a = (y1 - b) / x1 
    return a, b

# Kiểm tra xem điểm (x, y) có nằm trên đường thẳng 2 điểm (x1, y1) và (x2, y2) không
def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2) # tính đhệ số góc và độ dốc đường thẳng
    y_pred = a*x+b # lấy ra kết quả y dự đoán
    # |y - y_pred| <= 3 --> True
    return(math.isclose(y_pred, y, abs_tol = 3))

# detect character and number in license plate: phát hiện ký tự và biển số xe
def read_plate(yolo_license_plate, im, LP_type):
    global names

    # LP_type = "1"
    results = yolo_license_plate.predict(im)[0]
    bb_list = results.boxes.data.tolist()
    # kiểm tra số lượng ký tự phát hiện có nằm trong giới hạn SL ký tự biển số VN ?
    if len(bb_list) == 0 or len(bb_list) < 8 or len(bb_list) > 9:
        return "unknown", -1, -1
    center_list = [] # chứa vị trí trung tâm và class_id của các ký tự biển só
    y_mean = 0
    y_sum = 0
    total_score = 0
    for bb in bb_list:
        x1, y1, x2, y2, score, class_id = bb
        # Tính toán vị trí trung tâm của bounding boxes của mỗi ký tự trong biển số
        x_c = (x1+x2)/2
        y_c = (y1+y2)/2
        y_sum += y_c
        center_list.append([x_c,y_c,class_id]) # thêm vào vị trí trung tâm và class_id
        total_score += score

    # find 2 point to draw line: tìm 2 điểm để vẽ ký tự biển số lên
    l_point = center_list[0]
    r_point = center_list[0]
    # Tìm điểm trái cực trị và phải cực trị
    for cp in center_list:
        if cp[0] < l_point[0]:
            l_point = cp
        if cp[0] > r_point[0]:
            r_point = cp
    # Kiểm tra xem tất các các ký tự trong biển số nằm trên 1 dòng hay 2 dòng
    # for ct in center_list:
    #     if l_point[0] != r_point[0]:
    #         if (check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]) == False):
    #             LP_type = "2"

    y_mean = int(int(y_sum) / len(bb_list)) # tính giá trị TB tọa độ y của các bounding box
    # size = results.pandas().s # Trích xuất kích thước của hình ảnh từ kết quả dự đoán

    # 1 line plates and 2 line plates: biển 1 dòng và 2 dòng
    line_1 = []
    line_2 = []
    license_plate = ""
    if LP_type == "2":
        # các định mỗi ký tự sẽ thuộc dòng 1 hay 2 của biển số
        for c in center_list:
            if int(c[1]) > y_mean:
                line_2.append(c)
            else:
                line_1.append(c)
        #  duyệt qua từng điểm trong danh sách line_1, đã được sắp xếp theo tọa độ x của trung tâm của các bounding box.
        for l1 in sorted(line_1, key = lambda x: x[0]):
            license_plate += names[int(l1[2])]
        license_plate += "-"
        for l2 in sorted(line_2, key = lambda x: x[0]):
            license_plate += names[int(l2[2])]
    else:
        for l in sorted(center_list, key = lambda x: x[0]):
            license_plate += names[int(l[2])]
    return license_plate, total_score/len(bb_list), len(bb_list)


def detect_traffic_light_color(image, rect):
    # Extract rectangle dimensions
    x, y, w, h = rect
    # Extract region of interest (ROI) from the image based on the rectangle
    roi = image[y:y+h, x:x+w]
    # cv2.imwrite(f"C:/Users/Tran Doan/OneDrive/DATN/results/traffic_light/{file_name}_crop.jpg", roi)
    # Chuyển đổi vùng ROI từ KG màu BGR sang HSV, HSV thường phù hợp hơn cho các hoạt động dựa trên màu sắc vì nó tách nội dung sắc độ (Hue) khỏi
    # nội dung độ sáng (Value)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Định nghĩa dải màu HSV cho màu đỏ
    red_lower = np.array([0, 200, 200]) # 0, 120, 70
    red_upper = np.array([8, 255, 255]) # 10, 255, 255

    # Định nghĩa dải màu HSV cho màu v
    yellow_lower = np.array([25, 30, 100]) # 20, 100, 100
    yellow_upper = np.array([30, 255, 255]) # 30, 255, 255

    # green_lower = np.array([50, 200, 100]) # 0, 120, 70
    # green_upper = np.array([70, 255, 255]) # 10, 255, 255

    # Tạo mặt nạ nhị phân để làm nổi bật vùng ROI --> để phát hiện màu đỏ và màu vàng trong ROI
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    # green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    # cv2.imwrite(f"C:/Users/Tran Doan/OneDrive/DATN/results/traffic_light/test/red_mask.jpg", red_mask)
    # Font chữ để hiển thị văn bản lên ảnh
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.75
    font_thickness = 2

    # Kiểm tra màu trên mặt nạ nhị phân
    if cv2.countNonZero(red_mask) > 0:
        text_color = (0, 0, 255)
        message = "RED" # "Detected Signal Status: Stop"
        color = 'red'
    elif cv2.countNonZero(yellow_mask) > 0:
        text_color = (0, 255, 255)
        message = "YELLOW" # "Detected Signal Status: Caution"
        color = 'yellow'
    else:
        text_color = (0, 255, 0)
        message = "GREEN" #"Detected Signal Status: Go"
        color = 'green'
    # else:
    #     color = 'unknown'
    # elif cv2.countNonZero(green_mask) > 0:
    #     text_color = (0, 255, 0)
    #     message = "Detected Signal Status: Go"
    #     color = 'green'
    # else:
    #     text_color = (0, 255, 255)
    #     message = "Detected Signal Status: Caution"
    #     color = 'yellow'

    # Overlay the detected traffic light status on the main image: Xếp chồng trạng thái đèn giao thông được phát hiện trên hình ảnh chính
    # cv2.putText(image, message, (186, 32), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    # Add a separator line: Thêm một dòng phân cách
    # cv2.putText(image, 34*'-', (10, 115), font, font_scale, (255,255,255), font_thickness, cv2.LINE_AA)

    # Return the modified image and detected color: Trả lại hình ảnh đã sửa đổi và màu được phát hiện
    return color

def my_ROIS(img):
    rect = (tl_x1, tl_y1, tl_x2-tl_x1, tl_y2-tl_y1)
    color = detect_traffic_light_color(img, rect)

    img_tl = img[tl_y1:tl_y2, tl_x1:tl_x2]
    img_vehicles = img[vehicles_y1:vehicles_y2, vehicles_x1:vehicles_x2]
    # Tạo vùng ROI (Region of Interest) trong ảnh vehicles
    roi = img_vehicles[5:5 + height_tl, width_vehicles - with_tl - 5:width_vehicles - 5]
    # Thực hiện việc ghép ảnh
    result = cv2.addWeighted(roi, 0, img_tl, 1, 0)
    # Cập nhật vùng ROI trong ảnh gốc với ảnh đã ghép
    img_vehicles[5:5 + height_tl, width_vehicles - with_tl - 5:width_vehicles - 5] = result
    return img_vehicles, color
# đầu vào là (tọa dộ box hiện tại, (x1, y1) box trước, ảnh trước trước)
def xac_dinh_toa_do_so_voi_anh_truoc(rect1, rect2, img):
    height, width, _ = img.shape
    x1, y1, x2, y2 = rect1
    a, b = rect2
    x1 = int(x1+ a) # xmin
    if x1 > width:
        x1 =width
    y1 = int( y1 + b) # ymin
    if y1 > height:
        y1 =height
    x2 = int(x2 + a) # xmin
    if x2 > width:
        x2 = width
    y2 = y2 + b # ymin
    if y2 > height:
        y2 = height
    return x1, y1, x2, y2

# Ghi thông tin ra file
def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{}\n'.format('Frame Number', 'Vehicle',
                                                'License Plate', 'Average Score'))
        for frame_nmr in results.keys():
          for vehicle_count in results[frame_nmr].keys():
            # print(results[frame_nmr][vehicle_count])
            if 'vehicle' in results[frame_nmr][vehicle_count].keys() and \
            'license_plate' in results[frame_nmr][vehicle_count].keys():
              f.write('{},{},{},{}\n'.format(frame_nmr,
                results[frame_nmr][vehicle_count]['vehicle']['type'],
                results[frame_nmr][vehicle_count]['license_plate']['text'],
                results[frame_nmr][vehicle_count]['license_plate']['ave_score'])
                            )
        f.close()
        
# tăng cường độ tương phản hình ảnh
def changeContrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # chuyển màu từ BGR sang LAB, L là độ sáng Light, A (xanh-đỏ) và B (vàng-xanh) đại diện cho máu sắc
    l_channel, a, b = cv2.split(lab)
    # clipLimit kiểm soát mức độ tương phản tối đa để tránh hiện tượng quá sáng. tileGridSize xác định kích thước của các ô nhỏ mà CLAHE sẽ áp dụng riêng rẽ.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    # sau khi thay đổi độ sáng thì lại kết hợp lại thành 1 hình ảnh LAB
    limg = cv2.merge((cl,a,b))
    # chuyển lại ảnh về BGR
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img
# xoay ảnh
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
#  tính độ nghiêng
def compute_skew(src_img, center_thres):
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('upsupported image type')
    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 1.5, maxLineGap=h/3.0)
    if lines is None:
        return 1

    min_line = 100
    min_line_pos = 0
    for i in range (len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            center_point = [((x1+x2)/2), ((y1+y2)/2)]
            if center_thres == 1:
                if center_point[1] < 7:
                    continue
            if center_point[1] < min_line:
                min_line = center_point[1]
                min_line_pos = i

    angle = 0.0
    nlines = lines.size
    cnt = 0
    for x1, y1, x2, y2 in lines[min_line_pos]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if math.fabs(ang) <= 30: # excluding extreme rotations
            angle += ang
            cnt += 1
    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/math.pi

def deskew(src_img, change_cons, center_thres):
    if change_cons == 1:
        return rotate_image(src_img, compute_skew(changeContrast(src_img), center_thres))
    else:
        return rotate_image(src_img, compute_skew(src_img, center_thres))

# ==================================== MỚI===================================

# format biển
dict_char_to_int = {'D': '0',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    'B': '8',
                    'G': '6'}

dict_int_to_char = {'0': 'D',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '9': 'P'}
# TH biển 2 dòng: 
# Dòng 1: 2 ký tự đầu chắc chắn là số, ký tự thứ 3 là chữ, ký tự thứ 4 có thể là chữ hoặc số nên k mapping
# Dòng 2: 5 ký tự đều là số 
mapping_line_1 = {0: dict_char_to_int, 1: dict_char_to_int, 2: dict_int_to_char}
mapping_line_2 = {0: dict_char_to_int, 1: dict_char_to_int, 2: dict_char_to_int, 3: dict_char_to_int, 4: dict_char_to_int}
# TH biển 1 dòng: 2 ký tự đầu là số, ký tự thứ 3 là chứ, các ký tự tiếp theo là số
mapping_1_line = {0: dict_char_to_int, 1: dict_char_to_int, 2: dict_int_to_char, 3: dict_char_to_int, 4: dict_char_to_int,
               5: dict_char_to_int, 6: dict_char_to_int, 7: dict_char_to_int}
mapping_plate = {0: dict_char_to_int, 1: dict_char_to_int, 2: dict_int_to_char, 3: dict_char_to_int, 4: dict_char_to_int,
               5: dict_char_to_int, 6: dict_char_to_int, 7: dict_char_to_int, 8: dict_char_to_int, 9: dict_char_to_int,
               10: dict_char_to_int, 11: dict_char_to_int, 12: dict_char_to_int}

# kiểm tra xem văn bản biển số có tuân thủ đúng định dạng yêu cầu không?
numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
characters = ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "R", "S", "T", "U", "V", "X", "Y", "Z"]

def check_line1(text):
    try:
        if (text[0] in numbers or text[0] in dict_char_to_int.keys()) and \
        (text[1] in numbers or text[1] in dict_char_to_int.keys()) and \
        (text[2] in characters or text[2] in dict_int_to_char.keys()):
            return True
        else:
            return False
    except Exception as e:
        print('Lỗi ở hàm check_line1')
        return False

def check_line2(text):
    try:
        if (text[0] in numbers or text[0] in dict_char_to_int.keys()) and \
        (text[1] in numbers or text[1] in dict_char_to_int.keys()) and \
        (text[2] in numbers or text[2] in dict_char_to_int.keys()) and \
        (text[3] in numbers or text[3] in dict_char_to_int.keys()):
            if len(text) == 4:
                return True
            if (text[4] in numbers or text[4] in dict_char_to_int.keys()):
                return True
            else:
                return False
        else:
            return False
    except Exception as e:
        print('Lỗi ở hàm check_line2')
        return False

def check_1_line(text):
    try:
        if (text[0] in numbers or text[0] in dict_char_to_int.keys()) and \
        (text[1] in numbers or text[1] in dict_char_to_int.keys()) and \
        (text[2] in characters or text[2] in dict_int_to_char.keys()) and \
        (text[3] in numbers or text[3] in dict_char_to_int.keys()) and \
        (text[4] in numbers or text[4] in dict_char_to_int.keys()) and \
        (text[5] in numbers or text[5] in dict_char_to_int.keys()) and \
        (text[6] in numbers or text[6] in dict_char_to_int.keys()) and \
        (text[7] in numbers or text[7] in dict_char_to_int.keys()):
            return True
        else:
            return False
    except Exception as e:
        print('Lỗi ở hàm check_1_line')
        return False

# lấy ra biển số có ave_score max
def lay_bien_score_max(results, vehicle):
  lp = ""
  score_max = 0
  for frame_nmr in results.keys():
    for vehicle_count in results[frame_nmr].keys():
      if 'vehicle' in results[frame_nmr][vehicle_count].keys() and \
          'license_plate' in results[frame_nmr][vehicle_count].keys():
            if results[frame_nmr][vehicle_count]['vehicle']['type'] == vehicle:
              if score_max < results[frame_nmr][vehicle_count]['license_plate']['ave_score']:
                score_max = results[frame_nmr][vehicle_count]['license_plate']['ave_score']
                lp = results[frame_nmr][vehicle_count]['license_plate']['text']
  return lp, score_max

# hàm ghi thông tin phương tiện vi phạm ra file
def write_violate_vehicles(violated_vehicles, results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{}\n'.format('Vehicle', 'License Plate', 'Average Score', 'Violation', 'Date Time'))

        for vehicle in violated_vehicles.keys():
          print(violated_vehicles[vehicle])
          if 'violation' in violated_vehicles[vehicle].keys() and \
              'date_time' in violated_vehicles[vehicle].keys():
              lp, score_max = lay_bien_score_max(results, vehicle)
              f.write('{},{},{},{},{}\n'.format(vehicle, lp, score_max,
                                          violated_vehicles[vehicle]['violation'],
                                          violated_vehicles[vehicle]['date_time'])
                      )
        f.close()

# Nhận diện ký tự trên biển số xe: đầu vào là mô hình nhận diện, ảnh biển số, loại biển
def read_plate_2(yolo_license_plate, im, LP_type):
    results = yolo_license_plate.predict(im)[0]
    bb_list = results.boxes.data.tolist()

    # kiểm tra số lượng ký tự phát hiện có nằm trong giới hạn SL ký tự biển số VN ?
    if len(bb_list) == 0: # or len(bb_list) < 8 or len(bb_list) > 9:
        return "unknown", -1, -1
    
    
    center_list = [] # chứa vị trí trung tâm và class_id của các ký tự biển só
    y_mean = 0
    y_sum = 0
    total_score = 0
    for bb in bb_list:
        x1, y1, x2, y2, score, class_id = bb
        # Tính toán vị trí trung tâm của bounding boxes của mỗi ký tự trong biển số
        x_c = (x1+x2)/2
        y_c = (y1+y2)/2
        y_sum += y_c
        center_list.append([x_c,y_c,class_id]) # thêm ký tự và tâm của ký tự đó vào mảng
        total_score += score

    y_mean = int(int(y_sum) / len(bb_list)) # tọa độ y TB

    line_1 = []
    line_2 = []
    license_plate_ = ""
    if LP_type == "2":
        # xác định mỗi ký tự sẽ thuộc dòng 1 hay 2 của biển số
        for c in center_list:
            if int(c[1]) > y_mean:
                line_2.append(c)
            else:
                line_1.append(c)

        text_line_1 = []
        text_line_2 = []
        for l1 in sorted(line_1, key = lambda x: x[0]): # duyệt lần lượt các phần tử của line_1 sau khi các phần tử này được sắp xếp theo giá trị trục x
          text_line_1.append(names[int(l1[2])])
        for l2 in sorted(line_2, key = lambda x: x[0]):
            text_line_2.append(names[int(l2[2])])
        # thêm đk cho biển 2 line -- MỚI
        if len(text_line_1) > 4 or len(text_line_1) < 3 or len(text_line_2) > 5 or len(text_line_2) < 4:
          return "unknown", -1, -1

        # check format
        if (check_line1(text_line_1)==False) or (check_line2(text_line_2)==False):
          return "unknown", -1, -1

        # format biển số -- MỚI
        for j in [0, 1, 2]: # xử lý dòng 1
          if text_line_1[j] in mapping_line_1[j].keys():
              license_plate_ += mapping_line_1[j][text_line_1[j]]
          else:
              license_plate_ += text_line_1[j]
        if len(text_line_1) == 4:
          license_plate_ += text_line_1[3]
        license_plate_ += "-"

        for j in [0, 1, 2, 3]: # xử lý dòng 2
          if text_line_2[j] in mapping_line_2[j].keys():
              license_plate_ += mapping_line_2[j][text_line_2[j]]
          else:
              license_plate_ += text_line_2[j]
        if len(text_line_2) == 5:
          if text_line_2[4] in mapping_line_2[4].keys():
              license_plate_ += mapping_line_2[4][text_line_2[4]]
          else:
              license_plate_ += text_line_2[4]
    else:
        text_1_line = []
        for l in sorted(center_list, key = lambda x: x[0]):
            text_1_line.append(names[int(l[2])])

        # check format
        if (check_1_line(text_1_line)==False):
          return "unknown", -1, -1

        for j in [0, 1, 2, 3, 4, 5, 6, 7]:
          if text_1_line[j] in mapping_1_line[j].keys():
              license_plate_ += mapping_1_line[j][text_1_line[j]]
          else:
              license_plate_ += text_1_line[j]

    return license_plate_, total_score/len(bb_list), len(bb_list)

# Với mong muốn vẫn trả ra các kết quả với số lượng ký tự có thể không đủ theo tiêu chuẩn
# Xây dựng thêm hàm đọc biển thứ 3: 
# Giải pháp: cứ việc lấy ra các ký tự cho vào mảng license_plate[] --> format --> trả về chuỗi ký tự lisence_plate_
# Nhận diện ký tự trên biển số xe: đầu vào là mô hình nhận diện, ảnh biển số, loại biển
def read_plate_3(yolo_license_plate, im, LP_type):
    results = yolo_license_plate.predict(im)[0]
    bb_list = results.boxes.data.tolist()

    # kiểm tra số lượng ký tự phát hiện có nằm trong giới hạn SL ký tự biển số VN ?
    if len(bb_list) == 0: # or len(bb_list) < 8 or len(bb_list) > 9:
        return "unknown", -1, -1
    
    
    center_list = [] # chứa vị trí trung tâm và class_id của các ký tự biển só
    y_mean = 0
    y_sum = 0
    total_score = 0
    for bb in bb_list:
        x1, y1, x2, y2, score, class_id = bb
        # Tính toán vị trí trung tâm của bounding boxes của mỗi ký tự trong biển số
        x_c = (x1+x2)/2
        y_c = (y1+y2)/2
        y_sum += y_c
        center_list.append([x_c,y_c,class_id]) # thêm ký tự và tâm của ký tự đó vào mảng
        total_score += score

    y_mean = int(int(y_sum) / len(bb_list)) # tọa độ y TB

    line_1 = []
    line_2 = []
    license_plate_arr = []
    license_plate_ = ""
    if LP_type == "2":
        # xác định mỗi ký tự sẽ thuộc dòng 1 hay 2 của biển số
        for c in center_list:
            if int(c[1]) > y_mean:
                line_2.append(c)
            else:
                line_1.append(c)

        for l1 in sorted(line_1, key = lambda x: x[0]): # duyệt lần lượt các phần tử của line_1 sau khi các phần tử này được sắp xếp theo giá trị trục x
            license_plate_arr.append(names[int(l1[2])])
        license_plate_arr.append('-')
        for l2 in sorted(line_2, key = lambda x: x[0]):
            license_plate_arr.append(names[int(l2[2])])

        # format lại
        for j in range(len(license_plate_arr)):
            if license_plate_arr[j] == '-':
                license_plate_ += license_plate_arr[j]
                continue
            if j != 3:
                if license_plate_arr[j] in mapping_plate[j].keys():
                    license_plate_arr[j] = mapping_plate[j][license_plate_arr[j]]
            license_plate_ += license_plate_arr[j]
        
    else:
        for l in sorted(center_list, key = lambda x: x[0]):
            license_plate_arr.append(names[int(l[2])])

        # check format
        for j in range(len(license_plate_arr)):
            if license_plate_arr[j] == '-':
                license_plate_ += license_plate_arr[j]
                continue
            if j != 3:
                if license_plate_arr[j] in mapping_plate[j].keys():
                    license_plate_arr[j] = mapping_plate[j][license_plate_arr[j]]
            license_plate_ += license_plate_arr[j]

    return license_plate_, total_score/len(bb_list), len(bb_list)
