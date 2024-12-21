import cv2
import numpy as np
from ultralytics import YOLO

def getColours(cls_num):
    # mendapatkan kode warna tertentu untuk suatu kelas
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
        (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

def getLargestBox(boxes):
    # mengembalikan bounding box terbesar
    max_area = 0
    largest_box = None
    for box in boxes:
        [x1, y1, x2, y2] = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            largest_box = box

    return largest_box

def predict(yolo : YOLO, img : np.ndarray):

    # daftar kelas yang akan dideteksi beserta index kelas
    desired_classes = ['car', 'truck', 'motorcycle', 'bus']
    class_indices = [idx for idx, name in yolo.names.items() if name in desired_classes]

    # proses prediksi kelas dari citra masukan
    result = yolo.predict(img, classes=class_indices, verbose=False)

    # hasil bounding box yang terdeteksi
    boxes = result[0].boxes

    if (len(boxes) > 0):
        # pilih bounding box terbesar
        box = getLargestBox(boxes) 

        # memperoleh nama kelas
        class_id = int(box.cls.item())
        class_name = yolo.names[class_id]

        # menggambar bounding box
        colour = getColours(class_id)
        [x1, y1, x2, y2] = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

        # menggambar teks kelas dan persentase keyakinan pada citra
        text = f'{class_name} {box.conf[0]:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        padding = 5
        cv2.rectangle(img, (x1, y1), (x1 + text_width + 2 * padding, y1 + text_height + 2 * padding), colour, -1)
        cv2.putText(img, text, (x1 + padding, y1 + padding + text_height), font, font_scale, (0, 0, 0), font_thickness)

        return (yolo.names[class_id], img)
    else:
        # tidak ada bounding box terdeteksi
        return ("kendaraan tidak dikenal", img)