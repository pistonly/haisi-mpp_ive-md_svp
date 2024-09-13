import cv2
import numpy as np
from pathlib import Path
import struct
import time

url = "rtsp://172.23.24.52:8554/test"
tcp_data_dir = Path("/home/liuyang/Documents/haisi/ai-sd3403/ai-sd3403/test/test_tcp/x86/build/")
# tcp_data_dir = Path("/home/liuyang/Documents/tmp/tmp/")


def deserialize(buffer):
    data = []
    idx = 0
    while idx < len(buffer):
        # 读取子vector的大小
        vec_size = struct.unpack_from('Q', buffer, idx)[0]  # 'Q'表示unsigned long long（8字节）
        idx += struct.calcsize('Q')
        
        # 读取子vector的所有float值
        vec = []
        for _ in range(vec_size):
            value = struct.unpack_from('f', buffer, idx)[0]  # 'f'表示float（4字节）
            idx += struct.calcsize('f')
            vec.append(value)
        
        data.append(vec)
    
    return data

def get_decs(dec_file):
    if (Path(dec_file).is_file()):
        with open(dec_file, "rb") as f:
            data_buffer = f.read()
            data = deserialize(data_buffer)
            print(data)
            return data
    else:
        print(f"{dec_file} is not filename")
        return []

def draw_rec(image, decs):
    for x, y, h, w, conf, cls in decs:
        if conf < 0.1:
            continue
        if int(h) == 0:
            x0 = int(x)
            y0 = int(y)
            x1 = int(x + 32)
            y1 = int(y + 32)
            continue
        else:
            x0 = int(x - w // 2)
            y0 = int(y - h // 2)
            x1 = int(x + w // 2)
            y1 = int(y + h // 2)
        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 255), 2)
    return image

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Cannot open RTSP stream.")
    exit()


cv2.namedWindow('RTSP Stream', cv2.WINDOW_NORMAL)

img_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame (stream end?). Exiting ...")
        break

    decs = get_decs(str(tcp_data_dir / f"decs_image_{img_id}.bin"))
    # decs = get_decs(str(tcp_data_dir / f"decs_{img_id:06d}.bin"))
    frame = draw_rec(frame, decs)

    img_id += 1

    cv2.imshow('RTSP Stream', frame)
    while (1):
        if cv2.waitKey(1) == ord('n'):
            break


cap.release()
cv2.destroyAllWindows()
