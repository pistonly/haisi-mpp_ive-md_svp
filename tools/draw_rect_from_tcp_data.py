import cv2
import numpy as np
from pathlib import Path
import struct
import time
import glob

url = "rtsp://172.23.24.52:8554/test"
# tcp_data_dir = Path("/home/liuyang/Documents/haisi/ai-sd3403/ai-sd3403/test/test_tcp/x86/build/")
tcp_data_dir = Path("/home/liuyang/Documents/tmp/md-4k/")


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

def deserialize_v2(buffer):
    dec = []
    len = struct.unpack_from("I", buffer, 0)[0]
    cameraId = struct.unpack_from("B", buffer, 4)[0]
    timestamp = struct.unpack_from("Q", buffer, 5)[0]
    if len > 13:
        dec_num = (len - 13) // 24
        dec = np.ndarray((dec_num, 6), dtype=np.float32, buffer=buffer[13:])
    print(len, cameraId, timestamp, dec)
    return dec


def get_decs(dec_file):
    if (Path(dec_file).is_file()):
        with open(dec_file, "rb") as f:
            data_buffer = f.read()
            # data = deserialize(data_buffer)
            data = deserialize_v2(data_buffer)
            # print(data)
            return data
    else:
        print(f"{dec_file} is not filename")
        return []

def draw_rec(image, decs):
    for x, y, h, w, conf, cls in decs:
        if conf < 0.2:
            continue
        if int(h) == 0:
            # print(x, y, h, w, conf, cls)
            x0 = int(x)
            y0 = int(y)
            x1 = int(x + 32)
            y1 = int(y + 32)
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
        else:
            print(x, y, h, w, conf, cls)
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

    glob_res = glob.glob(str(tcp_data_dir / f"decs_camera-0_image-{img_id:06d}*.bin"))
    decs = []
    if len(glob_res):
        decs = get_decs(str(tcp_data_dir / glob_res[0]))

    frame = draw_rec(frame, decs)

    img_id += 1

    cv2.imshow('RTSP Stream', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    time.sleep(0.2)


cap.release()
cv2.destroyAllWindows()
