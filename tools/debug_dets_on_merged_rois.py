import numpy as np
import cv2
from pathlib import Path
import struct
import time

# image_dir = Path("/home/liuyang/Documents/haisi/ai-sd3403/ai-sd3403/mpp_ive-md_svp/out/")
image_dir = Path("/home/liuyang/Documents/tmp/nnn_debug/")
img_fns = [fn for fn in image_dir.iterdir() if fn.name.startswith("merged_roi") and fn.name.endswith(".bin")]
img_fns = sorted(img_fns
                 )

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




cv2.namedWindow('RTSP Stream', cv2.WINDOW_NORMAL)

img_id = 0
img_H, img_W = 640, 640
while True:
    yuv_path = image_dir / f"merged_roi_{img_id:06d}.bin"
    if not yuv_path.is_file():
        break

    with open(str(yuv_path), 'rb') as f:
        yuv = f.read()
        size = len(yuv)
        w = img_W
        h = size // w
        img = np.ndarray((h, w), dtype=np.uint8, buffer=yuv)
        frame = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12)

    decs = get_decs(str(image_dir / f"0decs_image_{img_id:06d}.bin"))
    frame = draw_rec(frame, decs)

    img_id += 1

    cv2.imshow('RTSP Stream', frame)
    while (1):
        if cv2.waitKey(1) == ord('n'):
            break


cv2.destroyAllWindows()
