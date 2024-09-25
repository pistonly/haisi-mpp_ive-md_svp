import cv2
import numpy as np
from pathlib import Path

def yuv420sp2jpg(yuv_name, width, height):
    with open(yuv_name, "rb") as f:
        data = f.read()
        yuv = np.ndarray((int(height * 1.5), width), dtype=np.uint8, buffer=data)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        cv2.imwrite(str(Path(yuv_name).with_suffix(".jpg")), bgr)

yuv420sp2jpg("/home/liuyang/Documents/haisi/mpp_ive-md_svp/data/dog_bike_car_640x640_yuv420sp.bin", 640, 640)
for i in range(4):
    yuv_name = f"split_yuv_{i}.bin"
    yuv420sp2jpg(yuv_name, 320, 320)

    yuv_name = f"split_yuv_{i}_v2.bin"
    yuv420sp2jpg(yuv_name, 320, 320)

yuv420sp2jpg("/home/liuyang/Documents/haisi/mpp_ive-md_svp/test/combined_yuv.bin", 640, 640)
