import cv2
import numpy as np
from pathlib import Path

yvu_path = "/home/liuyang/Documents/haisi/mpp_ive-md_svp/out/image_from_vpss.bin"
imgH = 2160
imgW = 3840

with open(yvu_path, "rb") as f:
    data = f.read()
    yvu = np.ndarray((int(imgH * 1.5), imgW), dtype=np.uint8, buffer=data)
    bgr = cv2.cvtColor(yvu, cv2.COLOR_YUV2BGR_NV21);
    cv2.imwrite(str(Path(yvu_path).with_suffix(".jpg")), bgr)
