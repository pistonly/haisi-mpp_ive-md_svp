import cv2
import numpy as np
from pathlib import Path

yvu_path = "/home/liuyang/Documents/tmp/md-4k/tmp/merged_roi_000054.bin"
imgH = 1280
imgW = 1280

with open(yvu_path, "rb") as f:
    data = f.read()
    print(len(data), imgH * imgW * 1.5)
    yvu = np.ndarray((int(imgH * 1.5), imgW), dtype=np.uint8, buffer=data)
    bgr = cv2.cvtColor(yvu, cv2.COLOR_YUV2BGR_NV21);
    cv2.imwrite(str(Path(yvu_path).with_suffix(".jpg")), bgr)

import glob
yuv_paths = glob.glob(str(Path(yvu_path).parent / "merged_roi_*.bin"))

for yuv_name in yuv_paths:
    yuv_p = str(Path(yvu_path).parent / yuv_name)

    with open(yuv_p, "rb") as f:
        data = f.read()
        print(len(data), imgH * imgW * 1.5)
        yvu = np.ndarray((int(imgH * 1.5), imgW), dtype=np.uint8, buffer=data)
        bgr = cv2.cvtColor(yvu, cv2.COLOR_YUV2BGR_NV21);
        cv2.imwrite(str(Path(yuv_p).with_suffix(".jpg")), bgr)
