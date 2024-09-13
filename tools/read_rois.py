import numpy as np
import cv2

with open("/home/liuyang/Documents/haisi/ai-sd3403/ai-sd3403/mpp_ive-md_svp/out/rois.bin", "rb") as f:
    imgs = f.read()
    size = len(imgs)
    start = 0
    i = 0
    h = int(32 * 1.5)
    w = 32
    while True:
        end = start + h * w
        if end >= size:
            break
        img_buf = imgs[start:end]
        img_i = np.ndarray((h, w), dtype=np.uint8, buffer=img_buf)
        print(img_i.max())
        cv2.imwrite(f"roi_{i:02d}.png", img_i)
        i += 1
        start = end
