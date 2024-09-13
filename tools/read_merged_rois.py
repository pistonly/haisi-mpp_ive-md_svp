import numpy as np
import cv2
from pathlib import Path

image_dir = Path("/home/liuyang/Documents/haisi/ai-sd3403/ai-sd3403/mpp_ive-md_svp/out/")
img_fns = [fn for fn in image_dir.iterdir() if fn.name.startswith("merged_roi") and fn.name.endswith(".bin")]
img_fns = sorted(img_fns
                 )
img_H, img_W = 640, 640
for fn in img_fns:
    with open(str(fn), "rb") as f:
        img = f.read()
        size = len(img)
        w = img_W
        h = size // w
        img = np.ndarray((h, w), dtype=np.uint8, buffer=img)
        # import pdb; pdb.set_trace()
        bgr = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV21)
        fn_out = fn.with_suffix(".png")
        cv2.imwrite(str(fn_out), bgr)
        # img = img[:w, :]
        # fn_out = fn.with_suffix(".png")
        # print(img.shape)
        # cv2.imwrite(str(fn_out), img)


