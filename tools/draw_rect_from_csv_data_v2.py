import cv2
import numpy as np
from pathlib import Path
import struct
import time
import glob
import pandas as pd
from haisi_md_yolo import main as main_haisi


url = "rtsp://172.23.24.52:8554/test"
camera_id = 121
# csv_data_dir = Path("/home/liuyang/Documents/haisi/ai-sd3403/ai-sd3403/test/test_tcp/x86/build/")
csv_data_dir = Path("/home/liuyang/Documents/tmp/md/guanting_dji_705_20")
saved_video = "sod4bird-35_54.mp4"


def deserialize_v2(buffer):
    dec = []
    len = struct.unpack_from("I", buffer, 0)[0]
    cameraId = struct.unpack_from("B", buffer, 4)[0]
    timestamp = struct.unpack_from("Q", buffer, 5)[0]
    if len > 13:
        dec_num = (len - 13) // 24
        dec = np.ndarray((dec_num, 6), dtype=np.float32, buffer=buffer[13:])
    # print(len, cameraId, timestamp, dec)
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
        if conf < 0.01:
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

def draw_rec_xyxy(image, decs):
    for x0, y0, x1, y1, conf, cls in decs:
        x0, y0, x1, y1 = 2 * int(x0), 2 * int(y0), 2 * int(x1), 2 * int(y1)
        try:
            if int(cls) == 0:
                cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 255), 2)
            else:
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
        except:
            import pdb; pdb.set_trace()
    return image


def main(url, save):
    csv_list = [f for f in csv_data_dir.iterdir() if str(f).endswith(".csv")]
    main_haisi(url, csv_list=csv_list, saved_video=saved_video)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=url, help="rtsp-url")
    parser.add_argument("--save", action="store_true", help="save drew frames")
    args = parser.parse_args()
    main(args.url, args.save)
