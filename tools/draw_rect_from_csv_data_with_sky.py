import cv2
import numpy as np
from pathlib import Path
import struct
import time
import glob
import pandas as pd


url = "rtsp://172.23.24.52:8554/test"
camera_id = 121
# tcp_data_dir = Path("/home/liuyang/Documents/haisi/ai-sd3403/ai-sd3403/test/test_tcp/x86/build/")
tcp_data_dir = Path("/home/liuyang/Documents/tmp/tmp_sky_2")


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
            elif int(cls) == 10:
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
            else:
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
        except:
            import pdb; pdb.set_trace()
    return image

def draw_sky(image, decs, offset_x, offset_y, scale):
    for x0, y0, x1, y1, conf, cls in decs:
        x0 = int((x0 - offset_x) / scale)
        y0 = int((y0 - offset_y) / scale)
        x1 = int((x1 - offset_x) / scale)
        y1 = int((y1 - offset_y) / scale)
        try:
            if int(cls) == 0:
                cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 3)
            elif int(cls) == 1:
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 3)
            else:
                print("error")
        except:
            import pdb; pdb.set_trace()
    return image

def main(url, save):
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Cannot open RTSP stream.")
        exit()

    window_name = 'RTSP Stream'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    output_dir = Path("./tmp")
    if save:
        output_dir.mkdir(exist_ok=True)

    img_id = 0
    stop = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame (stream end?). Exiting ...")
            break

        glob_res = glob.glob(str(tcp_data_dir / f"decs_camera-{camera_id}_image-{img_id + 1:06d}*.csv"))
        decs = []
        if len(glob_res):
            decs = pd.read_csv(str(tcp_data_dir / glob_res[0]), header=None).values

        frame = draw_rec_xyxy(frame, decs)

        # draw sky region
        glob_res = glob.glob(str(tcp_data_dir / f"sky_result_frame_id_{img_id + 1}*.csv"))
        decs = []
        if len(glob_res):
            offset_x, offset_y, scale = [float(x) for x in glob_res[0][0:-4].split("_")[-3:]]
            try:
                decs = pd.read_csv(str(tcp_data_dir/ glob_res[0]), header=None).values
            except:
                print(glob_res[0])
                decs = []

        frame = draw_sky(frame, decs, offset_x, offset_y, scale)

        img_id += 1

        cv2.setWindowTitle(window_name, f'RTSP Stream - img_id: {img_id}')
        while True:
            cv2.imshow(window_name, frame)
            # if save:
            #     cv2.imwrite(str(output_dir / f"frame_{img_id:06d}.jpg"), frame)

            key = cv2.waitKey(10)
            if key == ord('q'):
                stop = True
                break

            if key == 32:
                break
        if stop:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=url, help="rtsp-url")
    parser.add_argument("--save", action="store_true", help="save drew frames")
    args = parser.parse_args()
    main(args.url, args.save)
