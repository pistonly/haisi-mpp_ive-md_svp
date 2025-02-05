from haisi_md import MD
from opencv_onnx_yolo import YOLO
from pathlib import Path
import cv2
import numpy as np
import subprocess

default_onnx = "/home/liuyang/Documents/YOLO/yolov8_scripts/yolov8-air_little_obj_roi/yolov8n-32_/weights/best.onnx"
def img_generator(videoPath_or_imgDir):
    if Path(videoPath_or_imgDir).is_dir():
        img_list = sorted([str(img) for img in Path(videoPath_or_imgDir).iterdir() if img.name.endswith(".jpg")])
        # 返回iterator
        for img_path in img_list:
            img = cv2.imread(img_path)
            yield(img)
    else:
        capture = cv2.VideoCapture(videoPath_or_imgDir)
        while True:
            flag, frame = capture.read()
            if flag:
                yield(frame)
            else:
                break

def xywh_to_xyxy(xywhs, scale=4):
    # scale for MD
    if len(xywhs) < 1:
        return []
    xywhs = np.array(xywhs).astype(np.float64) * scale
    xyxys = np.empty_like(xywhs)
    xyxys[:, 0] = xywhs[:, 0] - xywhs[:, 2] / 2
    xyxys[:, 1] = xywhs[:, 1] - xywhs[:, 3] / 2
    xyxys[:, 2] = xywhs[:, 0] + xywhs[:, 2] / 2
    xyxys[:, 3] = xywhs[:, 1] + xywhs[:, 3] / 2
    return xyxys

def cut_image_roi(img, xyxys):
    H, W = img.shape[0:2]
    for i, (x0, y0, x1, y1) in enumerate(xyxys):
        img_cut = np.zeros((32, 32, 3), dtype=np.uint8)
        x = int((x0 + x1) * 0.5)
        y = int((y0 + y1) * 0.5)
        # cut 32x32 region around center
        x_cut0 = int(np.clip(x - 16, 0, W-1))
        y_cut0 = int(np.clip(y - 16, 0, H-1))
        x_cut1 = int(np.clip(x + 16, 0, W-1))
        y_cut1 = int(np.clip(y + 16, 0, H-1))
        try:
            img_cut_tmp = img[y_cut0:y_cut1, x_cut0:x_cut1]
            _h, _w = img_cut_tmp.shape[0:2]
            img_cut[:_h, :_w] = img_cut_tmp
        except:
            import pdb; pdb.set_trace()

def get_merged_img(img, bboxes):
    xyxys = xywh_to_xyxy(bboxes, scale=4)
    H, W = img.shape[:2]
    merged_img = np.zeros((640, 640, 3), dtype=np.uint8)
    cut_img_num_per_row = 640 // 32
    cut_img_max_row = 640 // 32
    toplefts = {}
    for i, (x0, y0, x1, y1) in enumerate(xyxys):
        img_cut = np.zeros((32, 32, 3), dtype=np.uint8)
        x = int((x0 + x1) * 0.5)
        y = int((y0 + y1) * 0.5)
        # cut 32x32 region around center
        x_cut0 = int(np.clip(x - 16, 0, W-1))
        y_cut0 = int(np.clip(y - 16, 0, H-1))
        x_cut1 = int(np.clip(x + 16, 0, W-1))
        y_cut1 = int(np.clip(y + 16, 0, H-1))
        try:
            img_cut_tmp = img[y_cut0:y_cut1, x_cut0:x_cut1]
            _h, _w = img_cut_tmp.shape[0:2]
            img_cut[:_h, :_w] = img_cut_tmp
        except:
            import pdb; pdb.set_trace()

        cut_img_row = i // cut_img_num_per_row
        cut_img_col = i % cut_img_num_per_row
        if cut_img_row < cut_img_max_row:
            cut_img_x0 = cut_img_col * 32
            cut_img_y0 = cut_img_row * 32
            cut_img_h, cut_img_w = img_cut.shape[:2]
            merged_img[cut_img_y0:cut_img_y0 + cut_img_h,
                       cut_img_x0:cut_img_x0 + cut_img_w] = img_cut
            toplefts[i] = [x_cut0, y_cut0]
    return merged_img, toplefts


def main(videoPath_or_imgDir, onnx_model=default_onnx, conf=0.25, result_dir="merged_tmp"):
    result_dir_p = Path(result_dir)
    if result_dir_p.is_dir():
        return

    window_name = "md-yolo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_diff = MD()
    yolo = YOLO(onnx_model, classes=["0"], conf=conf)

    img_gen = img_generator(videoPath_or_imgDir)
    stop = False
    img_id = 0

    result_dir_p.mkdir(exist_ok=False)

    for img in img_gen:
        number_label, labels, img_foreground, bboxes = frame_diff.process(img)
        merged_img, toplefts = get_merged_img(img, bboxes)

        # cv2.imwrite(str(result_dir_p / f"merged_{img_id}.jpg"), merged_img)


        decs, drawed_img = yolo.process_one_image(merged_img)

        for dec in decs:
            x0, y0, w, h = dec['box']
            if max(w, h) > 32:
                continue
            x_c = int(x0 + w / 2)
            y_c = int(y0 + h / 2)
            grid_x = x_c // 32
            grid_y = y_c // 32
            grid_id = grid_y * 20 + grid_x
            tl = toplefts[grid_id]
            tl_tmp = (grid_x * 32, grid_y * 32)
            x0 = x0 - tl_tmp[0] + tl[0]
            y0 = y0 - tl_tmp[1] + tl[1]
            x1 = int(x0 + w)
            y1 = int(y0 + h)
            x0 = int(x0)
            y0 = int(y0)
            color = (0, 255, 0)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            # cv2.putText(img, "target", (x0 - 10, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(str(result_dir_p / f"frame_{img_id}.jpg"), img)
        cv2.setWindowTitle(window_name, f"md-yolo: {result_dir}:{img_id}")
        img_id += 1
        # while True:
        #     cv2.imshow(window_name, img)
        #     key = cv2.waitKey(10) & 0xFF

        #     if key == 27:
        #         stop = True
        #         break
        #     if key == 32:
        #         break

        #     if key == ord("s"):
        #         return True, videoPath_or_imgDir
        if stop:
            break
    cv2.destroyAllWindows()
    return False, videoPath_or_imgDir


if __name__ == "__main__":
    # video_path = "/home/liuyang/Documents/qiyuan_jiaojie/tools/results/guanting_dji_705_0.hevc"
    # video_path = "rtsp://172.23.24.52:8554/test"
    video_path = "/home/liuyang/Documents/qiyuan_jiaojie/data/yanshou/20241022104514376-11-1-main"
    # video_path = "/home/liuyang/Documents/qiyuan_jiaojie/tools/202410191444_0.hevc"
    # onnx_model = "/home/liuyang/Documents/YOLO/yolov8_scripts/yolov8-air_little_obj_roi/yolov8n-32_yanshou_10-percent_2/weights/best.onnx"
    onnx_model = "/home/liuyang/Documents/YOLO/yolov8_scripts/yolov8-air_little_obj_roi/yolov8n-32_/weights/best.onnx"
    # onnx_model = "/home/liuyang/Documents/qiyuan_jiaojie/nnn_om_convert/models/yolov8n_air-little-obj_32-roi_yanshou-4video-10percent.onnx"
    main(video_path, onnx_model)


