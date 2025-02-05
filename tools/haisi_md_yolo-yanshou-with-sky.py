from haisi_md import MD
from opencv_onnx_yolo import YOLO
from pathlib import Path
import cv2
import numpy as np
import subprocess
import random 

default_onnx = "/home/liuyang/Documents/YOLO/yolov8_scripts/yolov8-air_little_obj_roi/yolov8n-32_/weights/best.onnx"
def img_generator(videoPath_or_imgDir, start_frame=900):
    if Path(videoPath_or_imgDir).is_dir():
        img_list = sorted([str(img) for img in Path(videoPath_or_imgDir).iterdir() if img.name.endswith(".jpg")])
        # 返回iterator
        for i, img_f in enumerate(img_list):
            if i < start_frame:
                continue
            img = cv2.imread(img_f)
            yield(i, img)
    else:
        capture = cv2.VideoCapture(videoPath_or_imgDir)
        i = 0
        while True:
            flag, frame = capture.read()
            if flag:
                yield(i, frame)
            else:
                break
            i += 1

def xywh_to_xyxy(xywhs, scale=4):
    # xywh: x0, y0, w, h
    # scale for MD
    if len(xywhs) < 1:
        return []
    xywhs = np.array(xywhs).astype(np.float64) * scale
    xyxys = np.empty_like(xywhs)
    xyxys[:, 0] = xywhs[:, 0]
    xyxys[:, 1] = xywhs[:, 1]
    xyxys[:, 2] = xywhs[:, 0] + xywhs[:, 2]
    xyxys[:, 3] = xywhs[:, 1] + xywhs[:, 3]
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
        # NOTE debug: + 10
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

def plot(img_orig, decs, toplefts, roi=None):
    img = img_orig.copy()
    if roi:
        roi = np.array(roi)
        roi = roi.reshape((-1, 4))
    for dec in decs:
        x0, y0, w, h = dec['box']
        if max(w, h) > 32:
            continue
        x_c = int(x0 + w / 2)
        y_c = int(y0 + h / 2)
        grid_x = x_c // 32
        grid_y = y_c // 32
        grid_id = grid_y * 20 + grid_x
        if grid_id not in toplefts:
            continue
        tl = toplefts[grid_id]
        tl_tmp = (grid_x * 32, grid_y * 32)
        x0 = x0 - tl_tmp[0] + tl[0]
        y0 = y0 - tl_tmp[1] + tl[1]
        x1 = int(x0 + w)
        y1 = int(y0 + h)
        x0 = int(x0)
        y0 = int(y0)
        if roi:
            in_roi = False
            for roi_i in roi:
                x0_r, y0_r, w, h = roi_i
                x1_r, y1_r = x0_r + w, y0_r + h
                x_c = (x0 + x1) / 2
                y_c = (y0 + y1) / 2
                if (x_c >= x0_r) and (x_c <= x1_r) and (y_c >= y0_r) and (y_c <= y1_r):
                    in_roi = True
                    break
            if not in_roi:
                continue
        color = (0, 255, 0)
        # cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        # cv2.putText(img, f"{dec['confidence']:.2f}", (x0 - 10, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img, f"{dec['confidence']:.2f}", (x0 - 10, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def image_letter_box(origin_image:np.ndarray, output_size=640):
    h, w = origin_image.shape[:2]
    ratio = min(output_size / h, output_size / w)
    h_target, w_target = int(h * ratio), int(w * ratio)
    offset_x, offset_y = (output_size - w_target) // 2, (output_size - h_target) // 2
    output_img = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    output_img[offset_y:offset_y + h_target, offset_x: offset_x + w_target] = cv2.resize(origin_image, (w_target, h_target))
    return output_img, offset_x, offset_y, ratio

def main(videoPath_or_imgDir, onnx_model=default_onnx, start_frame=0, csv_list=[], conf=0.1, saved_video="video.mp4",
         sky_model=""):
    window_name = "md-yolo"
    window_name_md = "md"
    window_name_md_merged = "md_merged"
    window_sky = "sky"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_md, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_md_merged, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_sky, cv2.WINDOW_NORMAL)

    Id_ts = None
    if len(csv_list):
        Id_ts = [Path(f).stem.split("-")[-1].split("_") for f in csv_list]
        Id_ts = {int(Id): ts for Id, ts in Id_ts}

    frame_diff = MD()
    yolo = YOLO(onnx_model, classes=["0"], conf=conf)
    yolo_sky = YOLO(sky_model, classes=["0", "1"], conf=0.1)

    target_dir = Path(videoPath_or_imgDir).parent / f"{Path(videoPath_or_imgDir).name}_results"
    target_dir.mkdir(exist_ok=True, parents=True)
    # target_dir_2 = Path(videoPath_or_imgDir).parent / f"{Path(videoPath_or_imgDir).name}_md_results"
    # target_dir_2.mkdir(exist_ok=True, parents=True)
    img_gen = img_generator(videoPath_or_imgDir, start_frame)
    stop = False
    result_dir = Path("./merged_tmp")
    result_dir.mkdir(exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = 2160, 3840

    # video = cv2.VideoWriter(saved_video, fourcc, 25, (width, height))
    for img_id, img in img_gen:
        # sky detection
        img_sky, sky_offset_x, sky_offset_y, sky_scale = image_letter_box(img, 640)
        decs_sky, drawed_img_sky = yolo_sky.process_one_image(img_sky)
        bbox_sky = []
        for decs_i in decs_sky:
            bbox_sky.append(decs_i['box'])
        if len(bbox_sky):
            bbox_sky = np.array(bbox_sky)
            bbox_sky /= sky_scale 
            bbox_sky[:, 0] -= sky_offset_x / sky_scale
            bbox_sky[:, 1] -= sky_offset_y / sky_scale

        number_label, labels, img_foreground, bboxes = frame_diff.process(img)
        merged_img, toplefts = get_merged_img(img, bboxes)

        img_foreground_upscaled = cv2.resize(img_foreground, (3840, 2160))
        md_result_image = cv2.addWeighted(img, 0.8, img_foreground_upscaled, 0.2, 0)

        if not len(bboxes):
            continue

        cv2.imwrite(str(result_dir / f"merged_{img_id}.jpg"), merged_img)

        delete = False
        # if Id_ts and img_id in Id_ts:
        #     cv2.setWindowTitle(window_name, f"{window_name} TS: {Id_ts[img_id]}")
        #     cv2.setWindowTitle(window_name_md, f"{window_name} TS: {Id_ts[img_id]}")
        # else:
        #     delete = True


        decs, drawed_img = yolo.process_one_image(merged_img)
        if delete:
            decs = []

        roi = None
        img_display = plot(img, decs, toplefts, roi)
        cv2.setWindowTitle(window_name, f"frame_id: {img_id}")

        # video.write(img)
        # cv2.imwrite(str(target_dir / f"frame_{img_id}.jpg"), img_display)
        # cv2.imshow(window_name, img_display)
        cv2.waitKey(5)
        while True:
            cv2.imshow(window_name, img_display)
            cv2.imshow(window_name_md, img_foreground)
            cv2.imshow(window_name_md_merged, md_result_image)
            cv2.imshow(window_sky, drawed_img_sky)
            key = cv2.waitKey(10) & 0xFF

            if key == 27:
                stop = True
                break
            if key == 32:
                break

            if key == ord("r"):
                roi = list(cv2.selectROI(window_name, img_display, fromCenter=False))
                img_display = plot(img, decs, toplefts, roi)

            if key == ord("s"): # save
                cv2.imwrite(str(target_dir / f"frame_{img_id}.jpg"), img_display)
                # cv2.imwrite(str(target_dir_2 / f"frame_{img_id}.jpg"), md_result_image)
                break
        if stop:
            break
    # video.release()
    cv2.destroyAllWindows()
    return False, videoPath_or_imgDir


if __name__ == "__main__":
    # video_path = "/home/liuyang/Documents/qiyuan_jiaojie/tools/results/guanting_dji_705_0.hevc"
    # video_path = "/home/liuyang/Downloads/als/1214pm_images/20241214151008180-11-1-main/"
    # video_path = "/home/liuyang/Downloads/als/1214pm_images/20241214151408307-11-1-main"
    # video_path = "/home/liuyang/Downloads/als/1215pm_images/20241215133254860-11-1-main"
    # start_frame, video_path = 314, "/home/liuyang/Downloads/als/1215pm_images/20241215133254860-11-1-main"
    # start_frame, video_path = 157, "/home/liuyang/Downloads/als/1215pm_images/20241215133254860-11-1-main-2f"
    # start_frame, video_path = 78, "/home/liuyang/Downloads/als/1215pm_images/20241215133254860-11-1-main-4f"
    # start_frame, video_path = 2538, "/home/liuyang/Downloads/als/1215pm_images/20241215133254860-11-1-main"
    # start_frame, video_path = 0, "/home/liuyang/Downloads/als/1215pm_images/20241215133854943-11-1-main"
    # start_frame, video_path = 2200, "/home/liuyang/Downloads/als/1215pm_images/20241215133854943-11-1-main"
    # start_frame, video_path = 1480, "/home/liuyang/Downloads/als/1215pm_images/20241215134054996-11-1-main/"
    # start_frame, video_path = 878, "/home/liuyang/Downloads/als/1215pm_images/20241215134255043-11-1-main/"
    # start_frame, video_path = 976, "/home/liuyang/Downloads/als/1215pm_images/20241215134455093-11-1-main/"
    # start_frame, video_path = 1533, "/home/liuyang/Downloads/als/1215pm_images/20241215134655160-11-1-main/"
    # start_frame, video_path = 740, "/home/liuyang/Downloads/als/1215pm_images/20241215134855203-11-1-main/"
    # start_frame, video_path = 1450, "/home/liuyang/Downloads/als/1215pm_images/20241215134855203-11-1-main/"
    # start_frame, video_path = 2943, "/home/liuyang/Downloads/als/1215pm_images/20241215134855203-11-1-main/"
    # start_frame, video_path = 0, "/home/liuyang/Downloads/als/1215pm_images/20241215135055247-11-1-main/"
    # start_frame, video_path = 650, "/home/liuyang/Downloads/als/1215pm_images/20241215135055247-11-1-main/"
    # start_frame, video_path = 2875, "/home/liuyang/Downloads/als/1215pm_images/20241215135055247-11-1-main/"
    # start_frame, video_path = 2158, "/home/liuyang/Downloads/als/1215pm_images/20241215135055247-11-1-main/"
    # start_frame, video_path = 1312, "/home/liuyang/Downloads/als/1215pm_images/20241215135255273-11-1-main/"
    # xfd
    start_frame, video_path = 277, "/home/liuyang/Downloads/als/1214pm_images/20241214151008180-11-1-main/"
    # start_frame, video_path = 800, "/home/liuyang/Downloads/als/1214pm_images/20241214151008180-11-1-main/"
    # start_frame, video_path = 0, "/home/liuyang/Downloads/als/1214pm_images/20241214151208270-11-1-main/"
    # start_frame, video_path = 2209, "/home/liuyang/Downloads/als/1214pm_images/20241214151008180-11-1-main/"
    # start_frame, video_path = 0, "/home/liuyang/Downloads/als/1214pm_images/20241214151408307-11-1-main/"
    # start_frame, video_path = 2150, "/home/liuyang/Downloads/als/1214pm_images/20241214151408307-11-1-main/"
    # start_frame, video_path = 1468, "/home/liuyang/Downloads/als/1214pm_images/20241214151408307-11-1-main/"
    # start_frame, video_path = 215, "/home/liuyang/Downloads/als/1214pm_images/20241214151608343-11-1-main/"
    # start_frame, video_path = 1225, "/home/liuyang/Downloads/als/1214pm_images/20241214151608343-11-1-main/"
    # start_frame, video_path = 1397, "/home/liuyang/Downloads/als/1214pm_images/20241214151808397-11-1-main/"
    # start_frame, video_path = 97, "/home/liuyang/Downloads/als/1214pm_images/20241214152041110-11-1-main/"
    # start_frame, video_path = 175, "/home/liuyang/Downloads/als/1214pm_images/20241214152041110-11-1-main/"
    # start_frame, video_path = 900, "/home/liuyang/Downloads/als/1215pm_images/20241215133654936-11-1-main/"
    # start_frame, video_path = 1543, "/media/liuyang/WD_BLACK/als_20241216_images/20241216101114810-20-2-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216101114823-11-1-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216101314863-20-2-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216101314923-11-1-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216101514903-20-2-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216101514943-11-1-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216101714940-20-2-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216101714990-11-1-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216101914983-20-2-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216101915020-11-1-main/"
    # good
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216102115026-20-2-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216102115056-11-1-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216102315093-20-2-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216102315100-11-1-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216102515127-20-2-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216102515133-11-1-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216102715140-20-2-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216102715150-11-1-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216102915153-20-2-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216102915193-11-1-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216103115197-20-2-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216103115223-11-1-main/"
    # video_path = "/home/liuyang/Documents/qiyuan_jiaojie/tools/results/sod4bird-35_36.hevc"
    # video_path = "/home/liuyang/Documents/qiyuan_jiaojie/tools/results/sod4bird-35_42.hevc"
    # video_path = "/home/liuyang/Documents/qiyuan_jiaojie/tools/results/20241019144645023-11-1-main.h265"
    # video_path = "/home/liuyang/Documents/qiyuan_jiaojie/tools/results/sod4bird-35_49.hevc"
    # video_path = "/home/liuyang/Documents/qiyuan_jiaojie/tools/results/sod4bird-35_55.hevc"
    # video_path = "/home/liuyang/Documents/qiyuan_jiaojie/tools/results/20240208_5.hevc"
    # video_path = "/home/liuyang/Documents/qiyuan_jiaojie/tools/results/20241020150105793-11-1-main.h265"
    # video_path = "rtsp://172.23.24.52:8554/test"
    # video_path = "/home/liuyang/Documents/qiyuan_jiaojie/tools/202410191444_0.hevc"
    # onnx_model = "/home/liuyang/Documents/YOLO/yolov8_scripts/yolov8-air_little_obj_roi/yolov8n-32_yanshou_10-percent_2/weights/best.onnx"
    # onnx_model = "/home/liuyang/Documents/YOLO/yolov8_scripts/yolov8-air_little_obj_roi/yolov8n-32_/weights/best.onnx"
    sky_model = "/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj/sky/weights/best.onnx"
    # onnx_model = "/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj_roi/als-1214_sz-32_freeze-10_lr0-0005_2/weights/best.onnx"
    # onnx_model = "/home/liuyang/Documents/qiyuan_jiaojie/nnn_om_convert/models/yolov8n_air-little-obj_32-roi_yanshou-4video-10percent.onnx"
    # onnx_model = "/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj_roi/als-1214_sz-32_lr0-0005_/weights/best.onnx"
    # onnx_model = "/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj_roi/als-1214_sz-32_lr0-001_/weights/best.onnx"
    # onnx_model = "/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj_roi/als-1214_sz-32_lr0-001-add_some_images_/weights/best.onnx"
    onnx_model = "/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj_roi/als-1216_sz-32_lr0-001-add_some_images_/weights/best.onnx"
    main(video_path, onnx_model, start_frame, sky_model=sky_model)


