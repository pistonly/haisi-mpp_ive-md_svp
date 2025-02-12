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
        # Return iterator
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

# Optional: update this function to support variable patch_size
def cut_image_roi(img, xyxys, patch_size=32):
    H, W = img.shape[0:2]
    half_patch = patch_size // 2
    for i, (x0, y0, x1, y1) in enumerate(xyxys):
        img_cut = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        x = int((x0 + x1) * 0.5)
        y = int((y0 + y1) * 0.5)
        # Cut patch_size x patch_size region around center
        x_cut0 = int(np.clip(x - half_patch, 0, W-1))
        y_cut0 = int(np.clip(y - half_patch, 0, H-1))
        x_cut1 = int(np.clip(x + half_patch, 0, W-1))
        y_cut1 = int(np.clip(y + half_patch, 0, H-1))
        try:
            img_cut_tmp = img[y_cut0:y_cut1, x_cut0:x_cut1]
            _h, _w = img_cut_tmp.shape[0:2]
            img_cut[:_h, :_w] = img_cut_tmp
        except:
            import pdb; pdb.set_trace()

def get_merged_img(img, bboxes, patch_size=32):
    """
    Merge small patches extracted from the image into a 640x640 image.
    - patch_size: size of each small patch (e.g., 32, 16, etc.)
    - The number of patches per row/column = 640 // patch_size.
    """
    xyxys = xywh_to_xyxy(bboxes, scale=4)
    H, W = img.shape[:2]
    merged_img = np.zeros((640, 640, 3), dtype=np.uint8)
    half_patch = patch_size // 2
    num_per_row = 640 // patch_size  # number of patches per row
    max_rows = 640 // patch_size      # maximum number of rows
    toplefts = {}
    for i, (x0, y0, x1, y1) in enumerate(xyxys):
        img_cut = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        # Get center of the bbox
        x = int((x0 + x1) * 0.5)
        y = int((y0 + y1) * 0.5)
        # Cut patch_size x patch_size region around center
        x_cut0 = int(np.clip(x - half_patch, 0, W-1))
        y_cut0 = int(np.clip(y - half_patch, 0, H-1))
        x_cut1 = int(np.clip(x + half_patch, 0, W-1))
        y_cut1 = int(np.clip(y + half_patch, 0, H-1))
        try:
            img_cut_tmp = img[y_cut0:y_cut1, x_cut0:x_cut1]
            _h, _w = img_cut_tmp.shape[0:2]
            img_cut[:_h, :_w] = img_cut_tmp
        except:
            import pdb; pdb.set_trace()

        # Determine the patch position in the merged image
        patch_row = i // num_per_row
        patch_col = i % num_per_row
        if patch_row < max_rows:
            x_offset = patch_col * patch_size
            y_offset = patch_row * patch_size
            merged_img[y_offset:y_offset + patch_size, x_offset:x_offset + patch_size] = img_cut
            toplefts[i] = [x_cut0, y_cut0]
    return merged_img, toplefts

def plot(img_orig, decs, toplefts, roi=None, patch_size=32):
    """
    Plot detection results by mapping coordinates from the merged image patches back to the original image.
    - patch_size: the size of each small patch.
    - It computes grid indices based on patch_size, where the number of patches per row = 640 // patch_size.
    """
    img = img_orig.copy()
    img_h, img_w = img.shape[:2]
    yolo_labels = []
    num_per_row = 640 // patch_size
    if roi is not None:
        roi = np.array(roi)
        roi = roi.reshape((-1, 4))
    for dec in decs:
        x0, y0, w, h = dec['box']
        if max(w, h) > patch_size:
            continue
        x_c = int(x0 + w / 2)
        y_c = int(y0 + h / 2)
        grid_x = x_c // patch_size
        grid_y = y_c // patch_size
        grid_id = grid_y * num_per_row + grid_x
        if grid_id not in toplefts:
            continue
        tl = toplefts[grid_id]
        tl_tmp = (grid_x * patch_size, grid_y * patch_size)
        # Adjust detection coordinates back to the original image
        x0_adj = int(x0 - tl_tmp[0] + tl[0])
        y0_adj = int(y0 - tl_tmp[1] + tl[1])
        x1_adj = int(x0_adj + w)
        y1_adj = int(y0_adj + h)
        if roi is not None:
            in_roi = False
            for roi_i in roi:
                x0_r, y0_r, rw, rh = roi_i
                x1_r, y1_r = x0_r + rw, y0_r + rh
                x_center = (x0_adj + x1_adj) / 2
                y_center = (y0_adj + y1_adj) / 2
                if (x_center >= x0_r) and (x_center <= x1_r) and (y_center >= y0_r) and (y_center <= y1_r):
                    in_roi = True
                    break
            if not in_roi:
                continue
        color = (0, 255, 0)
        cv2.rectangle(img, (x0_adj, y0_adj), (x1_adj, y1_adj), color, 2)
        cv2.putText(img, f"{dec['confidence']:.2f}", (x0_adj - 10, y0_adj - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        x_center, y_center = (x0_adj + x1_adj) / 2, (y0_adj + y1_adj) / 2
        yolo_labels.append([x_center / img_w, y_center / img_h, w / img_w, h / img_h])
    return img, yolo_labels

def image_letter_box(origin_image:np.ndarray, output_size=640):
    h, w = origin_image.shape[:2]
    ratio = min(output_size / h, output_size / w)
    h_target, w_target = int(h * ratio), int(w * ratio)
    offset_x, offset_y = (output_size - w_target) // 2, (output_size - h_target) // 2
    output_img = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    output_img[offset_y:offset_y + h_target, offset_x: offset_x + w_target] = cv2.resize(origin_image, (w_target, h_target))
    return output_img, offset_x, offset_y, ratio

def main(videoPath_or_imgDir, onnx_model=default_onnx, start_frame=0, csv_list=[], conf=0.1, saved_video="video.mp4",
         sky_model="", patch_size=32):
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

    target_dir = Path(videoPath_or_imgDir).parent / f"{Path(videoPath_or_imgDir).name}_yolo"
    target_image_dir = target_dir / "images"
    target_label_dir = target_dir / "labels"
    target_image_dir.mkdir(exist_ok=True, parents=True)
    target_label_dir.mkdir(exist_ok=True, parents=True)
    img_gen = img_generator(videoPath_or_imgDir, start_frame)
    stop = False
    result_dir = Path("./merged_tmp")
    result_dir.mkdir(exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = 2160, 3840

    for img_id, img in img_gen:
        # Sky detection
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
        # Pass patch_size parameter to get_merged_img
        merged_img, toplefts = get_merged_img(img, bboxes, patch_size=patch_size)

        img_foreground_upscaled = cv2.resize(img_foreground, (3840, 2160))
        md_result_image = cv2.addWeighted(img, 0.8, img_foreground_upscaled, 0.2, 0)

        if not len(bboxes):
            continue

        cv2.imwrite(str(result_dir / f"merged_{img_id}.jpg"), merged_img)

        delete = False

        decs, drawed_img = yolo.process_one_image(merged_img)

        # Only keep max confidence bbox (最多保留两个)
        if len(decs):
            decs = sorted(decs, key=lambda x: x['confidence'], reverse=True)
            decs = decs[0:2]

        if delete:
            decs = []

        roi = None
        # Pass patch_size parameter to plot for coordinate mapping
        img_display, boxes = plot(img, decs, toplefts, roi, patch_size=patch_size)
        cv2.setWindowTitle(window_name, f"frame_id: {img_id}")

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
                img_display, boxes = plot(img, decs, toplefts, roi, patch_size=patch_size)

            if key == ord("s"):  # Save
                if len(boxes):
                    cv2.imwrite(str(target_dir / "images" / f"frame_{img_id}.jpg"), img)
                    with open(str(target_dir / "labels" / f"frame_{img_id}.txt"), "w") as label_f:
                        for box in boxes:
                            label_f.write(f"0 {box[0]:.4f} {box[1]:.4f} {box[2]:.4f} {box[3]:.4f}\n")
                else:
                    print('skip empty image')
                break
        if stop:
            break
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
    # start_frame, video_path = 277, "/home/liuyang/Downloads/als/1214pm_images/20241214151008180-11-1-main/"
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
    # start_frame, video_path = 2450, "/media/liuyang/WD_BLACK/als_20241216_images/20241216101514903-20-2-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216101514943-11-1-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216101714940-20-2-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216101714990-11-1-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216101914983-20-2-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216101915020-11-1-main/"
    # good
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216102115026-20-2-main/"
    # start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216102115056-11-1-main/"
    start_frame, video_path = 0, "/media/liuyang/WD_BLACK/als_20241216_images/20241216102315093-20-2-main/"
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
    sky_model = "/home/liuyang/Documents/YOLO/yolov8_scripts/runs/runs_yolov8-air_little_obj/sky/weights/best.onnx"
    # onnx_model = "/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj_roi/als-1214_sz-32_freeze-10_lr0-0005_2/weights/best.onnx"
    # onnx_model = "/home/liuyang/Documents/qiyuan_jiaojie/nnn_om_convert/models/yolov8n_air-little-obj_32-roi_yanshou-4video-10percent.onnx"
    # onnx_model = "/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj_roi/als-1214_sz-32_lr0-0005_/weights/best.onnx"
    # onnx_model = "/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj_roi/als-1214_sz-32_lr0-001_/weights/best.onnx"
    # onnx_model = "/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj_roi/als-1214_sz-32_lr0-001-add_some_images_/weights/best.onnx"
    onnx_model = "/home/liuyang/Documents/YOLO/yolov8_scripts/runs/runs_yolov8-air_little_obj_roi/als-1216_sz-32_lr0-001-add_some_images_/weights/best.onnx"

    # You can adjust patch_size as needed, for example patch_size=32, patch_size=16, etc.
    main(video_path, onnx_model, start_frame, sky_model=sky_model, patch_size=64)



