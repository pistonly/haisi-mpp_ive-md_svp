import cv2
import numpy as np
import time


def ive_sub(img0, img1, mode='abs')-> np.array:
    if mode == 'abs':
        res = cv2.absdiff(img0, img1)
        return res

def ive_threshold(img, mode='binary', low_threshold=15, min_val=0, max_val=255):
    if mode == "binary":
        if min_val == 0:
            ret, img = cv2.threshold(img, low_threshold, max_val, cv2.THRESH_BINARY)
        else:
            max_val_tmp = max_val - min_val
            ret, img = cv2.threshold(img, low_threshold, max_val_tmp, cv2.THRESH_BINARY)
            img += min_val
        return ret, img
    else:
        None, None

def ive_sad(img0, img1, mode='4x4', max_val=255, min_val=0, threshold=100):
    diff = ive_sub(img0, img1, mode='abs')

    # 获取原始图像的尺寸
    height, width = diff.shape

    # 计算新图像的尺寸
    new_height = height // 4
    new_width = width // 4

    # 初始化新图像
    sad = np.zeros((new_height, new_width), dtype=np.uint8)

    reshaped_image = diff.reshape((new_height, 4, new_width, 4))
    summed_blocks = reshaped_image.sum(axis=(1, 3)).astype(np.uint16)

    _, sad = ive_threshold(summed_blocks, low_threshold=threshold, min_val=min_val, max_val=max_val)
    return sad.astype(np.uint8)

def ive_ccl(binary_image, mode="4c", init_area_threshold=0, area_step=10):
    # 查找连通区域
    num_labels, labels = cv2.connectedComponents(binary_image, 4 if mode == "4c" else 8)

    # 创建一个彩色图像来显示结果
    output_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)

    # 为每个连通区域分配一个随机颜色
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] *= 0  # 背景设为黑色

    # 使用标签图和颜色数组来生成彩色图像
    output_image = colors[labels]

    # 存储每个连通区域的边框
    bounding_boxes = []

    # 遍历每个连通区域（跳过标签0，因为它是背景）
    for label in range(1, num_labels):
        mask = (labels == label).astype(np.uint8)  # 创建当前标签的掩码
        x0, y0, w, h = cv2.boundingRect(mask)  # 计算边框, NOTE: x0, y0 not x_center, y_center
        bounding_boxes.append((x0, y0, w, h))
        # 在图像上绘制边框
        cv2.rectangle(output_image, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)

    return num_labels, labels, output_image, bounding_boxes

def ive_ccl_del(binary_image, mode="4c", init_area_threshold=0, area_step=10):
    # 查找连通区域
    num_labels, labels = cv2.connectedComponents(binary_image, 4 if mode == "4c" else 8)

    import random
    label_all = np.arange(1, num_labels)
    random.shuffle(label_all)
    label_all[1] = 1
    # 创建一个彩色图像来显示结果
    output_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)

    # 为每个连通区域分配一个随机颜色
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] *= 0  # 背景设为黑色
    colors[label_all[40:]] *= 0

    # 使用标签图和颜色数组来生成彩色图像
    output_image = colors[labels]

    # 存储每个连通区域的边框
    bounding_boxes = []

    # # 遍历每个连通区域（跳过标签0，因为它是背景）
    # for label in range(1, num_labels):
    #     mask = (labels == label).astype(np.uint8)  # 创建当前标签的掩码
    #     x0, y0, w, h = cv2.boundingRect(mask)  # 计算边框, NOTE: x0, y0 not x_center, y_center
    #     bounding_boxes.append((x0, y0, w, h))
    #     # 在图像上绘制边框
    #     cv2.rectangle(output_image, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)
    # 遍历每个连通区域（跳过标签0，因为它是背景）
    for label in label_all[0:40]:
        mask = (labels == label).astype(np.uint8)  # 创建当前标签的掩码
        x0, y0, w, h = cv2.boundingRect(mask)  # 计算边框, NOTE: x0, y0 not x_center, y_center
        bounding_boxes.append((x0, y0, w, h))
        # 在图像上绘制边框
        cv2.rectangle(output_image, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)

    return num_labels, labels, output_image, bounding_boxes

class MD:
    def __init__(self, sub_args={"mode": 'abs'},
                 threshold_args={'mode': 'binary', 'low_threshold': 15, 'min_val': 0, 'max_val': 255},
                 sad_args={'mode':'4x4', 'max_val':255, 'min_val':0, 'threshold':100},
                 ccl_args={'mode': '4c', 'init_area_threshold': 0, 'area_step': 10}) -> None:
        self.md_images = [None, None]
        self.is_first_image = True
        self.sub_args = sub_args
        self.threshold_args = threshold_args
        self.sad_args = sad_args
        self.ccl_args = ccl_args
        self.current_img_id = 0

    def process(self, img_input):
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        self.md_images[self.current_img_id] = img_input
        if self.is_first_image:
            self.is_first_image = False
            self.img_h, self.img_w = img_input.shape[0:2]
            if self.sad_args['mode'] == "4x4":
                self.out_h = self.img_h // 4
                self.out_w = self.img_w // 4
            number_labels = 0
            labels = np.zeros((self.out_h, self.out_w, 3), dtype=np.uint16)
            output_image = np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)
            bboxes = []

            self.sad_zeros = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        else:
            t0 = time.time()
            diff = ive_sub(self.md_images[self.current_img_id], self.md_images[1-self.current_img_id],
                           self.sub_args['mode'])
            t1 = time.time()
            _, diff = ive_threshold(diff, **self.threshold_args)
            t2 = time.time()
            sad = ive_sad(diff, self.sad_zeros, **self.sad_args)
            t3 = time.time()
            number_labels, labels, output_image, bboxes = ive_ccl(sad, **self.ccl_args)
            t4 = time.time()
            # print(t1 - t0, t2 - t1, t3 - t2, t4 - t3)

        self.current_img_id = 1 - self.current_img_id
        # print(number_labels)
        return number_labels, labels, output_image, bboxes

def add_bbox(image, bboxes, scale=4, frame_id=0):
    image_bac = image.copy()
    img_h, img_w = image.shape[0:2]
    roi_size = 32
    merged_image = np.zeros((640, 640, 3), dtype=np.uint8)
    roi_num_each_row = 640 / roi_size
    for i, (x, y, w, h) in enumerate(bboxes):
        if i >= 400:
            break
        x, y, w, h = scale * x, scale * y, scale * w, scale * h
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 1)
        # print(x, y, x+w, y+h)

        # cut roi
        center_x, center_y = (x + x + w) / 2, (y + y + h) / 2
        upleft_x, upleft_y = center_x - roi_size / 2, center_y - roi_size / 2
        downright_x, downright_y = center_x + roi_size / 2, center_y + roi_size / 2
        upleft_x = int(np.clip(upleft_x, 0, img_w))
        upleft_y = int(np.clip(upleft_y, 0, img_h))
        downright_x = int(np.clip(downright_x, 0, img_w))
        downright_y = int(np.clip(downright_y, 0, img_h))
        roi = image_bac[upleft_y:downright_y, upleft_x:downright_x]
        # roi = image[upleft_y:downright_y, upleft_x:downright_x]
        roi_h, roi_w = roi.shape[0:2]

        merge_x0 = int((i % roi_num_each_row) * roi_size)
        merge_y0 = int((i // roi_num_each_row) * roi_size)
        merged_image[merge_y0:(merge_y0 + roi_h), merge_x0:(merge_x0 + roi_w)] = roi

        # label
        x0, y0 = x - upleft_x, y - upleft_y
        x1, y1 = x0 + w, y0 + h
        x1 = np.clip(x1, 0, roi_w)
        y1 = np.clip(y1, 0, roi_h)
        x_c, y_c = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = x1 - x0, y1 - y0
        xn, yn, wn, hn = x_c / roi_w, y_c / roi_h, w / roi_w, h / roi_h
        label = np.array([[0, xn, yn, wn, hn]])
    #     np.savetxt(f"{frame_id:04d}_{i:03d}.txt", label)
    #     cv2.imwrite(f"{frame_id:04d}_{i:03d}.jpg", roi)
    # cv2.imwrite(f"merged_{frame_id:04d}.jpg", merged_image)
    return image

def save_one_image_to_dataset(img, bboxes, ds_dir, name_stem, scale=4):
    '''
    bboxes: [[x0, y0, w, h], ...]
    '''
    if len(bboxes) == 0:
        return 
    h, w = img.shape[:2]
    bboxes = np.array(bboxes).astype(np.float64) * scale
    bboxes[:, [0, 2]] /= w
    bboxes[:, [1, 3]] /= h
    bboxes[:, [0, 1]] += (bboxes[:, [2, 3]] / 2)
    cls = np.zeros((len(bboxes), 1))
    labels = np.hstack([cls, bboxes])
    np.savetxt(str(ds_dir / "labels"/ f"{name_stem}.txt"), labels)
    cv2.imwrite(str(ds_dir / "images"/ f"{name_stem}.jpg"), img)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import time


    argparser = argparse.ArgumentParser()
    argparser.add_argument("--image_dir", default="", type=str, help="input dir for images")
    argparser.add_argument("--threshold", default=15, type=int, help="threshold for foreground")
    argparser.add_argument("--is_video", action='store_true', help="get frames from video")
    argparser.add_argument("--from_frame", type=int, default=250, help="from which frame to save")
    argparser.add_argument("--frame_num", type=int, default=100, help="from which frame to save")
    argparser.add_argument("--dataset_dir", type=str, default="./dataset_tmp", help="output dataset dir")
    args = argparser.parse_args()

    frame_diff = MD()
    cost_times = []

    ds_dir = Path(args.dataset_dir)
    ds_image_dir = ds_dir / "images"
    ds_image_dir.mkdir(parents=True, exist_ok=True)
    ds_label_dir = ds_dir / "labels"
    ds_label_dir.mkdir(parents=True, exist_ok=True)

    cv2.namedWindow('img_input', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('img_input', 1920, 1080)
    cv2.resizeWindow('img_input', 3840, 2160)
    cv2.namedWindow("img_foreground", cv2.WINDOW_NORMAL)

    if args.is_video:
        capture = cv2.VideoCapture(args.image_dir)
        if not capture.isOpened():
            print("Error: Cannot open video.")
        else:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Total number of frames: {total_frames}")

        img_num = 0
        for i in range(total_frames):
            flag, frame = capture.read()
            if i < args.from_frame:
                continue

            if img_num >= args.frame_num:
                break

            if flag:
                t0 = time.time()
                number_label, labels, img_foreground, bboxes = frame_diff.process(frame)
                save_one_image_to_dataset(frame, bboxes, ds_dir, f"frame_{i:04d}")
                if img_foreground is not None:
                    cv2.imshow("img_foreground", img_foreground)

                    # add bboxes
                    frame = add_bbox(frame, bboxes, frame_id=i)
                    cv2.imshow("img_input", frame)
                    cv2.imwrite(f"img_{img_num:04d}.jpg", frame)
                    img_num += 1

                    if 0xFF & cv2.waitKey(10) == 27:
                        break
    else:
        img_array = sorted([str(img_i) for img_i in Path(args.image_dir).iterdir() if img_i.name.endswith(".jpg")])
        img_num = 0
        for i, img_i in enumerate(img_array):
            img = cv2.imread(img_i)
            t0 = time.time()
            number_label, labels, img_foreground, bboxes = frame_diff.process(img)
            save_one_image_to_dataset(img, bboxes, ds_dir, f"img_{i:04d}")
            if img_foreground is not None:
                cv2.imshow("img_foreground", img_foreground)

                # add bboxes
                img = add_bbox(img, bboxes, frame_id=i)
                cv2.imshow("img_input", img)
                # cv2.imwrite(f"img_{img_num:04d}.jpg", img_foreground)
                # cv2.imwrite(f"img_{img_num:04d}.jpg", img)
                img_num += 1

                # if img_num >= args.frame_num:
                #     break

                if 0xFF & cv2.waitKey(10) == 27:
                    break
                time.sleep(0.04)

    cv2.destroyAllWindows()
