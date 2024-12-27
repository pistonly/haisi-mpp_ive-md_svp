#include "ffmpeg_vdec_vpss.hpp"
#include "ive_md.hpp"
#include "ot_common_ive.h"
#include "ot_type.h"
#include "utils.hpp"
#include "yolov8_nnn.hpp"
#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <future>
#include <half.hpp>
#include <iomanip>
#include <iostream>
#include <linux/limits.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using half_float::half;
using json = nlohmann::json;

extern Logger logger;

std::atomic<bool> running(true);
void signal_handler(int signum) { running = false; }

// 新的函数，用于在独立线程中执行 yolov8.process_one_image
void processInThread(uint8_t cameraId, std::vector<unsigned char> &merged_roi,
                     std::vector<unsigned char> &sky_img,
                     std::vector<std::pair<int, int>> &v_top_lefts,
                     std::vector<std::vector<float>> &v_blob_xyxy,
                     YOLOV8Sync &yolov8_sky, YOLOV8Sync &yolov8, int frame_id,
                     int64_t pts) {
  std::vector<std::vector<std::vector<half>>> sky_det_bbox;
  std::vector<std::vector<half>> sky_det_conf;
  std::vector<std::vector<half>> sky_det_cls;
  yolov8_sky.process_one_image(sky_img, sky_det_bbox, sky_det_conf,
                               sky_det_cls);
  yolov8.process_one_image(merged_roi, v_top_lefts, v_blob_xyxy, sky_det_bbox,
                           sky_det_conf, sky_det_cls, cameraId, frame_id, pts);
}

int main(int argc, char *argv[]) {
  std::cout << "Usage: " << argv[0] << " <config_path>" << std::endl;
  std::string configure_path = "../data/configure_4k_nnn.json";

  if (argc > 1)
    configure_path = argv[1];

  // read configure
  std::ifstream config_file(configure_path);
  if (!config_file.is_open()) {
    logger.log(ERROR, "Can't open configure file: ", configure_path);
    return 1;
  }

  json config_data;
  try {
    config_file >> config_data;
  } catch (json::parse_error &e) {
    logger.log(ERROR, "JSON parse error: ", e.what());
    return 1;
  }

  // 设置日志级别
  if (config_data.contains("log_level")) {
    std::string level = config_data["log_level"];
    if (level == "DEBUG") {
      logger.setLogLevel(DEBUG);
    } else if (level == "INFO") {
      logger.setLogLevel(INFO);
    } else if (level == "WARNING") {
      logger.setLogLevel(WARNING);
    } else if (level == "ERROR") {
      logger.setLogLevel(ERROR);
    } else {
      logger.log(WARNING, "Unknown log level: ", level, ", using INFO level.");
      logger.setLogLevel(INFO);
    }
  }

  std::vector<std::string> required_keys = {
      "rtsp_url", "om_path",     "tcp_ip",   "tcp_port",        "output_dir",
      "roi_hw",   "save_result", "save_csv", "decode_step_mode"};
  for (const auto &key : required_keys) {
    if (!config_data.contains(key)) {
      logger.log(ERROR, "Can't find key: ", key);
      return 1;
    }
  }

  std::string rtsp_url = config_data["rtsp_url"];
  std::string omPath = config_data["om_path"];
  std::string tcp_ip = config_data["tcp_ip"];
  std::string tcp_port = config_data["tcp_port"];
  std::string output_dir = config_data["output_dir"];
  const int roi_hw = config_data["roi_hw"];
  bool b_save_result = config_data["save_result"];
  bool b_save_csv = config_data["save_csv"];
  bool b_with_md_results = config_data["with_md_result"];
  bool decode_step_mode = config_data["decode_step_mode"];

  const int max_roi_num = config_data["max_roi_num"];

  const int roi_size = roi_hw * roi_hw * 1.5; // YUV420sp
  const int merged_hw = roi_hw * 20;
  const int merged_size = merged_hw * merged_hw * 1.5;

  // Pre-allocate buffers outside the loop
  std::vector<unsigned char> merged_roi(merged_size, 0);

  std::vector<ot_ive_ccblob> blob4(4, {0});

  char absolute_path[PATH_MAX];

  const int IMAGE_SIZE = OT_SAMPLE_MD_WIDTH * OT_SAMPLE_MD_HEIGHT * 1.5;

  // initialize md
  // NOTE: md should initialized before SVPNNN
  std::vector<IVE_MD> v_md4(4);

  // initialize ffmpeg_vdec_vpss
  HardwareDecoder decoder(rtsp_url, decode_step_mode);
  decoder.start_decode();

  // 初始化NPU
  YOLOV8Sync yolov8(omPath, output_dir);
  YOLOV8Sync yolov8_sky(omPath, output_dir, false);
  uint8_t cameraId = getCameraId();

  // tcp server
  yolov8.m_tcp_ip = tcp_ip;
  yolov8.m_tcp_port = std::stoi(tcp_port);
  yolov8.mb_tcp_send = false;
  yolov8.mb_with_md_results = true;

  yolov8.mb_save_results = b_save_result;
  yolov8.mb_save_csv = b_save_csv;
  if (config_data.contains("conf_thres"))
    yolov8.m_conf_thres = config_data["conf_thres"];
  if (config_data.contains("iou_thres"))
    yolov8.m_iou_thres = config_data["iou_thres"];
  if (config_data.contains("max_det"))
    yolov8.m_max_det = config_data["max_det"];

  Result sync_flag;

  signal(SIGINT, signal_handler); // capture Ctrl+C
  int frame_id = 0;

  // Pre-allocate buffers
  std::vector<std::vector<unsigned char>> img4(
      4, std::vector<unsigned char>(IMAGE_SIZE));

  auto start_time = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<float>> blob_xyxy;
  std::vector<std::pair<int, int>> top_lefts;
  std::vector<unsigned char> img_high(4 * IMAGE_SIZE);
  const int sky_img_h = 640;
  const int sky_img_w = 640;
  std::vector<unsigned char> sky_img(sky_img_h * sky_img_w * 1.5);

  while (running && !decoder.is_ffmpeg_exit()) {
    if (frame_id % 100 == 0) {
      auto _now = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::seconds>(_now - start_time)
              .count();
      logger.log(INFO, "frame id: ", frame_id,
                 " fps: ", frame_id * 1.f / duration);
    }

    {
      Timer timer("Get Frames");
      if (decoder.get_frame_without_release()) {
        {
          Timer timer("split frame");
          copy_split_yuv420_from_frame(img4, &decoder.frame_H);
        }
        {
          Timer timer("copy 4k frame");
          copy_yuv420_from_frame(reinterpret_cast<char *>(img_high.data()),
                                 &decoder.frame_H);
        }
        decoder.release_frames();
      } else {
        break;
      }
    }

    {
      Timer timer("md");
      for (int i = 0; i < 4; ++i) {
        v_md4[i].process(img4[i].data(), &blob4[i]);
        logger.log(DEBUG, "instance number: ",
                   static_cast<int>(blob4[i].info.bits.rgn_num));
      }
    }

    // 合并ROI
    if (yolov8.mb_yolo_ready) {
      Timer timer("merge");
      blob_xyxy.clear();
      for (int i = 0; i < 4; ++i) {
        std::vector<std::vector<float>> blob_xyxy_i;
        blob_to_xyxy(&blob4[i], blob_xyxy_i, 4.0f, 32);

        int top_lefts_offset_x_i = (i % 2) * 1920;
        int top_lefts_offset_y_i = (i / 2) * 1080;
        for (auto &xyxy : blob_xyxy_i) {
          xyxy[0] += top_lefts_offset_x_i;
          xyxy[1] += top_lefts_offset_y_i;
          xyxy[2] += top_lefts_offset_x_i;
          xyxy[3] += top_lefts_offset_y_i;
        }

        blob_xyxy.insert(blob_xyxy.end(), blob_xyxy_i.begin(),
                         blob_xyxy_i.end());
      }
      // sort blob_xyxy by second entry(blob_xyxy[i][1])
      std::sort(blob_xyxy.begin(), blob_xyxy.end(),
                [](const std::vector<float> &a, const std::vector<float> &b) {
                  return a[1] < b[1];
                });
      if (blob_xyxy.size() > 400) {
        blob_xyxy.resize(400);
      }

      merge_rois(img_high.data(), blob_xyxy, merged_roi, top_lefts, 2160, 3840,
                 640, 640, 400);
    }

    // 输入到NPU, 推理改为异步执行
    if (yolov8.mb_yolo_ready) {
      Timer timer("yolov8");
      uint64_t timestamp = decoder.frame_H.video_frame.pts / 1000; // ms

      // resize for sky
      {
        Timer timer("resize yuv420");
        int sky_offset_x, sky_offset_y;
        resize_yuv420(sky_img, sky_offset_x, sky_offset_y, img_high, 2160, 3840,
                      sky_img_h, sky_img_w);
      }

      // 创建新线程
      std::thread asyncTask(processInThread, cameraId, std::ref(merged_roi),
                            std::ref(sky_img),
                            std::ref(top_lefts), std::ref(blob_xyxy),
                            std::ref(yolov8_sky), std::ref(yolov8), frame_id,
                            timestamp);
      asyncTask.detach();
    }

    frame_id++;
  }

  return 0;
}
