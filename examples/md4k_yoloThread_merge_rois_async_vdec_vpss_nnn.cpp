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
void processInThread(
    uint8_t cameraId, std::vector<unsigned char> &merged_roi_combined,
    std::vector<std::vector<std::pair<int, int>>> &vv_top_lefts4,
    YOLOV8Sync_combine &yolov8, int frame_id, int64_t pts) {
  yolov8.process_one_image(merged_roi_combined, vv_top_lefts4, cameraId,
                           frame_id, pts);
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
      "rtsp_url",   "om_path", "tcp_ip",      "tcp_port",
      "output_dir", "roi_hw",  "save_result", "save_csv", "decode_step_mode"};
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
  bool decode_step_mode = config_data["decode_step_mode"];

  const int roi_size = roi_hw * roi_hw * 1.5; // YUV420sp
  const int merged_hw = roi_hw * 20;
  const int merged_size = merged_hw * merged_hw * 1.5;

  // Pre-allocate buffers outside the loop
  std::vector<std::vector<unsigned char>> v_merged_roi4(
      4, std::vector<unsigned char>(merged_size));
  std::vector<unsigned char> merged_roi_combined(merged_size * 4, 0);

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
  YOLOV8Sync_combine yolov8(omPath, output_dir);
  uint8_t cameraId = getCameraId();

  // connect to tcp server
  yolov8.connect_to_tcp(tcp_ip, std::stoi(tcp_port));
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
  std::vector<std::vector<std::pair<int, int>>> vv_top_lefts4(4);
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
        copy_split_yuv420_from_frame(img4, &decoder.frame_H);
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

      decoder.release_frames();
    }

    // 合并ROI
    if (yolov8.mb_yolo_ready) {
      Timer timer("merge");
      for (int i = 0; i < 4; ++i) {
        std::vector<std::pair<int, int>> top_lefts_i;
        merge_rois(img4[i].data(), &blob4[i], v_merged_roi4[i], top_lefts_i,
                   4.0f, 4.0f, 1080, 1920, merged_hw, merged_hw);
        int top_lefts_offset_x_i = (i % 2) * 1920;
        int top_lefts_offset_y_i = (i / 2) * 1080;
        for (auto &tl : top_lefts_i) {
          tl.first = tl.first + top_lefts_offset_x_i;
          tl.second = tl.second + top_lefts_offset_y_i;
        }
        vv_top_lefts4[i] = std::move(top_lefts_i);
      }
      // merge to 4k
      combine_YUV420sp(v_merged_roi4, merged_hw * 2, merged_hw * 2,
                       merged_roi_combined);
      // // debug
      // // save merged_roi
      // save_merged_rois(merged_roi_combined, output_dir, frame_id);
    }

    // 输入到NPU, 推理改为异步执行
    if (yolov8.mb_yolo_ready) {
      Timer timer("yolov8");
      uint64_t timestamp = decoder.frame_H.video_frame.pts / 1000; // ms
      // 创建新线程
      std::thread asyncTask(processInThread, cameraId,
                            std::ref(merged_roi_combined),
                            std::ref(vv_top_lefts4), std::ref(yolov8), frame_id,
                            timestamp);
      asyncTask.detach();
    }

    frame_id++;
  }

  return 0;
}
