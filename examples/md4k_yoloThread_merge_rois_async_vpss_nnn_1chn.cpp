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
// 提取通用的错误处理函数
bool handle_error(const char *action, int vpss_grp, int vpss_chn, int ret) {
  if (ret != TD_SUCCESS) {
    logger.log(ERROR, action, " error, grp: ", vpss_grp, " chn: ", vpss_chn,
               " Err code: ", ret);
    return false;
  }
  return true;
}

// 获取或释放帧数据
bool process_frames(std::vector<ot_video_frame_info> &v_frames,
                    bool release = false) {
  // only one chn: chn-1: {0, 0}, {0, 1}; chn-2: {2, 2}, {2, 3}
  std::vector<std::pair<td_s32, td_s32>> grp_chns{{0, 0}, {0, 1}};

  if (grp_chns.size() > v_frames.size()) {
    logger.log(ERROR, "Frame number should not be less than grp_chn pairs");
    return false;
  }

  for (size_t i = 0; i < grp_chns.size(); ++i) {
    td_s32 &vpss_grp = grp_chns[i].first;
    td_s32 &vpss_chn = grp_chns[i].second;

    if (release) {
      int ret = ss_mpi_vpss_release_chn_frame(vpss_grp, vpss_chn, &v_frames[i]);
      if (!handle_error("Release vpss chn", vpss_grp, vpss_chn, ret))
        return false;
    } else {
      int ret =
          ss_mpi_vpss_get_chn_frame(vpss_grp, vpss_chn, &v_frames[i], 100);
      if (!handle_error("Get vpss chn", vpss_grp, vpss_chn, ret))
        return false;
    }
  }
  return true;
}

// 新的函数，用于在独立线程中执行 yolov8.process_one_image
void processInThread(
    uint8_t cameraId, std::vector<std::vector<unsigned char>> &v_merged_roi4,
    std::vector<std::vector<std::pair<int, int>>> &vv_top_lefts4,
    std::vector<std::vector<std::vector<float>>> &vv_blob_xyxy4,
    YOLOV8Sync_combine &yolov8, int frame_id, int64_t pts) {
  yolov8.process_one_image_batched(v_merged_roi4, vv_top_lefts4, vv_blob_xyxy4,
                                   cameraId, frame_id, pts);
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
      "om_path", "tcp_ip",      "tcp_port", "output_dir",
      "roi_hw",  "save_result", "save_csv", "decode_step_mode"};
  for (const auto &key : required_keys) {
    if (!config_data.contains(key)) {
      logger.log(ERROR, "Can't find key: ", key);
      return 1;
    }
  }

  std::string omPath = config_data["om_path"];
  std::string tcp_ip = config_data["tcp_ip"];
  std::string tcp_port = config_data["tcp_port"];
  std::string output_dir = config_data["output_dir"];
  const int roi_hw = config_data["roi_hw"];
  bool b_save_result = config_data["save_result"];
  bool b_with_md_results = config_data["with_md_result"];
  bool b_save_csv = config_data["save_csv"];
  bool decode_step_mode = config_data["decode_step_mode"];
  const int max_roi_num = config_data["max_roi_num"];

  const int roi_size = roi_hw * roi_hw * 1.5; // YUV420sp
  const int merged_hw = roi_hw * 20;
  const int merged_size = merged_hw * merged_hw * 1.5;

  // Pre-allocate buffers outside the loop
  std::vector<std::vector<unsigned char>> v_merged_roi4(
      4, std::vector<unsigned char>(merged_size));

  std::vector<ot_ive_ccblob> blob4_camera0(4, {0});

  char absolute_path[PATH_MAX];

  const int IMAGE_SIZE = OT_SAMPLE_MD_WIDTH * OT_SAMPLE_MD_HEIGHT * 1.5;

  // initialize md
  // NOTE: md should initialized before SVPNNN
  bool b_sys_init = false;
  std::vector<IVE_MD> v_md4_camera0;
  v_md4_camera0.reserve(4);
  for (int i = 0; i < 4; ++i) {
    v_md4_camera0.emplace_back(b_sys_init);
  }

  // 初始化NPU
  YOLOV8Sync_combine yolov8(omPath, output_dir);
  uint8_t cameraId = getCameraId();

  // tcp
  yolov8.mb_with_md_results = b_with_md_results;
  yolov8.m_tcp_ip = tcp_ip;
  yolov8.m_tcp_port = std::stoi(tcp_port);

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
  std::vector<std::vector<unsigned char>> img4_camera0(
      4, std::vector<unsigned char>(IMAGE_SIZE));

  auto start_time = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<std::pair<int, int>>> vv_top_lefts4(4);
  std::vector<std::vector<std::vector<float>>> vv_blob_xyxy4(4);
  std::vector<ot_video_frame_info> v_frame_chns(2);
  bool b_yolo_on_camera0 = true;

  while (running) {
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
      if (!process_frames(v_frame_chns))
        break;
      copy_split_yuv420_from_frame(img4_camera0, &v_frame_chns[0]);
    }

    // MD
    {
      Timer timer("md processing ...");
      {
        Timer timer("md camera0 ...");
        for (int i = 0; i < 4; ++i) {
          v_md4_camera0.at(i).process(img4_camera0.at(i).data(),
                                      &blob4_camera0.at(i));
          logger.log(DEBUG, "camera-0 instance number: ",
                     static_cast<int>(blob4_camera0[i].info.bits.rgn_num));
        }
      }
    }

    // 合并ROI
    // yolov8.mb_yolo_ready = false;
    if (yolov8.mb_yolo_ready) {
      Timer timer("merge");

      std::vector<std::vector<unsigned char>> *p_img4;
      std::vector<ot_ive_ccblob> *p_blob4;

      p_img4 = &img4_camera0;
      p_blob4 = &blob4_camera0;

      for (int i = 0; i < 4; ++i) {
        std::vector<std::pair<int, int>> top_lefts_i;
        std::vector<std::vector<float>> blob_xyxy_i;
        merge_rois((*p_img4)[i].data(), &((*p_blob4)[i]), v_merged_roi4[i],
                   top_lefts_i, blob_xyxy_i, 4.0f, 4.0f, 1080, 1920, merged_hw,
                   merged_hw, max_roi_num);
        int top_lefts_offset_x_i = (i % 2) * 1920;
        int top_lefts_offset_y_i = (i / 2) * 1080;
        for (auto &tl : top_lefts_i) {
          tl.first = tl.first + top_lefts_offset_x_i;
          tl.second = tl.second + top_lefts_offset_y_i;
        }
        for (auto &xyxy : blob_xyxy_i) {
          xyxy[0] += top_lefts_offset_x_i;
          xyxy[1] += top_lefts_offset_y_i;
          xyxy[2] += top_lefts_offset_x_i;
          xyxy[3] += top_lefts_offset_y_i;
        }
        vv_top_lefts4[i] = std::move(top_lefts_i);
        vv_blob_xyxy4[i] = std::move(blob_xyxy_i);
      }
    }

    // 输入到NPU, 推理改为异步执行
    if (yolov8.mb_yolo_ready) {
      Timer timer("yolov8");
      uint64_t timestamp = 0;
      uint8_t cameraId_tmp;
      cameraId_tmp = cameraId;
      timestamp = v_frame_chns[1].video_frame.pts / 1000; // ms
      // 创建新线程
      std::thread asyncTask(processInThread, cameraId_tmp,
                            std::ref(v_merged_roi4), std::ref(vv_top_lefts4),
                            std::ref(vv_blob_xyxy4), std::ref(yolov8), frame_id,
                            timestamp);
      asyncTask.detach();
    }

    // 释放帧
    {
      Timer timer("Release Frames");
      process_frames(v_frame_chns, true);
    }
    frame_id++;
  }

  return 0;
}
