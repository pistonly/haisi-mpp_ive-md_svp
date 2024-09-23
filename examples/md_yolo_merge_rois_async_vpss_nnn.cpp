#include "ive_md.hpp"
#include "ot_common_ive.h"
#include "ot_common_video.h"
#include "ot_type.h"
#include "ss_mpi_vpss.h"
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
#include <unistd.h>
#include <utility>
#include <vector>

using half_float::half;
using json = nlohmann::json;

extern Logger logger;

std::atomic<bool> running(true);
void signal_handler(int signum) { running = false; }

bool get_frames(std::vector<ot_video_frame_info> &v_frames) {
  std::vector<std::pair<td_s32, td_s32>> grp_chns{
      {0, 0}, {0, 1}, {2, 2}, {2, 3}};
  if (grp_chns.size() > v_frames.size()) {
    logger.log(ERROR,
               "frame number should not less than that of grp_chn pairs");
  }
  for (auto i = 0; i < grp_chns.size(); ++i) {
    std::pair<td_s32, td_s32> &grp_chn_i = grp_chns[i];
    td_s32 &vpss_grp = grp_chn_i.first;
    td_s32 &vpss_chn = grp_chn_i.second;
    td_s32 ret =
        ss_mpi_vpss_get_chn_frame(vpss_grp, vpss_chn, &v_frames[i], 100);
    if (ret != TD_SUCCESS) {
      logger.log(ERROR, "get vpss chn error, grp: ", vpss_grp,
                 " chn: ", vpss_chn, "Err code: ", ret);
      return false;
    }
  }
  return true;
}

bool release_frames(std::vector<ot_video_frame_info> &v_frames) {
  std::vector<std::pair<td_s32, td_s32>> grp_chns{
      {0, 0}, {0, 1}, {2, 2}, {2, 3}};
  if (grp_chns.size() > v_frames.size()) {
    logger.log(ERROR,
               "frame number should not less than that of grp_chn pairs");
  }
  for (auto i = 0; i < grp_chns.size(); ++i) {
    std::pair<td_s32, td_s32> &grp_chn_i = grp_chns[i];
    td_s32 &vpss_grp = grp_chn_i.first;
    td_s32 &vpss_chn = grp_chn_i.second;
    td_s32 ret =
        ss_mpi_vpss_release_chn_frame(vpss_grp, vpss_chn, &v_frames[i]);
    if (ret != TD_SUCCESS) {
      logger.log(ERROR, "release vpss chn error, grp: ", vpss_grp,
                 " chn: ", vpss_chn, "Err code: ", ret);
      return false;
    }
  }
  return true;
}

int main(int argc, char *argv[]) {
  std::cout << "Usage: " << argv[0] << " <config_path>" << std::endl;
  std::string configure_path = "../data/configure.json";

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
      "rtsp_url",   "om_path", "tcp_id",     "tcp_port",
      "output_dir", "roi_hw",  "save_result"};
  for (const auto &key : required_keys) {
    if (!config_data.contains(key)) {
      logger.log(ERROR, "Can't find key: ", key);
      return 1;
    }
  }

  std::string rtsp_url = config_data["rtsp_url"];
  std::string omPath = config_data["om_path"];
  std::string tcp_ip = config_data["tcp_id"];
  std::string tcp_port = config_data["tcp_port"];
  std::string output_dir = config_data["output_dir"];
  const int roi_hw = config_data["roi_hw"];
  bool b_save_result = config_data["save_result"];

  const int roi_size = roi_hw * roi_hw * 1.5; // YUV420sp
  const int merged_hw = roi_hw * 20;
  const int merged_size = merged_hw * merged_hw * 1.5;

  // Pre-allocate buffers outside the loop
  std::vector<unsigned char> merged_roi(merged_size, 0);
  ot_ive_ccblob blob_0 = {0};
  ot_ive_ccblob blob_1 = {0};

  char absolute_path[PATH_MAX];

  const int IMAGE_SIZE = OT_SAMPLE_MD_WIDTH * OT_SAMPLE_MD_HEIGHT;
  const int IMAGE_SIZE2 = OT_SAMPLE_MD_WIDTH * OT_SAMPLE_MD_HEIGHT * 4 * 1.5;

  // initialize md
  // NOTE: md should initialized before SVPNNN
  bool b_sys_init = false;
  IVE_MD md_0(b_sys_init);
  IVE_MD md_1(b_sys_init);

  // 初始化NPU
  YOLOV8_nnn_2chns yolov8(omPath, output_dir);

  // connect to tcp server
  yolov8.connect_to_tcp(tcp_ip, std::stoi(tcp_port));
  yolov8.mb_save_results = b_save_result;
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
  std::vector<unsigned char> img(IMAGE_SIZE);
  std::vector<unsigned char> img_high_0(IMAGE_SIZE2);
  std::vector<unsigned char> img_high_1(IMAGE_SIZE2);

  std::vector<ot_video_frame_info> v_frame_chns(4);

  auto last_start_time = std::chrono::high_resolution_clock::now();
  while (running) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (!get_frames(v_frame_chns)) {
      break;
    }
    auto get_frame_end = std::chrono::high_resolution_clock::now();

    /* -------------------- process ch-0 -------------------- */
    auto &frame_H_0 = v_frame_chns[0];
    copy_yuv420_from_frame(reinterpret_cast<char *>(img_high_0.data()),
                           &frame_H_0);

    auto md_start_0 = std::chrono::high_resolution_clock::now();
    auto &frame_L_0 = v_frame_chns[1];
    md_0.process(frame_L_0, &blob_0);
    logger.log(DEBUG, "instance number of ch0: ",
               static_cast<int>(blob_0.info.bits.rgn_num));

    auto md_end_0 = std::chrono::high_resolution_clock::now();
    // 合并ROI
    std::vector<std::pair<int, int>> top_lefts;
    merge_rois(img_high_0.data(), &blob_0, merged_roi, top_lefts, 8.0f, 8.0f,
               2160, 3840, merged_hw, merged_hw);
    auto merge_end_0 = std::chrono::high_resolution_clock::now();

    // // debug
    // // save merged_roi
    // save_merged_rois(merged_roi, output_dir, frame_id);

    auto yolov8_syn_0 = std::chrono::high_resolution_clock::now();
    sync_flag = yolov8.SynchronizeStream();
    if (sync_flag != SUCCESS) {
      logger.log(ERROR, "synchronizeStream failed");
      return 1;
    }
    auto yolov8_syn_end_0 = std::chrono::high_resolution_clock::now();

    // 输入到NPU, 推理
    yolov8.m_toplefts = std::move(top_lefts);
    int current_ch = 0;
    yolov8.update_imageId(frame_id, current_ch);
    yolov8.Host2Device(reinterpret_cast<char *>(merged_roi.data()),
                       merged_size);
    yolov8.ExecuteRPN_Async();

    /* -------------------- process ch-1 -------------------- */

    auto ch1_start = std::chrono::high_resolution_clock::now();
    auto &frame_H_1 = v_frame_chns[2];
    copy_yuv420_from_frame(reinterpret_cast<char *>(img_high_1.data()),
                           &frame_H_1);
    release_frames(v_frame_chns);

    auto md_start_1 = std::chrono::high_resolution_clock::now();
    auto &frame_L_1 = v_frame_chns[3];
    md_1.process(frame_L_1, &blob_1);
    logger.log(DEBUG, "instance number of ch1: ",
               static_cast<int>(blob_1.info.bits.rgn_num));
    auto md_end_1 = std::chrono::high_resolution_clock::now();

    // merge
    std::vector<std::pair<int, int>> top_lefts_1;
    merge_rois(img_high_0.data(), &blob_0, merged_roi, top_lefts_1, 8.0f, 8.0f,
               2160, 3840, merged_hw, merged_hw);

    auto merge_end_1 = std::chrono::high_resolution_clock::now();

    sync_flag = yolov8.SynchronizeStream();
    if (sync_flag != SUCCESS) {
      logger.log(ERROR, "synchronizeStream failed");
      return 1;
    }
    auto yolov8_syn_end_1 = std::chrono::high_resolution_clock::now();

    // 输入到NPU, 推理
    yolov8.m_toplefts = std::move(top_lefts_1);
    current_ch = 1;
    yolov8.update_imageId(frame_id++, current_ch);
    yolov8.Host2Device(reinterpret_cast<char *>(merged_roi.data()),
                       merged_size);
    yolov8.ExecuteRPN_Async();

    auto yolov8_async_end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        last_start_time - start_time);
    auto decode_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
        md_start - start_time);
    auto md_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
        md_end - md_start);
    auto merge_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
        merge_end - md_end);
    auto syn_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
        yolov8_syn_end - yolov8_syn);
    auto asyn_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
        yolov8_async_end - yolov8_syn_end);

    logger.log(
        DEBUG, "Frame ", frame_id, " processed. Duration: ", duration.count(),
        "ms, Decode: ", decode_cost.count(), "ms, MD: ", md_cost.count(),
        "ms, Merge: ", merge_cost.count(), "ms, Sync: ", syn_cost.count(),
        "ms, Async: ", asyn_cost.count(), "ms");
    last_start_time = start_time;
  }

  sync_flag = yolov8.SynchronizeStream();
  if (sync_flag != SUCCESS) {
    logger.log(ERROR, "synchronizeStream failed");
    return 1;
  }
  return 0;
}
