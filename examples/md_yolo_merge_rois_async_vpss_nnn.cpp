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

// 时间测量工具类
class Timer {
public:
  Timer(const std::string &name)
      : name_(name), start_(std::chrono::high_resolution_clock::now()) {}
  ~Timer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start_)
            .count();
    logger.log(DEBUG, name_, " took ", duration, " ms");
  }

private:
  std::string name_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

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
  std::vector<std::pair<td_s32, td_s32>> grp_chns{
      {0, 0}, {0, 1}, {2, 2}, {2, 3}};

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

int main(int argc, char *argv[]) {
  std::cout << "Usage: " << argv[0] << " <config_path>" << std::endl;
  std::string configure_path = (argc > 1) ? argv[1] : "../data/configure.json";

  // 读取配置文件
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
  std::string log_level = config_data.value("log_level", "INFO");
  if (log_level == "DEBUG")
    logger.setLogLevel(DEBUG);
  else if (log_level == "INFO")
    logger.setLogLevel(INFO);
  else if (log_level == "WARNING")
    logger.setLogLevel(WARNING);
  else if (log_level == "ERROR")
    logger.setLogLevel(ERROR);
  else {
    logger.log(WARNING, "Unknown log level: ", log_level,
               ", using INFO level.");
    logger.setLogLevel(INFO);
  }

  // 检查必需的键值
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
  int roi_hw = config_data["roi_hw"];
  bool b_save_result = config_data["save_result"];

  const int roi_size = roi_hw * roi_hw * 1.5; // YUV420sp
  const int merged_hw = roi_hw * 20;
  const int merged_size = merged_hw * merged_hw * 1.5;

  // initialize md
  // NOTE: md should initialized before SVPNNN
  const int IMAGE_SIZE = OT_SAMPLE_MD_WIDTH * OT_SAMPLE_MD_HEIGHT;
  const int IMAGE_SIZE2 = OT_SAMPLE_MD_WIDTH * OT_SAMPLE_MD_HEIGHT * 4 * 1.5;
  bool b_sys_init = false;
  IVE_MD md_0(b_sys_init);
  IVE_MD md_1(b_sys_init);

  // initialize npu
  YOLOV8_nnn_2chns yolov8(omPath, output_dir);
  yolov8.connect_to_tcp(tcp_ip, std::stoi(tcp_port));
  yolov8.mb_save_results = b_save_result;
  yolov8.m_conf_thres = config_data.value("conf_thres", yolov8.m_conf_thres);
  yolov8.m_iou_thres = config_data.value("iou_thres", yolov8.m_iou_thres);
  yolov8.m_max_det = config_data.value("max_det", yolov8.m_max_det);

  signal(SIGINT, signal_handler); // 捕获 Ctrl+C
  int frame_id = 0;

  // Pre-allocate buffers
  std::vector<unsigned char> img(IMAGE_SIZE);
  std::vector<unsigned char> img_high_0(IMAGE_SIZE2);
  std::vector<unsigned char> img_high_1(IMAGE_SIZE2);

  std::vector<ot_video_frame_info> v_frame_chns(4);
  std::vector<unsigned char> merged_roi(merged_size, 0);
  ot_ive_ccblob blob = {0};

  while (running) {
    if (frame_id % 100 == 0)
      logger.log(INFO, "frame id: ", frame_id);
    Timer frame_timer("Total Frame Processing");

    // 获取帧
    {
      Timer timer("Get Frames");
      if (!process_frames(v_frame_chns))
        break;
    }

    /* 处理第一个通道 */
    {
      Timer timer("Process Channel 0");
      auto &frame_H_0 = v_frame_chns[0];
      copy_yuv420_from_frame(reinterpret_cast<char *>(img_high_0.data()),
                             &frame_H_0);

      // MD处理和合并
      {
        Timer md_timer("MD Processing for Channel 0");
        auto &frame_L_0 = v_frame_chns[1];
        md_0.process(frame_L_0, &blob);
        logger.log(DEBUG, "instance number of ch0: ",
                   static_cast<int>(blob.info.bits.rgn_num));
      }

      // 合并ROI
      std::vector<std::pair<int, int>> top_lefts;
      {
        Timer merge_timer("Merge ROI for Channel 0");
        merge_rois(img_high_0.data(), &blob, merged_roi, top_lefts, 8.0f,
                   8.0f, 2160, 3840, merged_hw, merged_hw);
      }

      // 推理
      {
        Timer yolov8_timer("YOLOV8 Inference for Channel 0");
        if (yolov8.SynchronizeStream() != SUCCESS) {
          logger.log(ERROR, "synchronizeStream failed");
          return 1;
        }
        yolov8.m_toplefts = std::move(top_lefts);
        int current_ch = 0;
        yolov8.update_imageId(frame_id, current_ch);
        yolov8.Host2Device(reinterpret_cast<char *>(merged_roi.data()),
                           merged_size);
        yolov8.ExecuteRPN_Async();
      }
    }

    /* 处理第二个通道 */
    {
      Timer timer("Process Channel 1");
      auto &frame_H_1 = v_frame_chns[2];
      copy_yuv420_from_frame(reinterpret_cast<char *>(img_high_1.data()),
                             &frame_H_1);

      // MD处理和合并
      {
        Timer md_timer("MD Processing for Channel 1");
        auto &frame_L_1 = v_frame_chns[3];
        md_1.process(frame_L_1, &blob);
        logger.log(DEBUG, "instance number of ch1: ",
                   static_cast<int>(blob.info.bits.rgn_num));
      }

      // 合并ROI
      std::vector<std::pair<int, int>> top_lefts_1;
      {
        Timer merge_timer("Merge ROI for Channel 1");
        merge_rois(img_high_1.data(), &blob, merged_roi, top_lefts_1, 8.0f,
                   8.0f, 2160, 3840, merged_hw, merged_hw);
      }

      // 推理
      {
        Timer yolov8_timer("YOLOV8 Inference for Channel 1");
        if (yolov8.SynchronizeStream() != SUCCESS) {
          logger.log(ERROR, "synchronizeStream failed");
          return 1;
        }
        yolov8.m_toplefts = std::move(top_lefts_1);
        int current_ch = 1;
        yolov8.update_imageId(frame_id, current_ch);
        yolov8.Host2Device(reinterpret_cast<char *>(merged_roi.data()),
                           merged_size);
        yolov8.ExecuteRPN_Async();
      }
    }

    // 释放帧
    {
      Timer timer("Release Frames");
      process_frames(v_frame_chns, true);
    }

    frame_id++;
  }

  yolov8.SynchronizeStream();
  return 0;
}
