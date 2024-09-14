#include "ffmpeg_vdec_vpss.hpp"
#include "ive_md.hpp"
#include "nnn_yolov8_callback.hpp"
#include "ot_common_ive.h"
#include "ot_type.h"
#include "post_process_tools.hpp"
#include "svp_model_pingpong.hpp"
#include "utils.hpp"
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
#include <netinet/in.h>
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

// 全局日志器实例，初始日志级别为 INFO
Logger logger(INFO);

class YOLOV8_new : public NNNYOLOV8_CALLBACK {
public:
  YOLOV8_new(const std::string &modelPath, const std::string &output_dir = "./",
             const std::string &aclJSON = "");
  ~YOLOV8_new();

  std::string m_output_dir;
  std::vector<std::pair<int, int>> m_toplefts;
  std::vector<std::vector<char>> m_outputs;
  std::vector<std::vector<size_t>> mv_outputs_dim;
  int m_imageId;
  int merge_h, merge_w;
  float m_conf_thres = 0.5;
  float m_iou_thres = 0.6;
  float m_max_det = 300;
  int m_sock;
  bool mb_sock_connected = false;
  bool mb_save_results = false;

  // mvp_bbox shape: batch x branch_num x (anchors * 4)
  std::vector<std::vector<const half *>> mvp_bbox;
  // mvp_conf shape: batch x branch_num x anchors
  std::vector<std::vector<const half *>> mvp_conf;
  // mvp_cls shape: batch x branch_num x anchors
  std::vector<std::vector<const half *>> mvp_cls;

  void connect_to_tcp(const std::string &ip, const int port);

  void CallbackFunc(void *data) override;
  void update_imageId(int id) { m_imageId = id; }
};

std::atomic<bool> running(true);
void signal_handler(int signum) { running = false; }

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
      "rtsp_url",   "om_path", "tcp_id",      "tcp_port",
      "output_dir", "roi_hw",  "save_result", "decode_step_mode"};
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
  bool decode_step_mode = config_data["decode_step_mode"];

  const int roi_size = roi_hw * roi_hw * 1.5; // YUV420sp
  const int merged_hw = roi_hw * 20;
  const int merged_size = merged_hw * merged_hw * 1.5;

  // Pre-allocate buffers outside the loop
  std::vector<unsigned char> merged_roi(merged_size, 0);
  ot_ive_ccblob blob = {0};

  char absolute_path[PATH_MAX];

  const int IMAGE_SIZE = OT_SAMPLE_MD_WIDTH * OT_SAMPLE_MD_HEIGHT;
  const int IMAGE_SIZE2 = OT_SAMPLE_MD_WIDTH * OT_SAMPLE_MD_HEIGHT * 4 * 1.5;

  // initialize md
  // NOTE: md should initialized before SVPNNN
  IVE_MD md;

  // initialize ffmpeg_vdec_vpss
  HardwareDecoder decoder(rtsp_url, decode_step_mode);
  decoder.start_decode();

  // 初始化NPU
  YOLOV8_new yolov8(omPath, output_dir);

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
  std::vector<unsigned char> img_high(IMAGE_SIZE2);

  while (running && !decoder.is_ffmpeg_exit()) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (decoder.get_frame_without_release()) {
      copy_yuv420_from_frame(reinterpret_cast<char *>(img_high.data()),
                             &decoder.frame_H);
    } else {
      break;
    }

    auto md_start = std::chrono::high_resolution_clock::now();
    md.process(decoder.frame_L, &blob);
    logger.log(DEBUG,
               "instance number: ", static_cast<int>(blob.info.bits.rgn_num));

    decoder.release_frames();

    auto md_end = std::chrono::high_resolution_clock::now();
    // 合并ROI
    std::vector<std::pair<int, int>> top_lefts;
    merge_rois(img_high.data(), &blob, merged_roi, top_lefts, 8.0f, 8.0f, 2160,
               3840, merged_hw, merged_hw);
    auto merge_end = std::chrono::high_resolution_clock::now();

    // // debug
    // // save merged_roi
    // save_merged_rois(merged_roi, output_dir, frame_id);

    auto yolov8_syn = std::chrono::high_resolution_clock::now();
    sync_flag = yolov8.SynchronizeStream();
    if (sync_flag != SUCCESS) {
      logger.log(ERROR, "synchronizeStream failed");
      return 1;
    }
    auto yolov8_syn_end = std::chrono::high_resolution_clock::now();

    // 输入到NPU, 推理
    yolov8.m_toplefts = std::move(top_lefts);
    yolov8.update_imageId(frame_id++);
    yolov8.Host2Device(reinterpret_cast<char *>(merged_roi.data()),
                       merged_size);
    yolov8.ExecuteRPN_Async();
    auto yolov8_async_end = std::chrono::high_resolution_clock::now();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
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

    logger.log(DEBUG, "Frame ", frame_id, " processed. Duration: ", duration.count(),
               "ms, Decode: ", decode_cost.count(), "ms, MD: ", md_cost.count(),
               "ms, Merge: ", merge_cost.count(), "ms, Sync: ", syn_cost.count(),
               "ms, Async: ", asyn_cost.count(), "ms");
  }

  sync_flag = yolov8.SynchronizeStream();
  if (sync_flag != SUCCESS) {
    logger.log(ERROR, "synchronizeStream failed");
    return 1;
  }
  return 0;
}

YOLOV8_new::YOLOV8_new(const std::string &modelPath,
                       const std::string &output_dir,
                       const std::string &aclJSON)
    : NNNYOLOV8_CALLBACK(modelPath, aclJSON), m_imageId(0) {

  char c_output_dir[PATH_MAX];
  if (realpath(output_dir.c_str(), c_output_dir) == NULL) {
    logger.log(ERROR, "Output directory error: ", output_dir);
  }
  m_output_dir = std::string(c_output_dir);

  if (m_output_dir.back() != '/')
    m_output_dir += '/';
  logger.log(INFO, "Output directory is: ", m_output_dir);
  std::vector<size_t> outbuf_size;
  GetOutBufferSize(outbuf_size);
  m_outputs.resize(outbuf_size.size());
  logger.log(INFO, "out num: ", outbuf_size.size());
  for (size_t i = 0; i < outbuf_size.size(); ++i) {
    m_outputs[i].resize(outbuf_size[i], 0);
    logger.log(INFO, "size of output_", i, ": ", outbuf_size[i]);
  }

  GetModelInfo(nullptr, &merge_h, &merge_w, nullptr, nullptr, &mv_outputs_dim);
  for (auto &dim_i : mv_outputs_dim) {
    std::stringstream ss;
    ss << "out dim: " << std::endl;
    for (auto dim_i_j : dim_i) {
      ss << dim_i_j << ", ";
    }
    logger.log(INFO, ss.str());
  }
}

void YOLOV8_new::connect_to_tcp(const std::string &ip, const int port) {
  m_sock = 0;
  struct sockaddr_in serv_addr;

  if ((m_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    logger.log(ERROR, "Socket creation failed.");
    return;
  }

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port);

  if (inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr) <= 0) {
    logger.log(ERROR, "Invalid address / Address not supported.");
    return;
  }

  if (connect(m_sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    logger.log(ERROR, "Connection Failed.");
    return;
  }

  mb_sock_connected = true;
  logger.log(INFO, "Connected to TCP server at ", ip, ":", port);
  return;
}

void YOLOV8_new::CallbackFunc(void *data) {
  logger.log(DEBUG, "callback from yolov8_new");
  auto d2h_start =
      std::chrono::high_resolution_clock::now();
  std::vector<const char *> vp_outputs;
  Result ret = Device2Host(vp_outputs);
  if (ret != SUCCESS) {
    logger.log(ERROR, "Device2Host error");
    return;
  }
  // post process
  auto postp_start = std::chrono::high_resolution_clock::now();

  split_bbox_conf_reduced(vp_outputs, mv_outputs_dim, mvp_bbox, mvp_conf,
                          mvp_cls);

  const int batch_num = mvp_bbox.size();
  std::vector<std::vector<std::vector<half>>> det_bbox(batch_num);
  std::vector<std::vector<half>> det_conf(batch_num);
  std::vector<std::vector<half>> det_cls(batch_num);

  const int roi_hw = 32;
  for (int i = 0; i < batch_num; ++i) {
    const int box_num = mv_outputs_dim[0][2];
    const std::vector<const half *> &bbox_batch_i = mvp_bbox[i];
    const std::vector<const half *> &conf_batch_i = mvp_conf[i];
    const std::vector<const half *> &cls_batch_i = mvp_cls[i];
    NMS_bboxTranspose(box_num, bbox_batch_i, conf_batch_i, cls_batch_i,
                      det_bbox[i], det_conf[i], det_cls[i], m_conf_thres,
                      m_iou_thres, m_max_det);

    // filter detection. each DEC assign to one 32x32 patch.
    const int grid_num_x = merge_w / roi_hw;
    const int grid_num_y = merge_h / roi_hw;
    float c_x, c_y, w, h, conf;
    int grid_x, grid_y;
    std::array<std::array<float, 6>, 400> filted_decs = {0};
    for (auto j = 0; j < det_bbox[i].size(); ++j) {
      const std::vector<half> &box = det_bbox[i][j];
      c_x = 0.5f * (box[0] + box[2]);
      c_y = 0.5f * (box[1] + box[3]);
      conf = det_conf[i][j];
      grid_x = static_cast<int>(c_x / roi_hw);
      grid_y = static_cast<int>(c_y / roi_hw);
      const int filted_id = grid_y * grid_num_x + grid_x;
      const float &current_best_conf = filted_decs[filted_id][4];
      if (filted_id < 100 && conf > current_best_conf) {
        float w = box[2] - box[0];
        float h = box[3] - box[1];
        filted_decs[filted_id] = {c_x, c_y, w, h, conf, det_cls[i][j]};
      }
    }
    // change to real location
    std::vector<std::vector<float>> real_decs;
    const int instance_num = m_toplefts.size();

    for (int k = 0; k < instance_num; ++k) {
      const auto &tl = m_toplefts[k];
      const auto &dec = filted_decs[k];
      grid_x = k % grid_num_x;
      grid_y = k / grid_num_x;
      if (dec[4] > 0) {
        c_x = dec[0] - roi_hw * grid_x + tl.first;
        c_y = dec[1] - roi_hw * grid_y + tl.second;
      } else {
        c_x = static_cast<float>(tl.first);
        c_y = static_cast<float>(tl.second);
      }
      real_decs.push_back({c_x, c_y, dec[2], dec[3], dec[4], dec[5]});
    }

    if (mb_sock_connected) {
      std::string fileName = "decs_image_" + std::to_string(m_imageId) + ".bin";
      send_file_and_data(m_sock, fileName, real_decs);
    }
    if (mb_save_results) {
      save_detect_results(real_decs, m_output_dir, m_imageId);
    }
  }
  auto postp_end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      postp_end - postp_start);
  auto d2h_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
      postp_start - d2h_start);

  logger.log(DEBUG, "Post-processing cost: ", duration.count(),
             "ms, D2H cost: ", d2h_cost.count(), "ms");
}

YOLOV8_new::~YOLOV8_new() {
  if (mb_sock_connected) {
    close(m_sock);
    mb_sock_connected = false;
    logger.log(INFO, "Disconnected from TCP server.");
  }
}
