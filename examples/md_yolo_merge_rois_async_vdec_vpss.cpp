#include "ffmpeg_vdec_vpss.hpp"
#include "ive_md.hpp"
#include "ot_common_ive.h"
#include "ot_common_video.h"
#include "ot_type.h"
#include "svp_model_pingpong.hpp"
#include "svp_yolov8.hpp"
#include "svp_yolov8_callback.hpp"
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
#include <iomanip>
#include <iostream>
#include <linux/limits.h>
#include <netinet/in.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

class YOLOV8_new : public SVPYOLOV8_CALLBACK {
public:
  YOLOV8_new(const std::string &modelPath, const std::string &output_dir = "./",
             const std::string &aclJSON = "");
  ~YOLOV8_new();

  std::string m_output_dir;
  std::vector<std::pair<int, int>> m_toplefts;
  std::vector<std::vector<char>> m_outputs;
  int m_imageId;
  int merge_h, merge_w, total_dec_num;
  int m_sock;
  bool mb_sock_connected = false;
  bool mb_save_results = false;
  void connect_to_tcp(const std::string &ip, const int port);

  void CallbackFunc(void *data) override;
  void update_imageId(int id) { m_imageId = id; }
};

std::atomic<bool> running(true);
void signal_handler(int signum) { running = false; }

int main(int argc, char *argv[]) {
  std::cout << "Usage: " << argv[0] << " <RTSP URL>"
            << " <om_path>" << std::endl;
  std::string rtsp_url = "rtsp://172.23.24.52:8554/test";
  std::string omPath = "/home/liuyang/Documents/haisi/ai-sd3403/"
                       "ai-sd3403/models/"
                       "yolov8n_air-little-obj_32_roi_new_640x640_rpn_original."
                       "om";
  std::string tcp_ip = "172.23.24.52";
  std::string tcp_port = "8880";
  std::string output_dir = "/mnt/disk/tmp_svp/";

  if (argc > 1)
    rtsp_url = argv[1];

  if (argc > 2)
    omPath = std::string(argv[2]);

  if (argc > 3)
    tcp_ip = std::string(argv[3]);

  if (argc > 4)
    tcp_port = std::string(argv[4]);

  if (argc > 5)
    output_dir = std::string(argv[5]);

  const int roi_hw = 32;
  const int roi_size = roi_hw * roi_hw * 1.5; // YUV420sp
  const int merged_hw = roi_hw * 20;
  const int merged_size = merged_hw * merged_hw * 1.5;
  std::vector<unsigned char> merged_roi(merged_size, 0);
  ot_ive_ccblob blob = {0};

  char absolute_path[PATH_MAX];

  const int IMAGE_SIZE = OT_SAMPLE_MD_WIDTH * OT_SAMPLE_MD_HEIGHT;
  const int IMAGE_SIZE2 = OT_SAMPLE_MD_WIDTH * OT_SAMPLE_MD_HEIGHT * 4 * 1.5;

  // initialize md
  // NOTE: md should initialized before SVPNNN
  IVE_MD md;

  // initialize ffmpeg_vdec_vpss
  HardwareDecoder decoder(rtsp_url);
  decoder.start_decode();

  // 初始化NPU
  YOLOV8_new yolov8(omPath, output_dir);

  // connect to tcp server
  yolov8.connect_to_tcp(tcp_ip, std::stoi(tcp_port));
  yolov8.mb_save_results = true;

  Result sync_flag;
  td_s32 decoder_flag;

  signal(SIGINT, signal_handler); // capture Ctrl+C
  int frame_id = 0;
  while (running && !decoder.is_ffmpeg_exit()) {
    std::vector<unsigned char> img(IMAGE_SIZE);
    std::vector<unsigned char> img_high(IMAGE_SIZE2);

    decoder_flag = decoder.get_frame_without_release();
    if (decoder_flag) {
      copy_yuv420_from_frame(reinterpret_cast<char *>(img_high.data()),
                             &decoder.frame_H);
    } else {
      continue;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    md.process(decoder.frame_L, &blob);
    std::cout << "instance number: " << static_cast<int>(blob.info.bits.rgn_num)
              << std::endl;

    decoder.release_frames();

    auto md_end = std::chrono::high_resolution_clock::now();
    // 合并ROI
    std::vector<std::pair<int, int>> top_lefts;
    merge_rois(img_high.data(), &blob, merged_roi, top_lefts, 8.0f, 8.0f, 2160,
               3840, merged_hw, merged_hw);
    auto merge_end = std::chrono::high_resolution_clock::now();

    // // save merged_roi
    // save_merged_rois(merged_roi, output_dir, i);

    auto yolov8_syn = std::chrono::high_resolution_clock::now();
    sync_flag = yolov8.SynchronizeStream();
    if (sync_flag != SUCCESS) {
      std::cerr << "synchronizeStream failed" << std::endl;
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
    auto md_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
        md_end - start_time);
    auto merge_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
        merge_end - md_end);
    auto syn_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
        yolov8_syn_end - yolov8_syn);
    auto asyn_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
        yolov8_async_end - yolov8_async_end);
    std::cout << "------------duration: " << duration.count()
              << ", md cost: " << md_cost.count()
              << ", merge cost: " << merge_cost.count()
              << ", syn cost: " << syn_cost.count()
              << ", async const: " << asyn_cost.count()
              << " milliseconds----------------" << std::endl;
  }

  sync_flag = yolov8.SynchronizeStream();
  if (sync_flag != SUCCESS) {
    std::cerr << "synchronizeStream failed" << std::endl;
    return 1;
  }
  return 0;
}

YOLOV8_new::YOLOV8_new(const std::string &modelPath,
                       const std::string &output_dir,
                       const std::string &aclJSON)
    : SVPYOLOV8_CALLBACK(modelPath, aclJSON), m_imageId(0) {

  char c_output_dir[PATH_MAX];
  if (realpath(output_dir.c_str(), c_output_dir) == NULL) {
    std::cerr << "output_dir error: " << c_output_dir << std::endl;
  }
  m_output_dir = std::string(c_output_dir);

  if (m_output_dir.back() != '/')
    m_output_dir += '/';
  std::cout << "output_dir is: " << m_output_dir << std::endl;
  std::vector<size_t> outbuf_size;
  GetOutBufferSize(outbuf_size);
  m_outputs.resize(outbuf_size.size());
  for (size_t i = 0; i < outbuf_size.size(); ++i) {
    m_outputs[i].resize(outbuf_size[i], 0);
  }

  std::vector<std::vector<size_t>> vv_output_dims;
  GetModelInfo(nullptr, &merge_h, &merge_w, nullptr, nullptr, &vv_output_dims);
  if (vv_output_dims.size() > 0)
    total_dec_num = vv_output_dims.back().back();
}

void YOLOV8_new::connect_to_tcp(const std::string &ip, const int port) {
  m_sock = 0;
  struct sockaddr_in serv_addr;

  if ((m_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    std::cerr << "Socket creation failed." << std::endl;
    return;
  }

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port);

  if (inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr) <= 0) {
    std::cerr << "Invalid address / Address not supported." << std::endl;
    return;
  }

  if (connect(m_sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    std::cerr << "Connection Failed." << std::endl;
    return;
  }

  mb_sock_connected = true;
  return;
}

void YOLOV8_new::CallbackFunc(void *data) {
  std::cout << "callback from yolov8_new" << std::endl;
  Result ret = Device2Host(m_outputs);
  if (ret != SUCCESS) {
    std::cerr << "Device2Host error" << std::endl;
    return;
  }

  const int roi_hw = 32;
  const std::vector<std::vector<char>> &merge_decs = m_outputs;

  std::ostringstream oss;
  oss << m_output_dir << "rec_result_" << std::setw(6) << std::setfill('0')
      << m_imageId << ".csv";
  std::string csv_path = oss.str();

  const float *p_dec_num =
      reinterpret_cast<const float *>(merge_decs[0].data());
  // shape: [1, 1, 6, total_dec_num]
  const float *p_dec = reinterpret_cast<const float *>(merge_decs[1].data());

  std::array<std::array<float, 6>, 400> filted_decs = {0};
  const int dec_num = static_cast<int>(*p_dec_num);

  const int grid_num_x = merge_w / roi_hw;
  const int grid_num_y = merge_h / roi_hw;

  int grid_x, grid_y;
  float c_x, c_y;
  // filt decs
  for (int i = 0; i < dec_num; ++i) {
    const float *p = p_dec + i;
    float x0 = p[0 * total_dec_num];
    float y0 = p[1 * total_dec_num];
    float x1 = p[2 * total_dec_num];
    float y1 = p[3 * total_dec_num];
    float conf = p[4 * total_dec_num];
    float cl = p[5 * total_dec_num];

    if (x0 < 1.0f) {
      x0 *= merge_w;
      y0 *= merge_h;
      x1 *= merge_w;
      y1 *= merge_h;
    }

    c_x = 0.5f * (x0 + x1);
    c_y = 0.5f * (y0 + y1);
    grid_x = static_cast<int>(c_x / roi_hw);
    grid_y = static_cast<int>(c_y / roi_hw);
    const int filted_id = grid_y * grid_num_x + grid_x;
    const float &current_best_conf = filted_decs[filted_id][4];

    if (filted_id < 100 && conf > current_best_conf) {
      float w = x1 - x0;
      float h = y1 - y0;
      filted_decs[filted_id] = {c_x, c_y, w, h, conf, cl};
    }
  }

  // change to real location
  std::vector<std::vector<float>> real_decs;
  const int instance_num = m_toplefts.size();

  for (int i = 0; i < instance_num; ++i) {
    const auto &tl = m_toplefts[i];
    const auto &dec = filted_decs[i];
    grid_x = i % grid_num_x;
    grid_y = i / grid_num_x;
    if (dec[4] > 0) {
      c_x = dec[0] - roi_hw * grid_x + tl.first;
      c_y = dec[1] - roi_hw * grid_y + tl.second;
    } else {
      c_x = (float)tl.first;
      c_y = (float)tl.second;
    }
    real_decs.push_back({c_x, c_y, dec[2], dec[3], dec[4], dec[5]});
    // std::cout << "x, y, w, h, conf, cls" << std::endl;
    // std::cout << c_x << ", " << c_y << ", " << dec[2] << ", " << dec[3]
    // << ", " << dec[4] << ", " << dec[5] << std::endl;
  }

  if (mb_sock_connected) {
    std::string fileName = "decs_image_" + std::to_string(m_imageId) + ".bin";
    send_file_and_data(m_sock, fileName, real_decs);
  }
  if (mb_save_results) {
    save_detect_results(real_decs, m_output_dir, m_imageId);
  }
}

YOLOV8_new::~YOLOV8_new() {
  if (mb_sock_connected) {
    close(m_sock);
    mb_sock_connected = false; // 更新连接状态
  }
}
