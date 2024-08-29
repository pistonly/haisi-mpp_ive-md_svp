#include "ffmpeg_vdec_vpss.hpp"
#include "ive_md.hpp"
#include "nnn_yolov8_callback.hpp"
#include "ot_common_ive.h"
#include "ot_type.h"
#include "post_process_tools.hpp"
#include "svp_model_pingpong.hpp"
#include "utils.hpp"
#include <algorithm>
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
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>


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
                       "yolov8n-nnn_640x640_1_FP32.om";
  std::string tcp_ip = "172.23.24.52";
  std::string tcp_port = "8880";
  std::string output_dir = "./";

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
    : NNNYOLOV8_CALLBACK(modelPath, aclJSON), m_imageId(0) {

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
  std::cout << "out num: " << outbuf_size.size() << std::endl;
  for (size_t i = 0; i < outbuf_size.size(); ++i) {
    m_outputs[i].resize(outbuf_size[i], 0);
    std::cout << "out size of " << i << ": " << outbuf_size[i] << std::endl;
  }

  GetModelInfo(nullptr, &merge_h, &merge_w, nullptr, nullptr, &mv_outputs_dim);
  for (auto &dim_i : mv_outputs_dim) {
    std::cout << "out dim: " << std::endl;
    for (auto dim_i_j : dim_i) {
      std::cout << dim_i_j << ", ";
    }
    std::cout << std::endl;
  }
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
  // post process
  std::vector<std::vector<std::vector<std::vector<float>>>> bbox;
  std::vector<std::vector<std::vector<float>>> conf;
  std::vector<std::vector<std::vector<int>>> cls;
  const std::vector<std::vector<float>> bbox_conf =
      reinterpret_cast<std::vector<std::vector<float>> &>(m_outputs);
  split_bbox_conf(bbox_conf, mv_outputs_dim, bbox, conf, cls);
  const int batch_num = bbox.size();
  std::vector<std::vector<std::vector<float>>> det_bbox(batch_num);
  std::vector<std::vector<float>> det_conf(batch_num);
  std::vector<std::vector<int>> det_cls(batch_num);

  const int roi_hw = 32;
  for (int i = 0; i < batch_num; ++i) {
    const std::vector<std::vector<std::vector<float>>> &bbox_batch_i = bbox[i];
    const std::vector<std::vector<float>> &conf_batch_i = conf[i];
    const std::vector<std::vector<int>> &cls_batch_i = cls[i];
    NMS(bbox_batch_i, conf_batch_i, cls_batch_i, det_bbox[i], det_conf[i],
        det_cls[i], m_conf_thres, m_iou_thres, m_max_det);
    // // save to csv
    // std::ostringstream oss;
    // oss << m_output_dir << "rec_result_" << std::setw(6) << std::setfill('0')
    //     << m_imageId << "_" << i << ".csv";
    // std::string csv_path = oss.str();
    // std::ofstream file(csv_path);

    // filter detection. each DEC assign to one 32x32 patch.
    const int grid_num_x = merge_w / roi_hw;
    const int grid_num_y = merge_h / roi_hw;
    float c_x, c_y, w, h, conf;
    int grid_x, grid_y;
    std::array<std::array<float, 6>, 400> filted_decs = {0};
    for (auto j = 0; j < det_bbox[i].size(); ++j) {
      const std::vector<float> &box = det_bbox[i][j];
      c_x = 0.5f * (box[0] + box[2]);
      c_y = 0.5f * (box[1] + box[3]);
      conf = det_conf[i][j];
      grid_x = static_cast<int>(c_x / roi_hw);
      grid_y = static_cast<int>(c_y / roi_hw);
      int filted_id = grid_y * grid_num_x + grid_x;
      if (filted_id < 100 && conf > filted_decs[filted_id][4]) {
        float w = box[2] - box[0];
        float h = box[3] - box[1];
        filted_decs[filted_id] = {c_x, c_y, w, h, conf, det_cls[i][j]};
      }
    }
    // change to real location
    std::vector<std::vector<float>> real_decs;
    const int instance_num = m_toplefts.size();

    for (int i=0; i<instance_num; ++i){
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
    }

    if (mb_sock_connected) {
      std::string fileName = "decs_image_" + std::to_string(m_imageId) + ".bin";
      send_file_and_data(m_sock, fileName, real_decs);
    }
  }
}

YOLOV8_new::~YOLOV8_new() {
  if (mb_sock_connected) {
    close(m_sock);
    mb_sock_connected = false; // 更新连接状态
  }
}

