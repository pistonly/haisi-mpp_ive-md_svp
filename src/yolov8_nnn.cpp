#include "yolov8_nnn.hpp"
#include "utils.hpp"
#include <iomanip>
#include <netinet/in.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <vector>

extern Logger logger;

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

  GetModelInfo(nullptr, &m_input_h, &m_input_w, nullptr, nullptr,
               &mv_outputs_dim);
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
  auto d2h_start = std::chrono::high_resolution_clock::now();
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
    const int grid_num_x = m_input_w / roi_hw;
    const int grid_num_y = m_input_h / roi_hw;
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

    std::stringstream ss;
    ss << "decs_image_" << std::setw(6) << std::setfill('0') << m_imageId << "_"
       << m_timestamp << ".bin";
    if (mb_sock_connected) {
      send_file_and_data(m_sock, ss.str(), real_decs);
    }
    if (mb_save_results) {
      save_detect_results(real_decs, m_output_dir, ss.str());
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

YOLOV8_nnn_2chns::YOLOV8_nnn_2chns(const std::string &modelPath,
                                   const std::string &output_dir,
                                   const std::string &aclJSON)
    : NNNYOLOV8_CALLBACK(modelPath, aclJSON), m_imageId_0(0), m_imageId_1(0) {
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

  GetModelInfo(nullptr, &m_input_h, &m_input_w, nullptr, nullptr,
               &mv_outputs_dim);
  for (auto &dim_i : mv_outputs_dim) {
    std::stringstream ss;
    ss << "out dim: " << std::endl;
    for (auto dim_i_j : dim_i) {
      ss << dim_i_j << ", ";
    }
    logger.log(INFO, ss.str());
  }
}

void YOLOV8_nnn_2chns::connect_to_tcp(const std::string &ip, const int port) {
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

void YOLOV8_nnn_2chns::update_imageId(int id, int ch) {
  if (ch == 0) {
    m_current_ch = 0;
    m_imageId_0 = id;
  } else if (ch == 1) {
    m_current_ch = 1;
    m_imageId_1 = id;
  }
}

void YOLOV8_nnn_2chns::update_imageId(int id, uint64_t time_stamp, int ch) {
  if (ch == 0) {
    m_current_ch = 0;
    m_imageId_0 = id;
    m_timestamp_0 = time_stamp;
  } else if (ch == 1) {
    m_current_ch = 1;
    m_imageId_1 = id;
    m_timestamp_1 = time_stamp;
  }
}

void YOLOV8_nnn_2chns::CallbackFunc(void *data) {
  logger.log(DEBUG, "callback from yolov8_new");
  auto d2h_start = std::chrono::high_resolution_clock::now();
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
    const int grid_num_x = m_input_w / roi_hw;
    const int grid_num_y = m_input_h / roi_hw;
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

    std::stringstream ss;
    if (m_current_ch == 0) {
      ss << "decs_image_ch-0_" << std::setw(6) << std::setfill('0')
         << m_imageId_0 << "_" << m_timestamp_0 << ".bin";
    } else {
      ss << "decs_image_ch-1_" << std::setw(6) << std::setfill('0')
         << m_imageId_1 << "_" << m_timestamp_1 << ".bin";
    }

    if (mb_sock_connected) {
      send_file_and_data(m_sock, ss.str(), real_decs);
    }
    if (mb_save_results) {
      save_detect_results(real_decs, m_output_dir, ss.str());
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

YOLOV8_nnn_2chns::~YOLOV8_nnn_2chns() {
  if (mb_sock_connected) {
    close(m_sock);
    mb_sock_connected = false;
    logger.log(INFO, "Disconnected from TCP server.");
  }
}

YOLOV8_combine::YOLOV8_combine(const std::string &modelPath,
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

  GetModelInfo(nullptr, &m_input_h, &m_input_w, nullptr, nullptr,
               &mv_outputs_dim);
  for (auto &dim_i : mv_outputs_dim) {
    std::stringstream ss;
    ss << "out dim: " << std::endl;
    for (auto dim_i_j : dim_i) {
      ss << dim_i_j << ", ";
    }
    logger.log(INFO, ss.str());
  }
}

void YOLOV8_combine::connect_to_tcp(const std::string &ip, const int port) {
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

void YOLOV8_combine::CallbackFunc(void *data) {
  logger.log(DEBUG, "callback from yolov8_new");
  auto d2h_start = std::chrono::high_resolution_clock::now();
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
    const int grid_num_x = m_input_w / roi_hw;
    const int grid_num_y = m_input_h / roi_hw;
    float c_x, c_y, w, h, conf;
    int grid_x, grid_y;
    int grid_group_x, grid_group_y;
    int grid_sub_x, grid_sub_y;
    std::array<std::array<std::array<float, 6>, 400>, 4> v_filted_decs = {0.f};
    for (auto j = 0; j < det_bbox[i].size(); ++j) {
      // //debug
      // std::cout << "here" << std::endl;
      const std::vector<half> &box = det_bbox[i][j];
      c_x = 0.5f * (box[0] + box[2]);
      c_y = 0.5f * (box[1] + box[3]);
      conf = det_conf[i][j];
      grid_x = static_cast<int>(c_x / roi_hw);
      grid_y = static_cast<int>(c_y / roi_hw);
      grid_group_x = grid_x / 20;
      grid_group_y = grid_y / 20;
      grid_sub_x = grid_x % 20;
      grid_sub_y = grid_y % 20;

      const int group_id = grid_group_y * 2 + grid_group_x;
      const int filted_id = grid_sub_y * 20 + grid_sub_x;
      auto &filted_dec_i = v_filted_decs[group_id][filted_id];
      const float &current_best_conf = filted_dec_i[4];
      std::cout << "filted_id: " << filted_id << std::endl;
      if (filted_id < 400 && conf > current_best_conf) {
        float w = box[2] - box[0];
        float h = box[3] - box[1];
        filted_dec_i = {c_x, c_y, w, h, conf, det_cls[i][j]};
      }
    }
    // change to real location
    std::vector<std::vector<float>> real_decs;
    for (int group_i = 0; group_i<4; ++group_i){
      grid_group_x = group_i % 2;
      grid_group_y = group_i / 2;
      const int instance_num = mvv_toplefts4[group_i].size();
      for (int k = 0; k < instance_num; ++k){
        const auto& tl = mvv_toplefts4[group_i][k];
        const auto &dec = v_filted_decs[group_i][k];
        grid_sub_x = k % 20;
        grid_sub_y = k / 20;
        grid_x = grid_group_x * 20 + grid_sub_x;
        grid_y = grid_group_y * 20 + grid_sub_y;
        if (dec[4] > 0){
          c_x = dec[0] - roi_hw * grid_x + tl.first;
          c_y = dec[0] - roi_hw * grid_y + tl.second;
        } else{
          c_x = static_cast<float>(tl.first);
          c_y = static_cast<float>(tl.second);
        }
        real_decs.push_back({c_x, c_y, dec[2], dec[3], dec[4], dec[5]});
      }
    }

    std::stringstream ss;
    ss << "decs_image_" << std::setw(6) << std::setfill('0') << m_imageId << "_"
       << m_timestamp << ".bin";
    if (mb_sock_connected) {
      send_file_and_data(m_sock, ss.str(), real_decs);
    }
    if (mb_save_results) {
      save_detect_results(real_decs, m_output_dir, ss.str());
    }
  }
  auto postp_end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      postp_end - postp_start);
  auto d2h_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
      postp_start - d2h_start);

  logger.log(DEBUG, "Post-processing cost: ", duration.count(),
             "ms, D2H cost: ", d2h_cost.count(), "ms");
  mb_yolo_ready = true;
}

YOLOV8_combine::~YOLOV8_combine() {
  if (mb_sock_connected) {
    close(m_sock);
    mb_sock_connected = false;
    logger.log(INFO, "Disconnected from TCP server.");
  }
}
