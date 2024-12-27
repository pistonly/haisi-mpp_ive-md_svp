#include "yolov8_nnn.hpp"
#include "utils.hpp"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <netinet/in.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <vector>

extern Logger logger;

// static std::time_t lastTime = std::chrono::system_clock::to_time_t(now);

void send_save_results(bool sock_connected, bool save_bin, bool save_csv,
                       int sock, std::vector<std::vector<float>> &real_decs,
                       uint8_t cameraId, int imageId, uint64_t timestamp,
                       const std::string &output_dir) {
  real_decs.push_back({0.f, 0.f, 0.f, 0.f, 0.f, 100.f});
  if (real_decs.size() > 0) {
    std::stringstream ss;

    // get timestamp
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis =
        std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    ss << "decs_camera-" << static_cast<int>(cameraId) << "_image-"
       << std::setw(6) << std::setfill('0') << imageId << "_" << millis;
    if (sock_connected) {
      // add fake for display software
      logger.log(DEBUG, "real_decs");
      send_dection_results(sock, real_decs, cameraId, timestamp);
    }
    if (save_bin) {
      save_detect_results(real_decs, cameraId, timestamp, output_dir,
                          ss.str() + ".bin");
    }
    if (save_csv) {
      save_detect_results_csv(real_decs, output_dir, ss.str() + ".csv");
    }
  }
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

void YOLOV8_new::update_imageId(int imageId, uint64_t time_stamp,
                                uint8_t cameraId) {
  m_cameraId = cameraId;
  m_imageId = imageId;
  m_timestamp = time_stamp;
}

void YOLOV8_new::CallbackFunc(void *data) {
  logger.log(DEBUG, "callback from yolov8_new");
  std::vector<const char *> vp_outputs;
  {
    Timer timer("D2H ...");
    Result ret = Device2Host(vp_outputs);
    if (ret != SUCCESS) {
      logger.log(ERROR, "Device2Host error");
      return;
    }
  }

  // post process
  {
    Timer timer("postprocess...");
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
      float c_x, c_y, w, h, conf, x0, y0, x1, y1;
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
        const auto &blob_xyxy = m_blob_xyxy[k];
        auto &dec = filted_decs[k];
        grid_x = k % grid_num_x;
        grid_y = k / grid_num_x;
        if (dec[4] > 0) {
          c_x = dec[0] - roi_hw * grid_x + tl.first;
          c_y = dec[1] - roi_hw * grid_y + tl.second;
          x0 = c_x - dec[2] / 2;
          y0 = c_y - dec[3] / 2;
          x1 = x0 + dec[2];
          y1 = y0 + dec[3];
        } else {
          x0 = blob_xyxy[0];
          y0 = blob_xyxy[1];
          x1 = blob_xyxy[2];
          y1 = blob_xyxy[3];
          dec[5] = 100;
        }
        real_decs.push_back({x0 / 2, y0 / 2, x1 / 2, y1 / 2, dec[4], dec[5]});
      }

      // 获取当前时间点
      auto currentTime = std::chrono::steady_clock::now();
      static auto lastTime = std::chrono::steady_clock::now();
      // 计算时间间隔
      std::chrono::duration<double> elapsedSeconds = currentTime - lastTime;

      if (elapsedSeconds.count() > m_save_interval) {
        // 执行函数体
        connect_to_tcp(m_tcp_ip, m_tcp_port);
        send_save_results(mb_sock_connected, mb_save_results, mb_save_csv,
                          m_sock, real_decs, m_cameraId, m_imageId, m_timestamp,
                          m_output_dir);
        if (mb_sock_connected) {
          close(m_sock);
          mb_sock_connected = false;
        }

        // 更新 lastTime 为当前时间
        lastTime = currentTime;
      }
    }
  }
}

YOLOV8_combine::YOLOV8_combine(const std::string &modelPath,
                               const std::string &output_dir,
                               const std::string &aclJSON)
    : NNNYOLOV8_CALLBACK(modelPath, aclJSON), m_imageId(0), m_cameraId(0) {

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

void YOLOV8_combine::CallbackFunc(void *data) {
  logger.log(DEBUG, "callback from yolov8_new");
  std::vector<const char *> vp_outputs;
  {
    Timer timer("D2H...");
    Result ret = Device2Host(vp_outputs);
    if (ret != SUCCESS) {
      logger.log(ERROR, "Device2Host error");
      return;
    }
  }

  // post process
  {
    Timer timer("post process ...");

    split_bbox_conf_reduced(vp_outputs, mv_outputs_dim, mvp_bbox, mvp_conf,
                            mvp_cls);

    const int batch_num = mvp_bbox.size();
    std::vector<std::vector<std::vector<half>>> det_bbox(batch_num);
    std::vector<std::vector<half>> det_conf(batch_num);
    std::vector<std::vector<half>> det_cls(batch_num);

    const int roi_hw = 32;
    std::vector<std::vector<float>> real_decs;
    for (int i = 0; i < batch_num; ++i) {
      const int box_num = mv_outputs_dim[0][2];
      const std::vector<const half *> &bbox_batch_i = mvp_bbox[i];
      const std::vector<const half *> &conf_batch_i = mvp_conf[i];
      const std::vector<const half *> &cls_batch_i = mvp_cls[i];
      NMS_bboxTranspose(box_num, bbox_batch_i, conf_batch_i, cls_batch_i,
                        det_bbox[i], det_conf[i], det_cls[i], m_conf_thres,
                        m_iou_thres, m_max_det);

      // filter detection. each DEC assign to one 32x32 patch.
      assert(m_input_w / roi_hw == 40);
      assert(m_input_h / roi_hw == 40);
      float c_x, c_y, w, h, conf, x0, y0, x1, y1;
      int grid_x, grid_y;
      int grid_group_x, grid_group_y;
      int grid_sub_x, grid_sub_y;
      std::array<std::array<std::array<float, 6>, 400>, 4> v_filted_decs = {
          0.f};
      for (auto j = 0; j < det_bbox[i].size(); ++j) {
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
      for (int group_i = 0; group_i < 4; ++group_i) {
        grid_group_x = group_i % 2;
        grid_group_y = group_i / 2;
        const int instance_num = mvv_toplefts4[group_i].size();
        for (int k = 0; k < instance_num; ++k) {
          const auto &tl = mvv_toplefts4[group_i][k];
          const auto &blob_xyxy = mvv_blob_xyxy4[group_i][k];
          auto &dec = v_filted_decs[group_i][k];
          grid_sub_x = k % 20;
          grid_sub_y = k / 20;
          grid_x = grid_group_x * 20 + grid_sub_x;
          grid_y = grid_group_y * 20 + grid_sub_y;
          if (dec[4] > 0) {
            c_x = dec[0] - roi_hw * grid_x + tl.first;
            c_y = dec[0] - roi_hw * grid_y + tl.second;
            x0 = c_x - dec[2] / 2;
            y0 = c_y - dec[3] / 2;
            x1 = x0 + dec[2];
            y1 = y0 + dec[3];
          } else {
            x0 = blob_xyxy[0];
            y0 = blob_xyxy[1];
            x1 = blob_xyxy[2];
            y1 = blob_xyxy[3];
            dec[5] = 100;
          }
          real_decs.push_back({x0 / 2, y0 / 2, x1 / 2, y1 / 2, dec[4], dec[5]});
        }
      }
    }

    // 获取当前时间点
    auto currentTime = std::chrono::steady_clock::now();
    static auto lastTime = std::chrono::steady_clock::now();
    // 计算时间间隔
    std::chrono::duration<double> elapsedSeconds = currentTime - lastTime;

    if (elapsedSeconds.count() > m_save_interval) {
      // 执行函数体
      connect_to_tcp(m_tcp_ip, m_tcp_port);
      send_save_results(mb_sock_connected, mb_save_results, mb_save_csv, m_sock,
                        real_decs, m_cameraId, m_imageId, m_timestamp,
                        m_output_dir);
      if (mb_sock_connected) {
        close(m_sock);
        mb_sock_connected = false;
      }

      // 更新 lastTime 为当前时间
      lastTime = currentTime;
    }
  }
}

YOLOV8Sync::YOLOV8Sync(const std::string &modelPath,
                       const std::string &output_dir,
                       const std::string &aclJSON)
    : NNNYOLOV8(modelPath, aclJSON) {
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
  logger.log(INFO, "out num: ", outbuf_size.size());
  for (size_t i = 0; i < outbuf_size.size(); ++i) {
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

YOLOV8Sync::YOLOV8Sync(const std::string &modelPath,
                       const std::string &output_dir, const bool with_aclinit,
                       const std::string &aclJSON)
    : NNNYOLOV8(modelPath, with_aclinit, aclJSON) {
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
  logger.log(INFO, "out num: ", outbuf_size.size());
  for (size_t i = 0; i < outbuf_size.size(); ++i) {
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

void YOLOV8Sync::set_postprocess_parameters(float conf_thres, float iou_thres,
                                            int max_det) {
  m_conf_thres = conf_thres;
  m_iou_thres = iou_thres;
  m_max_det = max_det;
}

void YOLOV8Sync::post_process(
    std::vector<std::vector<std::vector<half>>> &det_bbox,
    std::vector<std::vector<half>> &det_conf,
    std::vector<std::vector<half>> &det_cls) {

  std::vector<const char *> vp_outputs;
  {
    Timer timer("post_process.D2H...");
    Result ret = Device2Host(vp_outputs);
    if (ret != SUCCESS) {
      logger.log(ERROR, "Device2Host error");
      return;
    }
  }

  {
    Timer timer("post_process.split ...");
    split_bbox_conf_reduced(vp_outputs, mv_outputs_dim, mvp_bbox, mvp_conf,
                            mvp_cls);
  }

  const int batch_num = mvp_bbox.size();
  det_bbox.resize(batch_num);
  det_conf.resize(batch_num);
  det_cls.resize(batch_num);

  for (int i = 0; i < batch_num; ++i) {
    const int box_num = mv_outputs_dim[0][2];
    const std::vector<const half *> &bbox_batch_i = mvp_bbox[i];
    const std::vector<const half *> &conf_batch_i = mvp_conf[i];
    const std::vector<const half *> &cls_batch_i = mvp_cls[i];
    NMS_bboxTranspose(box_num, bbox_batch_i, conf_batch_i, cls_batch_i,
                      det_bbox[i], det_conf[i], det_cls[i], m_conf_thres,
                      m_iou_thres, m_max_det);
  }
}

bool YOLOV8Sync::process_one_image(
    const std::vector<unsigned char> &input_yuv,
    std::vector<std::vector<std::vector<half>>> &det_bbox,
    std::vector<std::vector<half>> &det_conf,
    std::vector<std::vector<half>> &det_cls) {
  mb_yolo_ready = false;

  det_bbox.clear();
  det_conf.clear();
  det_cls.clear();
  // host to device
  {
    Timer timer("yolov8 H2D ...");
    Host2Device(input_yuv.data(), input_yuv.size());
  }

  // inference
  {
    Timer timer("yolov8 inferencing ...");
    auto t0 = std::chrono::high_resolution_clock::now();
    Execute();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    m_infer_total_time += elapsed_time;
  }

  // postprocess
  {
    Timer timer("yolov8 postprocessing ...");
    post_process(det_bbox, det_conf, det_cls);
  }

  mb_yolo_ready = true;
  m_processed_num++;
  return true;
}

bool YOLOV8Sync::process_one_image(
    const std::vector<unsigned char> &input_yuv,
    const std::vector<std::pair<int, int>> &v_toplefts,
    const std::vector<std::vector<float>> &v_blob_xyxy, uint8_t cameraId,
    int imageId, uint64_t timestamp) {

  mb_yolo_ready = false;

  std::vector<std::vector<std::vector<half>>> det_bbox;
  std::vector<std::vector<half>> det_conf;
  std::vector<std::vector<half>> det_cls;

  process_one_image(input_yuv, det_bbox, det_conf, det_cls);

  // filter detections. each DEC assign to one 32x32 patch
  int roi_hw = 32;
  const int grid_num_x = m_input_w / roi_hw;
  const int grid_num_y = m_input_h / roi_hw;
  float c_x, c_y, w, h, conf, x0, y0, x1, y1;
  int grid_x, grid_y;
  std::array<std::array<float, 6>, 400> filted_decs = {0};
  std::vector<std::vector<float>> real_decs;
  for (int i = 0; i < det_bbox.size(); ++i) {
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
    const int instance_num = static_cast<int>(v_toplefts.size());
    for (int k = 0; k < instance_num; ++k) {
      const auto &tl = v_toplefts[k];
      const auto &blob_xyxy = v_blob_xyxy[k];
      auto &dec = filted_decs[k];
      grid_x = k % grid_num_x;
      grid_y = k / grid_num_x;
      if (dec[4] > 0) {
        c_x = dec[0] - roi_hw * grid_x + tl.first;
        c_y = dec[1] - roi_hw * grid_y + tl.second;
        x0 = c_x - dec[2] / 2;
        y0 = c_y - dec[3] / 2;
        x1 = x0 + dec[2];
        y1 = y0 + dec[3];
      } else {
        if (!mb_with_md_results) {
          continue;
        }
        x0 = blob_xyxy[0];
        y0 = blob_xyxy[1];
        x1 = blob_xyxy[2];
        y1 = blob_xyxy[3];
        dec[5] = 100;
      }
      real_decs.push_back({x0 / 2, y0 / 2, x1 / 2, y1 / 2, dec[4], dec[5]});
    }
  }

  // 获取当前时间点
  auto currentTime = std::chrono::steady_clock::now();
  static auto lastTime = std::chrono::steady_clock::now();
  // 计算时间间隔
  std::chrono::duration<double> elapsedSeconds = currentTime - lastTime;

  if (elapsedSeconds.count() > m_save_interval) {
    // 执行函数体
    if (mb_tcp_send) {
      connect_to_tcp(m_tcp_ip, m_tcp_port);
    }
    send_save_results(mb_sock_connected, mb_save_results, mb_save_csv, m_sock,
                      real_decs, cameraId, imageId, timestamp, m_output_dir);
    // if (mb_sock_connected) {
    //   close(m_sock);
    //   mb_sock_connected = false;
    // }

    // 更新 lastTime 为当前时间
    lastTime = currentTime;
  }

  mb_yolo_ready = true;
  m_processed_num++;
  return true;
}

bool pt_in_sky(float c_x, float c_y,
               const std::vector<std::vector<std::vector<half>>> &sky_det_bbox
               ) {
  // only works for batch=1
  int rgn_num = sky_det_bbox.at(0).size();
  for(int i=0; i<rgn_num; ++i) {
    auto &xyxy = sky_det_bbox[0][i];
    // logger.log(INFO, "xyxy: ", xyxy[0], ", ", xyxy[1], ", ", xyxy[2], ", ", xyxy[3], "c_xy", c_x, ", ", c_y);
    if ((c_x >= xyxy[0]) && (c_x <= xyxy[2]) && (c_y >= xyxy[1]) && (c_y <= xyxy[3])){
      // logger.log(INFO, "true");
      return true;
    }
  }
  return false;
}

bool YOLOV8Sync::process_one_image(
    const std::vector<unsigned char> &input_yuv,
    const std::vector<std::pair<int, int>> &v_toplefts,
    const std::vector<std::vector<float>> &v_blob_xyxy,
    const std::vector<std::vector<std::vector<half>>> &sky_det_bbox,
    uint8_t cameraId,
    int imageId, uint64_t timestamp) {

  mb_yolo_ready = false;

  std::vector<std::vector<std::vector<half>>> det_bbox;
  std::vector<std::vector<half>> det_conf;
  std::vector<std::vector<half>> det_cls;

  process_one_image(input_yuv, det_bbox, det_conf, det_cls);

  // filter detections. each DEC assign to one 32x32 patch
  int roi_hw = 32;
  const int grid_num_x = m_input_w / roi_hw;
  const int grid_num_y = m_input_h / roi_hw;
  float c_x, c_y, w, h, conf, x0, y0, x1, y1;
  int grid_x, grid_y;
  std::array<std::array<float, 6>, 400> filted_decs = {0};
  std::vector<std::vector<float>> real_decs;
  for (int i = 0; i < det_bbox.size(); ++i) {
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
    const int instance_num = static_cast<int>(v_toplefts.size());
    for (int k = 0; k < instance_num; ++k) {
      const auto &tl = v_toplefts[k];
      const auto &blob_xyxy = v_blob_xyxy[k];
      auto &dec = filted_decs[k];
      grid_x = k % grid_num_x;
      grid_y = k / grid_num_x;
      if (dec[4] > 0) {
        c_x = dec[0] - roi_hw * grid_x + tl.first;
        c_y = dec[1] - roi_hw * grid_y + tl.second;
        x0 = c_x - dec[2] / 2;
        y0 = c_y - dec[3] / 2;
        x1 = x0 + dec[2];
        y1 = y0 + dec[3];

        if (pt_in_sky(c_x, c_y, sky_det_bbox))
          dec[5] += 10;

      } else {
        if (!mb_with_md_results) {
          continue;
        }
        x0 = blob_xyxy[0];
        y0 = blob_xyxy[1];
        x1 = blob_xyxy[2];
        y1 = blob_xyxy[3];
        dec[5] = 100;
      }
      real_decs.push_back({x0 / 2, y0 / 2, x1 / 2, y1 / 2, dec[4], dec[5]});
    }
  }

  // 获取当前时间点
  auto currentTime = std::chrono::steady_clock::now();
  static auto lastTime = std::chrono::steady_clock::now();
  // 计算时间间隔
  std::chrono::duration<double> elapsedSeconds = currentTime - lastTime;

  if (elapsedSeconds.count() > m_save_interval) {
    // 执行函数体
    if (mb_tcp_send) {
      connect_to_tcp(m_tcp_ip, m_tcp_port);
    }
    send_save_results(mb_sock_connected, mb_save_results, mb_save_csv, m_sock,
                      real_decs, cameraId, imageId, timestamp, m_output_dir);
    // if (mb_sock_connected) {
    //   close(m_sock);
    //   mb_sock_connected = false;
    // }

    // 更新 lastTime 为当前时间
    lastTime = currentTime;
  }

  mb_yolo_ready = true;
  m_processed_num++;
  return true;
}

YOLOV8Sync_combine::YOLOV8Sync_combine(const std::string &modelPath,
                                       const std::string &output_dir,
                                       const std::string &aclJSON)
    : YOLOV8Sync(modelPath, output_dir, aclJSON) {}

bool YOLOV8Sync_combine::process_one_image(
    const std::vector<unsigned char> &input_yuv,
    const std::vector<std::vector<std::pair<int, int>>> &vv_toplefts4,
    const std::vector<std::vector<std::vector<float>>> &vv_blob_xyxy4,
    uint8_t cameraId, int imageId, uint64_t timestamp) {

  mb_yolo_ready = false;

  std::vector<std::vector<std::vector<half>>> det_bbox;
  std::vector<std::vector<half>> det_conf;
  std::vector<std::vector<half>> det_cls;
  // host to device
  {
    Timer timer("yolov8 H2D ...");
    Host2Device(input_yuv.data(), input_yuv.size());
  }

  // inference
  {
    Timer timer("yolov8 inferencing ...");
    Execute();
  }

  // postprocess
  {
    Timer timer("yolov8 postprocessing ...");
    post_process(det_bbox, det_conf, det_cls);
  }

  // filter detections. each DEC assign to one 32x32 patch
  int roi_hw = 32;
  assert(m_input_w / roi_hw == 40);
  assert(m_input_h / roi_hw == 40);
  float c_x, c_y, w, h, conf, x0, y0, x1, y1;
  int grid_x, grid_y;
  int grid_group_x, grid_group_y;
  int grid_sub_x, grid_sub_y;
  std::array<std::array<std::array<float, 6>, 400>, 4> v_filted_decs = {0.f};
  std::vector<std::vector<float>> real_decs;
  for (int i = 0; i < det_bbox.size(); ++i) {
    for (auto j = 0; j < det_bbox[i].size(); ++j) {
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
      if (filted_id < 400 && conf > current_best_conf) {
        float w = box[2] - box[0];
        float h = box[3] - box[1];
        filted_dec_i = {c_x, c_y, w, h, conf, det_cls[i][j]};
      }
    }
    // change to real location
    for (int group_i = 0; group_i < 4; ++group_i) {
      grid_group_x = group_i % 2;
      grid_group_y = group_i / 2;
      const int instance_num = static_cast<int>(vv_toplefts4[group_i].size());
      for (int k = 0; k < instance_num; ++k) {
        const auto &tl = vv_toplefts4[group_i][k];
        const auto &blob_xyxy = vv_blob_xyxy4[group_i][k];
        auto &dec = v_filted_decs[group_i][k];
        grid_sub_x = k % 20;
        grid_sub_y = k / 20;
        grid_x = grid_group_x * 20 + grid_sub_x;
        grid_y = grid_group_y * 20 + grid_sub_y;
        if (dec[4] > 0) {
          c_x = dec[0] - roi_hw * grid_x + tl.first;
          c_y = dec[1] - roi_hw * grid_y + tl.second;
          x0 = c_x - dec[2] / 2;
          y0 = c_y - dec[3] / 2;
          x1 = x0 + dec[2];
          y1 = y0 + dec[3];
        } else {
          if (!mb_with_md_results) {
            // do not send md results
            continue;
          }
          x0 = blob_xyxy[0];
          y0 = blob_xyxy[1];
          x1 = blob_xyxy[2];
          y1 = blob_xyxy[3];
          dec[5] = 100;
        }
        real_decs.push_back({x0 / 2, y0 / 2, x1 / 2, y1 / 2, dec[4], dec[5]});
      }
    }
  }

  // 获取当前时间点
  auto currentTime = std::chrono::steady_clock::now();
  static auto lastTime = std::chrono::steady_clock::now();
  // 计算时间间隔
  std::chrono::duration<double> elapsedSeconds = currentTime - lastTime;

  if (elapsedSeconds.count() > m_save_interval) {
    // 执行函数体
    if (mb_tcp_send) {
      connect_to_tcp(m_tcp_ip, m_tcp_port);
    }
    send_save_results(mb_sock_connected, mb_save_results, mb_save_csv, m_sock,
                      real_decs, cameraId, imageId, timestamp, m_output_dir);
    // if (mb_sock_connected) {
    //   close(m_sock);
    //   mb_sock_connected = false;
    // }

    // 更新 lastTime 为当前时间
    lastTime = currentTime;
  }

  mb_yolo_ready = true;
  return true;
}

bool YOLOV8Sync_combine::process_one_image_batched(
    const std::vector<std::vector<unsigned char>> &v_input_yuv4,
    const std::vector<std::vector<std::pair<int, int>>> &vv_toplefts4,
    const std::vector<std::vector<std::vector<float>>> &vv_blob_xyxy4,
    uint8_t cameraId, int imageId, uint64_t timestamp) {

  mb_yolo_ready = false;

  assert(v_input_yuv4.size() == 4);
  assert(vv_toplefts4.size() == 4);
  assert(vv_blob_xyxy4.size() == 4);

  std::vector<std::vector<float>> real_decs;
  for (int batch_i = 0; batch_i < 4; ++batch_i) {
    std::vector<std::vector<std::vector<half>>> det_bbox;
    std::vector<std::vector<half>> det_conf;
    std::vector<std::vector<half>> det_cls;
    // host to device
    {
      Timer timer("yolov8 H2D ...");
      Host2Device(v_input_yuv4[batch_i].data(), v_input_yuv4[batch_i].size());
    }
    // inference
    {
      Timer timer("yolov8 inferencing ...");
      Execute();
    }
    // postprocess
    {
      Timer timer("yolov8 postprocessing ...");
      post_process(det_bbox, det_conf, det_cls);
    }

    // filter detections. each DEC assign to one 32x32 patch
    int roi_hw = 32;
    const int grid_num_x = m_input_w / roi_hw;
    const int grid_num_y = m_input_h / roi_hw;
    float c_x, c_y, w, h, conf, x0, y0, x1, y1;
    int grid_x, grid_y;

    std::array<std::array<float, 6>, 400> filted_decs = {0};

    for (int i = 0; i < det_bbox.size(); ++i) {
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
      const int instance_num = static_cast<int>(vv_toplefts4[batch_i].size());
      for (int k = 0; k < instance_num; ++k) {
        const auto &tl = vv_toplefts4[batch_i][k];
        const auto &blob_xyxy = vv_blob_xyxy4[batch_i][k];
        auto &dec = filted_decs[k];
        grid_x = k % grid_num_x;
        grid_y = k / grid_num_x;
        if (dec[4] > 0) {
          c_x = dec[0] - roi_hw * grid_x + tl.first;
          c_y = dec[1] - roi_hw * grid_y + tl.second;
          x0 = c_x - dec[2] / 2;
          y0 = c_y - dec[3] / 2;
          x1 = x0 + dec[2];
          y1 = y0 + dec[3];
        } else {
          if (!mb_with_md_results) {
            continue;
          }
          x0 = blob_xyxy[0];
          y0 = blob_xyxy[1];
          x1 = blob_xyxy[2];
          y1 = blob_xyxy[3];
          dec[5] = 100;
          // do not send md results
        }
        real_decs.push_back({x0 / 2, y0 / 2, x1 / 2, y1 / 2, dec[4], dec[5]});
      }
    }
  }

  // 获取当前时间点
  auto currentTime = std::chrono::steady_clock::now();
  static auto lastTime = std::chrono::steady_clock::now();
  // 计算时间间隔
  std::chrono::duration<double> elapsedSeconds = currentTime - lastTime;

  if (elapsedSeconds.count() > m_save_interval) {
    // 执行函数体
    if (mb_tcp_send) {
      connect_to_tcp(m_tcp_ip, m_tcp_port);
    }
    send_save_results(mb_sock_connected, mb_save_results, mb_save_csv, m_sock,
                      real_decs, cameraId, imageId, timestamp, m_output_dir);
    // if (mb_sock_connected) {
    //   close(m_sock);
    //   mb_sock_connected = false;
    // }

    // 更新 lastTime 为当前时间
    lastTime = currentTime;
  }

  mb_yolo_ready = true;
  m_processed_num++;
  return true;
}
