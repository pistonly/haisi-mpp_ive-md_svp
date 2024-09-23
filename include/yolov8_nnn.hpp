#ifndef YOLOV8_NEW_HPP
#define YOLOV8_NEW_HPP
#include "nnn_yolov8_callback.hpp"
#include "post_process_tools.hpp"
#include <half.hpp>
#include <string>
#include <vector>

using half_float::half;

static float default_conf_thres = 0.5;
static float default_iou_thres = 0.6;
static float default_max_det = 300;

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
  int m_input_h, m_input_w;
  float m_conf_thres = default_conf_thres;
  float m_iou_thres = default_iou_thres;
  float m_max_det = default_max_det;
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

class YOLOV8_nnn_2chns : public NNNYOLOV8_CALLBACK {
public:
  YOLOV8_nnn_2chns(const std::string &modelPath,
                   const std::string &output_dir = "./",
                   const std::string &aclJSON = "");
  ~YOLOV8_nnn_2chns();

  std::string m_output_dir;
  std::vector<std::pair<int, int>> m_toplefts;
  std::vector<std::vector<char>> m_outputs;
  std::vector<std::vector<size_t>> mv_outputs_dim;
  int m_imageId_0, m_imageId_1;
  int m_input_h, m_input_w;
  float m_conf_thres = default_conf_thres;
  float m_iou_thres = default_iou_thres;
  float m_max_det = default_max_det;
  int m_sock;
  bool mb_sock_connected = false;
  bool mb_save_results = false;
  int m_current_ch = 0;

  // mvp_bbox shape: batch x branch_num x (anchors * 4)
  std::vector<std::vector<const half *>> mvp_bbox;
  // mvp_conf shape: batch x branch_num x anchors
  std::vector<std::vector<const half *>> mvp_conf;
  // mvp_cls shape: batch x branch_num x anchors
  std::vector<std::vector<const half *>> mvp_cls;

  void connect_to_tcp(const std::string &ip, const int port);

  void CallbackFunc(void *data) override;
  void update_imageId(int id, int ch);
};

#endif
