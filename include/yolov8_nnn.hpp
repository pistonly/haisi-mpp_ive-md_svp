#ifndef YOLOV8_NEW_HPP
#define YOLOV8_NEW_HPP
#include "nnn_yolov8.hpp"
#include "nnn_yolov8_callback.hpp"
#include "post_process_tools.hpp"
#include <cstdint>
#include <half.hpp>
#include <string>
#include <vector>
#include "tcp_tools.hpp"

using half_float::half;

static float default_conf_thres = 0.5;
static float default_iou_thres = 0.6;
static float default_max_det = 300;

class YOLOV8_new : public NNNYOLOV8_CALLBACK, public TCP {
public:
  YOLOV8_new(const std::string &modelPath, const std::string &output_dir = "./",
             const std::string &aclJSON = "");

  std::string m_output_dir;
  std::vector<std::pair<int, int>> m_toplefts;
  std::vector<std::vector<char>> m_outputs;
  std::vector<std::vector<size_t>> mv_outputs_dim;
  int m_imageId;
  uint64_t m_timestamp = 0;
  int m_input_h, m_input_w;
  float m_conf_thres = default_conf_thres;
  float m_iou_thres = default_iou_thres;
  float m_max_det = default_max_det;
  bool mb_save_results = false;
  bool mb_save_csv = false;
  uint8_t m_cameraId = 0;

  // mvp_bbox shape: batch x branch_num x (anchors * 4)
  std::vector<std::vector<const half *>> mvp_bbox;
  // mvp_conf shape: batch x branch_num x anchors
  std::vector<std::vector<const half *>> mvp_conf;
  // mvp_cls shape: batch x branch_num x anchors
  std::vector<std::vector<const half *>> mvp_cls;


  void CallbackFunc(void *data) override;
  void update_imageId(int imageId, uint64_t time_stamp, int cameraId=0);
};

/*
 * add mb_yolo_ready, mvv_toplefts4, callbackfunc
 */
class YOLOV8_combine : public NNNYOLOV8_CALLBACK, public TCP {
public:
  YOLOV8_combine(const std::string &modelPath,
                 const std::string &output_dir = "./",
                 const std::string &aclJSON = "");

  std::string m_output_dir;
  std::vector<std::vector<std::pair<int, int>>> mvv_toplefts4;
  std::vector<std::vector<char>> m_outputs;
  std::vector<std::vector<size_t>> mv_outputs_dim;
  int m_imageId;
  uint8_t m_cameraId;
  uint64_t m_timestamp = 0;
  int m_input_h, m_input_w;
  float m_conf_thres = default_conf_thres;
  float m_iou_thres = default_iou_thres;
  float m_max_det = default_max_det;
  bool mb_save_results = false;
  bool mb_save_csv = false;
  bool mb_yolo_ready = true;

  // mvp_bbox shape: batch x branch_num x (anchors * 4)
  std::vector<std::vector<const half *>> mvp_bbox;
  // mvp_conf shape: batch x branch_num x anchors
  std::vector<std::vector<const half *>> mvp_conf;
  // mvp_cls shape: batch x branch_num x anchors
  std::vector<std::vector<const half *>> mvp_cls;


  void CallbackFunc(void *data) override;
  void update_imageId(int id, uint64_t timestamp, uint8_t cameraId=0) {
    m_cameraId = cameraId;
    m_imageId = id;
    m_timestamp = timestamp;
  }
};


class YOLOV8Sync : public NNNYOLOV8, public TCP {
public:
  YOLOV8Sync(const std::string &modelPath,
                     const std::string &output_dir = "./",
                     const std::string &aclJSON = "");

  std::string m_output_dir;
  uint64_t m_timestamp = 0;
  bool mb_save_results = false;
  bool mb_save_csv = false;
  bool mb_yolo_ready = true;

  std::vector<std::vector<size_t>> mv_outputs_dim;
  int m_input_h, m_input_w;

  // mvp_bbox shape: batch x branch_num x (anchors * 4)
  std::vector<std::vector<const half *>> mvp_bbox;
  // mvp_conf shape: batch x branch_num x anchors
  std::vector<std::vector<const half *>> mvp_conf;
  // mvp_cls shape: batch x branch_num x anchors
  std::vector<std::vector<const half *>> mvp_cls;

  void post_process(std::vector<std::vector<std::vector<half>>> &det_bbox,
                    std::vector<std::vector<half>> &det_conf,
                    std::vector<std::vector<half>> &det_cls);
  bool process_one_image(
      const std::vector<unsigned char> &input_yuv,
      const std::vector<std::pair<int, int>> &v_toplefts,
      uint8_t cameraId, int imageId, uint64_t timestamp);

  void set_postprocess_parameters(float conf_thres, float iou_thres,
                                  int max_det);

  float m_conf_thres = default_conf_thres;
  float m_iou_thres = default_iou_thres;
  int m_max_det = default_max_det;
  int m_imageId = 0;
};

class YOLOV8Sync_combine : public YOLOV8Sync {
public:
  YOLOV8Sync_combine(const std::string &modelPath,
                     const std::string &output_dir = "./",
                     const std::string &aclJSON = "");

  bool process_one_image(
      const std::vector<unsigned char> &input_yuv,
      const std::vector<std::vector<std::pair<int, int>>> &vv_toplefts4,
      uint8_t cameraId, int imageId, uint64_t timestamp);
};

#endif
