#include "ive_md.hpp"
#include "ot_common_ive.h"
#include "ot_type.h"
#include "svp_model_pingpong.hpp"
#include "svp_yolov8.hpp"
#include "svp_yolov8_callback.hpp"
#include "utils.hpp"
#include <algorithm>
#include <chrono>
#include <climits>
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

void merge_rois(const unsigned char *img, ot_ive_ccblob *p_blob,
                std::vector<unsigned char> &merged_rois,
                std::vector<std::pair<int, int>> &top_lefts, float scale_x,
                float scale_y, int imgH, int imgW, int merged_roi_H,
                int merged_roi_W);

int save_merged_rois(const std::vector<unsigned char> &merged_roi,
                     const std::string &out_dir, const int imgId);

std::pair<std::ifstream, int> process_file(const std::string &file_path, const int IMAGE_SIZE);

class YOLOV8_new : public SVPYOLOV8_CALLBACK {
public:
  YOLOV8_new(const std::string &modelPath, const std::string &output_dir = "./",
             const std::string &aclJSON = "");

  std::string m_output_dir;
  std::vector<std::pair<int, int>> m_toplefts;
  std::vector<std::vector<char>> m_outputs;
  int m_imageId;
  int merge_h, merge_w, total_dec_num;

  void CallbackFunc(void *data) override;
  void update_imageId(int id) { m_imageId = id; }
};

int main(int argc, char *argv[]) {
  std::string file_path = "./data/input/md/1080p.bin";
  std::string high_resolution_path = "./data/input/md/4k_yuv420sp.bin";
  std::string omPath = "/home/liuyang/Documents/haisi/ai-sd3403/"
                       "ai-sd3403/models/yolov8n_640x640_rpn_original.om";
  std::string output_dir = "./";
  
  if (argc > 1)
    file_path = std::string(argv[1]);

  if (argc > 2)
    omPath = std::string(argv[2]);

  if (argc > 3)
    output_dir = std::string(argv[3]);

  const int roi_hw = 32;
  const int roi_size = roi_hw * roi_hw * 1.5; // YUV420sp
  const int merged_hw = roi_hw * 20;
  const int merged_size = merged_hw * merged_hw * 1.5;
  std::vector<unsigned char> merged_roi(merged_size, 0);
  ot_ive_ccblob blob = {0};

  char absolute_path[PATH_MAX];

  const int IMAGE_SIZE = OT_SAMPLE_MD_WIDTH * OT_SAMPLE_MD_HEIGHT;
  auto [inputFile, numberOfImages] = process_file(file_path, IMAGE_SIZE);
  const int IMAGE_SIZE2 = OT_SAMPLE_MD_WIDTH * OT_SAMPLE_MD_HEIGHT * 4 * 1.5;
  auto [inputFile2, numberOfImages2] = process_file(high_resolution_path, IMAGE_SIZE2);

  if (numberOfImages > numberOfImages2)
    numberOfImages = numberOfImages2;

  // initialize md
  // NOTE: md should initialized before SVPNNN
  IVE_MD md;

  // 初始化NPU
  YOLOV8_new yolov8(omPath, output_dir);

  Result sync_flag;
  for (int i = 0; i < numberOfImages; ++i) {
    std::vector<unsigned char> img(IMAGE_SIZE);
    inputFile.read(reinterpret_cast<char *>(img.data()), IMAGE_SIZE);
    if (inputFile.fail()) {
      std::cerr << "Failed to read image " << i << std::endl;
      return 1;
    }
    std::vector<unsigned char> img_high(IMAGE_SIZE2);
    inputFile2.read(reinterpret_cast<char *>(img_high.data()), IMAGE_SIZE2);
    if (inputFile2.fail()) {
      std::cerr << "Failed to read image2 " << i << std::endl;
      return 1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    md.process(img.data(), &blob);
    std::cout << "instance number: " << static_cast<int>(blob.info.bits.rgn_num)
              << std::endl;

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
    yolov8.update_imageId(i);
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

  return 0;
}

void merge_rois(const unsigned char *img, ot_ive_ccblob *p_blob,
                std::vector<unsigned char> &merged_rois,
                std::vector<std::pair<int, int>> &top_lefts, float scale_x,
                float scale_y, int imgH, int imgW, int merged_roi_H,
                int merged_roi_W) {
  // img and merged_rois are YUV format images.
  const int roi_size = 32;
  const int roi_size_half = roi_size / 2;
  const int rgn_num =
      std::min(static_cast<int>(p_blob->info.bits.rgn_num), 200);
  const int num_rois_per_row = merged_roi_W / roi_size;

  const int Y_size = merged_roi_H * merged_roi_W;
  const int UV_size = Y_size / 2;
  // fill YUV to gray image
  std::fill(merged_rois.begin(), merged_rois.begin() + Y_size, 0);
  std::fill(merged_rois.begin() + Y_size, merged_rois.end(), 128);
  for (int i = 0; i < rgn_num; ++i) {
    const int offset_x = i % num_rois_per_row * roi_size;
    const int offset_y = i / num_rois_per_row * roi_size;

    auto &rgn = p_blob->rgn[i];
    const int center_x =
        scale_x * static_cast<int>(0.5f * (static_cast<float>(rgn.left) +
                                           static_cast<float>(rgn.right)));
    const int center_y =
        scale_y * static_cast<int>(0.5f * (static_cast<float>(rgn.top) +
                                           static_cast<float>(rgn.bottom)));

    const int w_start = std::max(center_x - roi_size_half, 0);
    const int w_end = std::min(center_x + roi_size_half, imgW);
    const int h_start = std::max(center_y - roi_size_half, 0);
    const int h_end = std::min(center_y + roi_size_half, imgH);

    top_lefts.push_back(std::make_pair(w_start, h_start));
    if (w_start < 0) {
      std::cout << "----------------w_start: " << w_start << std::endl;
    }
    // 预计算索引
    unsigned int roi_offset = offset_y * merged_roi_W + offset_x;
    unsigned int img_offset = h_start * imgW;
    unsigned int roi_uv_offset = (merged_roi_H + offset_y * 0.5) * merged_roi_W + offset_x;
    unsigned int img_uv_offset = (imgH + h_start * 0.5) * imgW;
    for (unsigned int h = h_start; h < h_end; ++h) {
      for (unsigned int w = w_start, roi_x = 0; w < w_end; ++w, ++roi_x) {
        // 检查是否越界
        if (roi_offset + roi_x < merged_rois.size() &&
            img_offset + w < imgH * imgW) {
          merged_rois[roi_offset + roi_x] = img[img_offset + w];
          // The number of rows in the Y channel is twice of the UV channel.
          if (h % 2 == 0) {
            // uv channel has the same w with Y channel
            merged_rois[roi_uv_offset + roi_x] = img[img_uv_offset + w];
          }
        }
      }
      roi_offset += merged_roi_W;
      img_offset += imgW;
      if (h % 2 == 0) {
        roi_uv_offset += merged_roi_W;
        img_uv_offset += imgW;
      }
    }
  }
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
    int filted_id = grid_y * grid_num_x + grid_x;

    if (filted_id < 100 && conf > filted_decs[filted_id][4]) {
      float w = x1 - x0;
      float h = y1 - y0;
      filted_decs[filted_id] = {c_x, c_y, w, h, conf, cl};
    }
  }

  // change to real location
  std::vector<std::vector<float>> real_decs;
  const int instance_num = m_toplefts.size();
  // std::ofstream file(csv_path);
  // if (file.is_open()) {
  //   file << "x,y,w,h,conf,cls\n";
  // } else {
  //   std::cerr << "Unable to open file: " << csv_path << std::endl;
  // }

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
    // if (file.is_open()) {
    //   file << c_x << "," << c_y << "," << dec[2] << "," << dec[3] << ","
    //        << dec[4] << "," << dec[5] << "\n";
    // }
  }

  // if (file.is_open()) {
  //   file.close();
  // }
}

int save_merged_rois(const std::vector<unsigned char> &merged_roi,
                     const std::string &out_dir, const int imgId) {

  std::ostringstream oss;
  oss << out_dir << "merged_roi_" << std::setw(6) << std::setfill('0') << imgId
      << ".bin";
  std::string out_name = oss.str();
  // 打开一个二进制文件以写入
  std::ofstream outFile(out_name, std::ios::binary);
  if (!outFile) {
    std::cerr << "Error opening file for writing." << std::endl;
    return 1;
  }

  // 将数据写入二进制文件
  outFile.write(reinterpret_cast<const char *>(merged_roi.data()),
                merged_roi.size() * sizeof(unsigned char));

  // 关闭文件
  outFile.close();

  return 0;
}

std::pair<std::ifstream, int> process_file(const std::string &file_path, const int IMAGE_SIZE) {
  char absolute_path[PATH_MAX];

  // step1: 将相对路径转换为绝对路径
  if (realpath(file_path.c_str(), absolute_path) == NULL) {
    perror("realpath");
    return {std::ifstream(), 0};
  }
  std::cout << "Absolute path: " << absolute_path << std::endl;

  // step2: 打开文件
  std::ifstream inputFile(absolute_path, std::ios::binary);
  if (!inputFile) {
    std::cerr << "Failed to open images.bin" << std::endl;
    return {std::ifstream(), 0};
  }

  // step3: 计算图像数量
  inputFile.seekg(0, std::ios::end);
  std::streampos fileSize = inputFile.tellg();
  inputFile.seekg(0, std::ios::beg);

  // const int IMAGE_SIZE = OT_SAMPLE_MD_WIDTH * OT_SAMPLE_MD_HEIGHT;
  int numberOfImages = fileSize / IMAGE_SIZE;

  return {std::move(inputFile), numberOfImages};
}
