#include "ive_md.hpp"
#include "ot_common_ive.h"
#include "ot_type.h"
#include "svp_model_pingpong.hpp"
#include "svp_yolov8.hpp"
#include "utils.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

void compute_real_rec(const std::vector<std::vector<char>> &merge_decs,
                      const std::vector<std::pair<int, int>> &top_lefts,
                      std::vector<std::vector<float>> &real_decs,
                      const int roi_hw, const int merge_h, const int merge_w,
                      const int total_dec_num,
                      const std::string &csv_path);

void merge_rois(const unsigned char *img, ot_ive_ccblob *p_blob,
                std::vector<unsigned char> &merged_rois,
                std::vector<std::pair<int, int>> &top_lefts, float scale_x,
                float scale_y, int imgH, int imgW, int merged_roi_H,
                int merged_roi_W);

int main() {
  const std::string file_path = "./data/input/md/md_1920x1080.bin";
  const std::string omPath = "/home/liuyang/Documents/haisi/ai-sd3403/"
                             "ai-sd3403/models/yolov8n_640x640_rpn_original.om";
  const int roi_hw = 32;
  const int roi_size = roi_hw * roi_hw * 1.5; // YUV420sp
  const int merged_hw = roi_hw * 20;
  const int merged_size = merged_hw * merged_hw * 1.5;
  std::vector<unsigned char> merged_roi(merged_size, 0);
  ot_ive_ccblob blob = {0};

  char absolute_path[PATH_MAX];

  // 将相对路径转换为绝对路径
  if (realpath(file_path.c_str(), absolute_path) == NULL) {
    perror("realpath");
    return 1;
  }

  std::cout << "Absolute path: " << absolute_path << std::endl;

  IVE_MD md;
  std::ifstream inputFile(absolute_path, std::ios::binary);
  if (!inputFile) {
    std::cerr << "Failed to open images.bin" << std::endl;
    return 1;
  }

  // 获取文件大小
  inputFile.seekg(0, std::ios::end);
  std::streampos fileSize = inputFile.tellg();
  inputFile.seekg(0, std::ios::beg);

  // 计算图片数量
  const int IMAGE_SIZE = OT_SAMPLE_MD_WIDTH * OT_SAMPLE_MD_HEIGHT;
  int numberOfImages = fileSize / IMAGE_SIZE;

  // 初始化NPU
  SVPYOLOV8 yolov8(omPath);
  std::vector<size_t> outbuf_size;
  yolov8.GetOutBufferSize(outbuf_size);
  std::vector<std::vector<size_t>> vv_out_dims;
  yolov8.GetOutDims(vv_out_dims);
  std::vector<std::vector<char>> outputs(outbuf_size.size());
  const int total_dec_num = vv_out_dims[1].back();

  for (size_t i = 0; i < outbuf_size.size(); ++i) {
    outputs[i].resize(outbuf_size[i], 0);
  }

  for (int i = 0; i < numberOfImages; ++i) {
    std::vector<unsigned char> img(IMAGE_SIZE);
    inputFile.read(reinterpret_cast<char *>(img.data()), IMAGE_SIZE);
    if (inputFile.fail()) {
      std::cerr << "Failed to read image " << i << std::endl;
      return 1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    md.process(img.data(), &blob);
    std::cout << "instance number: " << static_cast<int>(blob.info.bits.rgn_num)
              << std::endl;

    // 合并ROI
    std::vector<std::pair<int, int>> top_lefts;
    merge_rois(img.data(), &blob, merged_roi, top_lefts, 4.0f, 4.0f, 1080, 1920, merged_hw,
               merged_hw);

    // 输入到NPU, 推理
    yolov8.Host2Device(reinterpret_cast<char *>(merged_roi.data()),
                       merged_size);
    yolov8.ExecuteRPN(outputs);

    // convert to real decs
    std::vector<std::vector<float>> real_decs;

    std::ostringstream oss;
    oss << "rec_result_" << std::setw(6) << std::setfill('0') << i << ".csv";
    std::string file_name = oss.str();
    compute_real_rec(outputs, top_lefts, real_decs, roi_hw,
                     merged_hw, merged_hw, total_dec_num, file_name);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "------------duration: " << duration.count()
              << " milliseconds----------------" << std::endl;
  }

  return 0;
}


void compute_real_rec(const std::vector<std::vector<char>> &merge_decs,
                      const std::vector<std::pair<int, int>> &top_lefts,
                      std::vector<std::vector<float>> &real_decs,
                      const int roi_hw, const int merge_h, const int merge_w,
                      const int total_dec_num,
                      const std::string &csv_path) {
    const float *p_dec_num = reinterpret_cast<const float*>(merge_decs[0].data());
    // shape: [1, 1, 6, total_dec_num]
    const float *p_dec = reinterpret_cast<const float*>(merge_decs[1].data());

    std::array<std::array<float, 6>, 400> filted_decs = {0};
    const int dec_num = static_cast<int>(*p_dec_num);

    const int grid_num_x = merge_w / roi_hw;
    const int grid_num_y = merge_h / roi_hw;

    int grid_x, grid_y;
    float c_x, c_y;
    // filt decs
    for (int i = 0; i < dec_num; ++i) {
        const float* p = p_dec + i;
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
    real_decs.clear();
    const int instance_num = top_lefts.size();
    std::ofstream file(csv_path);
    if (file.is_open()) {
      file << "x,y,w,h,conf,cls\n";
    } else {
      std::cerr << "Unable to open file: " << csv_path << std::endl;
    }

    for (int i = 0; i < instance_num; ++i) {
      const auto &tl = top_lefts[i];
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
      if (file.is_open()) {
        file << c_x << "," << c_y << "," << dec[2] << "," << dec[3] << "," << dec[4] << "," << dec[5] << "\n";
      }
    }

    if (file.is_open()){
      file.close();
    }
}

void merge_rois(const unsigned char *img, ot_ive_ccblob *p_blob,
                std::vector<unsigned char> &merged_rois,
                std::vector<std::pair<int, int>> &top_lefts, float scale_x,
                float scale_y, int imgH, int imgW, int merged_roi_H,
                int merged_roi_W) {
  const int roi_size = 32;
  const int rgn_num =
      std::min(static_cast<int>(p_blob->info.bits.rgn_num), 200);
  const int grid_x_n = merged_roi_W / roi_size;
  const int grid_y_n = merged_roi_H / roi_size;

  std::fill(merged_rois.begin(), merged_rois.end(), 0);
  for (int i = 0; i < rgn_num; ++i) {
    const int grid_x = i % grid_x_n;
    const int grid_y = i / grid_x_n;
    const int offset_x = grid_x * roi_size;
    const int offset_y = grid_y * roi_size;

    const unsigned int center_x =
        static_cast<unsigned int>(scale_x * 0.5f *
                                  (static_cast<float>(p_blob->rgn[i].left) +
                                   static_cast<float>(p_blob->rgn[i].right)));
    const unsigned int center_y =
        static_cast<unsigned int>(scale_y * 0.5f *
                                  (static_cast<float>(p_blob->rgn[i].top) +
                                   static_cast<float>(p_blob->rgn[i].bottom)));

    const int w_start = std::max((int)center_x - roi_size / 2, 0);
    const int w_end =
        std::min(center_x + roi_size / 2, static_cast<unsigned int>(imgW));
    const int h_start = std::max((int)center_y - roi_size / 2, 0);
    const int h_end =
        std::min(center_y + roi_size / 2, static_cast<unsigned int>(imgH));

    top_lefts.push_back(std::make_pair(w_start, h_start));
    if (w_start < 0) {
      std::cout << "----------------w_start: " << w_start << std::endl;
    }
    // 预计算索引
    unsigned int roi_offset = offset_y * merged_roi_W + offset_x;
    unsigned int img_offset = h_start * imgW;
    for (unsigned int h = h_start; h < h_end; ++h) {
      for (unsigned int w = w_start, roi_x = 0; w < w_end; ++w, ++roi_x) {
        // 检查是否越界
        if (roi_offset + roi_x < merged_rois.size() &&
            img_offset + w < imgH * imgW) {
          merged_rois[roi_offset + roi_x] = img[img_offset + w];
        }
      }
      roi_offset += merged_roi_W;
      img_offset += imgW;
    }
  }
}
