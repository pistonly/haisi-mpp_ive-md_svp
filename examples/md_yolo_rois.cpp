#include "ive_md.hpp"
#include "ot_common_ive.h"
#include "ot_type.h"
#include "svp_model_pingpong.hpp"
#include "svp_yolov8.hpp"
#include "utils.hpp"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

using namespace std;

void cut_rois(const unsigned char *img, ot_ive_ccblob *p_blob,
              std::vector<std::vector<unsigned char>> &rois, float scale_x,
              float scale_y, int imgH, int imgW);

int main() {
  std::string file_path = "./data/input/md/md_1920x1080.bin";
  std::string omPath = "/home/liuyang/Documents/haisi/ai-sd3403/ai-sd3403/"
    "models/yolov8n_32x32_rpn_original.om";
  int roi_hw = 32;
  int roi_size = 32 * 32 * 1.5; // YUV420sp
  // unsigned char rois[roi_size * 200];
  std::vector<std::vector<unsigned char>> rois(200, std::vector<unsigned char>(32 * 32 * 1.5));
  td_u16 *p_roi_num;
  ot_ive_ccblob blob={0};

  char absolute_path[PATH_MAX];

  // 将相对路径转换为绝对路径
  if (realpath(file_path.c_str(), absolute_path) == NULL) {
    perror("realpath");
    return 1;
  }

  std::cout << "Absolute path: " << absolute_path << std::endl;

  IVE_MD md = IVE_MD();

  // process
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
  const int IMAGE_SIZE =
    OT_SAMPLE_MD_WIDTH *OT_SAMPLE_MD_HEIGHT;

  int numberOfImages =
          fileSize / IMAGE_SIZE;

  // std::ofstream outFile("rois.bin", std::ios::binary);
  // if (!outFile) {
  //   std::cerr << "Failed to open file: " << "rois.bin" << std::endl;
  //   return 1;
  // }

  // NPU
  SVPYOLOV8 yolov8(omPath);

  // YoloModelInfo modelInfo;
  // svp_model.GetModelInfo(modelInfo);
  // modelInfo.save_info("model_info.txt");

  std::string output_feature_tmp;
  std::vector<size_t> outbuf_size;
  yolov8.GetOutBufferSize(outbuf_size);

  std::vector<std::vector<char>> outputs;
  for (auto i: outbuf_size) {
    outputs.push_back(std::vector<char>(i, 0));
  }

  for (int i = 0; i < 10; i++) {
    unsigned char img[IMAGE_SIZE];
    inputFile.read(reinterpret_cast<char *>(img), IMAGE_SIZE);
    if (inputFile.fail()) {
      std::cerr << "Failed to read image " << i << std::endl;
      return 1;
    }
    auto last_time = std::chrono::high_resolution_clock::now();
    md.process(img, &blob);

    std::cout << "instance number: " << blob.info.bits.rgn_num << std::endl;

    // cut roi
    cut_rois(img, &blob, rois, 4, 4, 1080, 1920);

    // // save to rois.bin
    // for (int j=0; j<blob.info.bits.rgn_num; ++j) {
    //   outFile.write(reinterpret_cast<char *>(rois[j].data()),
    //                 rois[j].size());
    // }


    // input to NPU
    for (int j = 0; j < blob.info.bits.rgn_num; ++j) {
      yolov8.Host2Device(reinterpret_cast<char *>(rois[i].data()),
                         32 * 32 * 1.5);
      yolov8.ExecuteRPN(outputs);
    }

    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - last_time);
    std::cout << "------------duration: " << duration.count()
              << " milliseconds----------------" << std::endl;
    // last_time = current_time;
  }

  // outFile.close();

  return 0;
}


void cut_rois(const unsigned char *img, ot_ive_ccblob *p_blob,
              std::vector<std::vector<unsigned char>> &rois, float scale_x,
              float scale_y, int imgH, int imgW) {
    int rgn_num = 200 < p_blob->info.bits.rgn_num ? 200 : p_blob->info.bits.rgn_num;
    unsigned int center_x, center_y;
    unsigned int w_start, w_end, h_start, h_end;

    const int roi_size = 32;
    const int roi_area = roi_size * roi_size;

    for (int i = 0; i < rgn_num; ++i) {
        center_x = static_cast<unsigned int>(scale_x * 0.5 * 
                                             (static_cast<float>(p_blob->rgn[i].left) + 
                                              static_cast<float>(p_blob->rgn[i].right)));
        center_y = static_cast<unsigned int>(scale_y * 0.5 * 
                                             (static_cast<float>(p_blob->rgn[i].top) + 
                                              static_cast<float>(p_blob->rgn[i].bottom)));
        // Get img region edge
        w_start = (center_x > roi_size / 2 ? center_x - roi_size / 2 : 0);
        w_end = std::min(center_x + roi_size / 2, static_cast<unsigned int>(imgW));
        h_start = (center_y > roi_size / 2 ? center_y - roi_size / 2 : 0);
        h_end = std::min(center_y + roi_size / 2, static_cast<unsigned int>(imgH));

        // Reference to the existing roi vector
        std::vector<unsigned char> &roi = rois[i];

        // Clear the roi vector
        std::fill(roi.begin(), roi.end(), 0);

        // Copy img region to roi
        for (unsigned int h = h_start, roi_y = 0; h < h_end; ++h, ++roi_y) {
            unsigned int img_offset = h * imgW;
            unsigned int roi_offset = roi_y * roi_size;
            for (unsigned int w = w_start, roi_x = 0; w < w_end; ++w, ++roi_x) {
                roi[roi_offset + roi_x] = img[img_offset + w];
            }
        }
    }
}
