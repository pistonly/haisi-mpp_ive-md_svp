#include "utils.hpp"
#include "ot_common_ive.h"
#include "ot_common_video.h"
#include "ot_type.h"
#include "sample_comm.h"
#include "ss_mpi_sys.h"
#include "ss_mpi_vpss.h"
#include <arpa/inet.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <netinet/in.h>
#include <sstream>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <cassert>

// 全局日志器实例，初始日志级别为 INFO
Logger logger(INFO);

// 将std::vector<std::vector<float>>序列化为字节流
std::vector<char> serialize(const std::vector<std::vector<float>> &data) {
  std::vector<char> buffer;
  for (const auto &vec : data) {
    // 序列化每个子vector的大小
    size_t vec_size = vec.size();
    buffer.insert(buffer.end(), reinterpret_cast<const char *>(&vec_size),
                  reinterpret_cast<const char *>(&vec_size) + sizeof(vec_size));

    // 序列化每个float
    for (float value : vec) {
      buffer.insert(buffer.end(), reinterpret_cast<const char *>(&value),
                    reinterpret_cast<const char *>(&value) + sizeof(value));
    }
  }
  return buffer;
}

std::vector<char>
serialize_detect_data(const std::vector<std::vector<float>> &decs,
                      const uint8_t cameraId, const uint64_t timestamp) {
  std::vector<char> buffer;
  // total size
  unsigned int len = 13 + 24 * decs.size();
  buffer.insert(buffer.end(), reinterpret_cast<const char *>(&len),
                reinterpret_cast<const char *>(&len) + sizeof(len));
  // cameraId
  buffer.insert(buffer.end(), reinterpret_cast<const char *>(&cameraId),
                reinterpret_cast<const char *>(&cameraId) + sizeof(cameraId));

  // time stamp
  buffer.insert(buffer.end(), reinterpret_cast<const char *>(&timestamp),
                reinterpret_cast<const char *>(&timestamp) + sizeof(timestamp));

  // bboxes
  for (auto &dec : decs) {
    assert(dec.size() == 6);
    for (float value : dec) {
      buffer.insert(buffer.end(), reinterpret_cast<const char *>(&value),
                    reinterpret_cast<const char *>(&value) + sizeof(value));
    }
  }

  assert(sizeof(len) + sizeof(cameraId) + sizeof(timestamp) == 13);
  return buffer;
}

// 发送文件名和数据
void send_file_and_data(int sock, const std::string &filename,
                        const std::vector<std::vector<float>> &data) {
  // 发送文件名长度
  uint32_t filename_len = filename.size();
  send(sock, &filename_len, sizeof(filename_len), 0);

  // 发送文件名
  send(sock, filename.c_str(), filename_len, 0);

  // 序列化并发送数据
  std::vector<char> serialized_data = serialize(data);
  uint32_t data_len = serialized_data.size();

  // 发送数据长度
  send(sock, &data_len, sizeof(data_len), 0);

  // 发送数据
  send(sock, serialized_data.data(), serialized_data.size(), 0);
}

void send_dection_results(int sock, const std::vector<std::vector<float>> &decs,
                          uint8_t cameraId, uint64_t timestamp) {
  // serialize data
  std::vector<char> serialized_data =
      serialize_detect_data(decs, cameraId, timestamp);
  send(sock, serialized_data.data(), serialized_data.size(), 0);
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

void copy_yuv420_from_frame(char *yuv420, ot_video_frame_info *frame) {
  td_u32 height = frame->video_frame.height;
  td_u32 width = frame->video_frame.width;
  td_u32 size = height * width * 3 / 2; // 对于YUV420格式，大小为宽*高*1.5

  td_void *frame_data =
      ss_mpi_sys_mmap_cached(frame->video_frame.phys_addr[0], size);
  if (frame_data == NULL) {
    sample_print("mmap failed!\n");
    /* free(tmp); */
    return;
  }

  memcpy(yuv420, frame_data, size);
  // 解除内存映射
  ss_mpi_sys_munmap(frame_data, size);
}

void copy_split_yuv420_from_frame(unsigned char *outputImageDatas[4],
                                  ot_video_frame_info *frame) {
  td_u32 height = frame->video_frame.height;
  td_u32 width = frame->video_frame.width;
  td_u32 size = height * width * 3 / 2; // 对于YUV420格式，大小为宽*高*1.5

  td_void *frame_data =
      ss_mpi_sys_mmap_cached(frame->video_frame.phys_addr[0], size);
  if (frame_data == NULL) {
    sample_print("mmap failed!\n");
    /* free(tmp); */
    return;
  }

  // Calculate sizes
  int ySize = width * height;
  int uvSize = ySize / 2; // Since UV are interleaved in YUV420sp

  int yHalfWidth = width / 2;
  int yHalfHeight = height / 2;
  int yHalfSize = yHalfWidth * yHalfHeight;

  int uvHalfWidth = yHalfWidth;
  int uvHalfHeight =
      yHalfHeight / 2; // UV plane height is half of Y plane height
  int uvHalfSize =
      uvHalfWidth * uvHalfHeight * 2; // *2 because UV are interleaved

  // Pointers to the Y and UV planes in the input image
  const unsigned char *yPlane =
      reinterpret_cast<const unsigned char *>(frame_data);
  const unsigned char *uvPlane = yPlane + ySize;

  // Process each quadrant
  for (int q = 0; q < 4; ++q) {
    // Calculate starting positions
    int yStartX = (q % 2) * yHalfWidth;
    int yStartY = (q / 2) * yHalfHeight;
    int uvStartX = yStartX;
    int uvStartY = yStartY / 2; // Because UV height is half of Y height

    // Pointers to the output Y and UV data
    unsigned char *yOutput = outputImageDatas[q];
    unsigned char *uvOutput = outputImageDatas[q] + yHalfSize;

    // Copy Y plane data
    for (int i = 0; i < yHalfHeight; ++i) {
      std::memcpy(yOutput + i * yHalfWidth,
                  yPlane + (yStartY + i) * width + yStartX, yHalfWidth);
    }

    // Copy UV plane data
    for (int i = 0; i < uvHalfHeight; ++i) {
      std::memcpy(uvOutput + i * uvHalfWidth,
                  uvPlane + (uvStartY + i) * width + uvStartX, uvHalfWidth);
    }
  }
  // 解除内存映射
  ss_mpi_sys_munmap(frame_data, size);
}

void copy_split_yuv420_from_frame(
    std::vector<std::vector<unsigned char>> &outputImageDatas,
    ot_video_frame_info *frame) {
  td_u32 height = frame->video_frame.height;
  td_u32 width = frame->video_frame.width;
  td_u32 size = height * width * 3 / 2; // 对于YUV420格式，大小为宽*高*1.5

  td_void *frame_data =
      ss_mpi_sys_mmap_cached(frame->video_frame.phys_addr[0], size);
  if (frame_data == NULL) {
    sample_print("mmap failed!\n");
    /* free(tmp); */
    return;
  }

  // Calculate sizes
  int ySize = width * height;
  int uvSize = ySize / 2; // Since UV are interleaved in YUV420sp

  int yHalfWidth = width / 2;
  int yHalfHeight = height / 2;
  int yHalfSize = yHalfWidth * yHalfHeight;

  int uvHalfWidth = yHalfWidth;
  int uvHalfHeight =
      yHalfHeight / 2; // UV plane height is half of Y plane height
  int uvHalfSize =
      uvHalfWidth * uvHalfHeight * 2; // *2 because UV are interleaved

  // Pointers to the Y and UV planes in the input image
  const unsigned char *yPlane =
      reinterpret_cast<const unsigned char *>(frame_data);
  const unsigned char *uvPlane = yPlane + ySize;

  // Process each quadrant
  for (int q = 0; q < 4; ++q) {
    // Calculate starting positions
    int yStartX = (q % 2) * yHalfWidth;
    int yStartY = (q / 2) * yHalfHeight;
    int uvStartX = yStartX;
    int uvStartY = yStartY / 2; // Because UV height is half of Y height

    // Pointers to the output Y and UV data
    unsigned char *yOutput = outputImageDatas[q].data();
    unsigned char *uvOutput = outputImageDatas[q].data() + yHalfSize;

    // Copy Y plane data
    for (int i = 0; i < yHalfHeight; ++i) {
      std::memcpy(yOutput + i * yHalfWidth,
                  yPlane + (yStartY + i) * width + yStartX, yHalfWidth);
    }

    // Copy UV plane data
    for (int i = 0; i < uvHalfHeight; ++i) {
      std::memcpy(uvOutput + i * uvHalfWidth,
                  uvPlane + (uvStartY + i) * width + uvStartX, uvHalfWidth);
    }
  }
  // 解除内存映射
  ss_mpi_sys_munmap(frame_data, size);
}

void merge_rois(const unsigned char *img, ot_ive_ccblob *p_blob,
                std::vector<unsigned char> &merged_rois,
                std::vector<std::pair<int, int>> &top_lefts,
                std::vector<std::vector<float>> &blob_xyxy, float scale_x,
                float scale_y, int imgH, int imgW, int merged_roi_H,
                int merged_roi_W, int max_roi_num) {
  top_lefts.clear();
  blob_xyxy.clear();
  // img and merged_rois are YUV format images.
  const int roi_size = 32;
  const int roi_size_half = roi_size / 2;
  const int rgn_num =
      std::min(static_cast<int>(p_blob->info.bits.rgn_num), max_roi_num);
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
    float rgn_x0 = scale_x * static_cast<float>(rgn.left);
    float rgn_y0 = scale_y * static_cast<float>(rgn.top);
    float rgn_x1 = scale_x * static_cast<float>(rgn.right);
    float rgn_y1 = scale_y * static_cast<float>(rgn.bottom);
    const int center_x = static_cast<int>(0.5f * (rgn_x0 + rgn_x1));
    const int center_y = static_cast<int>(0.5f * (rgn_y0 + rgn_y1));

    const int w_start = std::max(center_x - roi_size_half, 0);
    const int w_end = std::min(center_x + roi_size_half, imgW);
    const int h_start = std::max(center_y - roi_size_half, 0);
    const int h_end = std::min(center_y + roi_size_half, imgH);

    top_lefts.push_back(std::make_pair(w_start, h_start));
    blob_xyxy.push_back({rgn_x0, rgn_y0, rgn_x1, rgn_y1});

    if (w_start < 0) {
      std::cout << "----------------w_start: " << w_start << std::endl;
    }
    // 预计算索引
    unsigned int roi_offset = offset_y * merged_roi_W + offset_x;
    unsigned int img_offset = h_start * imgW;
    unsigned int roi_uv_offset =
        (merged_roi_H + offset_y * 0.5) * merged_roi_W + offset_x;
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

void save_detect_results(const std::vector<std::vector<float>> &decs,
                         const std::string &out_dir, const int imageId,
                         const std::string &prefix) {
  std::ostringstream oss;
  std::string stem = prefix + "decs_image_";
  oss << out_dir << stem << std::setw(6) << std::setfill('0') << imageId
      << ".bin";
  std::ofstream outFile(oss.str(), std::ios::binary);
  if (!outFile) {
    std::cerr << "Error opening file " << oss.str() << " for writing."
              << std::endl;
    return;
  }

  std::vector<char> serialized_data = serialize(decs);
  outFile.write(reinterpret_cast<const char *>(serialized_data.data()),
                serialized_data.size());
  outFile.close();
  return;
}

void save_detect_results(const std::vector<std::vector<float>> &bbox,
                         const std::vector<float> &conf,
                         const std::vector<int> &cls,
                         const std::string &out_dir, const int imageId,
                         const std::string &prefix) {
  // concat to decs: x0, y0, x1, y1, conf, cls
  std::vector<std::vector<float>> decs{bbox.size(), std::vector<float>(6, 0.f)};
  for (auto i = 0; i < bbox.size(); ++i) {
    decs[i][0] = bbox[i][0];
    decs[i][1] = bbox[i][1];
    decs[i][2] = bbox[i][2];
    decs[i][3] = bbox[i][3];
    decs[i][4] = conf[i];
    decs[i][5] = static_cast<float>(cls[i]);
  }
  save_detect_results(decs, out_dir, imageId, prefix);
}

void save_detect_results(const std::vector<std::vector<float>> &decs,
                         const std::string &out_dir,
                         const std::string &filename) {
  std::ofstream outFile(out_dir + filename, std::ios::binary);
  if (!outFile) {
    std::cerr << "Error opening file " << filename << " for writing."
              << std::endl;
    return;
  }

  std::vector<char> serialized_data = serialize(decs);
  outFile.write(reinterpret_cast<const char *>(serialized_data.data()),
                serialized_data.size());
  outFile.close();
  return;
}

void save_detect_results(const std::vector<std::vector<float>> &decs,
                         const uint8_t cameraId, const uint64_t timestamp,
                         const std::string &out_dir,
                         const std::string &filename) {
  std::ofstream outFile(out_dir + filename, std::ios::binary);
  if (!outFile) {
    std::cerr << "Error opening file " << filename << " for writing."
              << std::endl;
    return;
  }

  std::vector<char> serialized_data = serialize_detect_data(decs, cameraId, timestamp);
  outFile.write(reinterpret_cast<const char *>(serialized_data.data()),
                serialized_data.size());
  outFile.close();
  return;
}

void save_detect_results_csv(const std::vector<std::vector<float>> &decs,
                         const std::string &out_dir,
                         const std::string &filename) {
  std::ofstream outFile(out_dir + filename, std::ios::binary);
  if (!outFile) {
    std::cerr << "Error opening file " << filename << " for writing."
              << std::endl;
    return;
  }

  for (auto &dec:decs){
    int i = 0;
    for (; i< dec.size() - 1; ++i){
      outFile << dec[i] << ", ";
    }
    outFile << dec[i] << std::endl;
  }
  outFile.close();
  return;
}

void splitYUV420sp(const unsigned char *inputImageData, int width, int height,
                   unsigned char *outputImageDatas[4]) {
  // Calculate sizes
  int ySize = width * height;
  int uvSize = ySize / 2; // Since UV are interleaved in YUV420sp

  int yHalfWidth = width / 2;
  int yHalfHeight = height / 2;
  int yHalfSize = yHalfWidth * yHalfHeight;

  int uvHalfWidth = yHalfWidth;
  int uvHalfHeight =
      yHalfHeight / 2; // UV plane height is half of Y plane height
  int uvHalfSize =
      uvHalfWidth * uvHalfHeight * 2; // *2 because UV are interleaved

  // Pointers to the Y and UV planes in the input image
  const unsigned char *yPlane = inputImageData;
  const unsigned char *uvPlane = inputImageData + ySize;

  // Process each quadrant
  for (int q = 0; q < 4; ++q) {
    // Calculate starting positions
    int yStartX = (q % 2) * yHalfWidth;
    int yStartY = (q / 2) * yHalfHeight;
    int uvStartX = yStartX;
    int uvStartY = yStartY / 2; // Because UV height is half of Y height

    // Pointers to the output Y and UV data
    unsigned char *yOutput = outputImageDatas[q];
    unsigned char *uvOutput = outputImageDatas[q] + yHalfSize;

    // Copy Y plane data
    for (int i = 0; i < yHalfHeight; ++i) {
      std::memcpy(yOutput + i * yHalfWidth,
                  yPlane + (yStartY + i) * width + yStartX, yHalfWidth);
    }

    // Copy UV plane data
    for (int i = 0; i < uvHalfHeight; ++i) {
      std::memcpy(uvOutput + i * uvHalfWidth,
                  uvPlane + (uvStartY + i) * width + uvStartX, uvHalfWidth);
    }
  }
}

void combine_YUV420sp(
    const std::vector<std::vector<unsigned char>> &v_yuv420sp_4, int width,
    int height, std::vector<unsigned char> &yuv420sp_combined) {
  // Calculate sizes
  int ySize = width * height;
  int uvSize = ySize / 2; // Since UV are interleaved in YUV420sp

  int yHalfWidth = width / 2;
  int yHalfHeight = height / 2;
  int yHalfSize = yHalfWidth * yHalfHeight;

  int uvHalfWidth = yHalfWidth;
  int uvHalfHeight =
      yHalfHeight / 2; // UV plane height is half of Y plane height
  int uvHalfSize =
      uvHalfWidth * uvHalfHeight * 2; // *2 because UV are interleaved

  // Pointers to the Y and UV planes in the target image
  unsigned char *yPlane = yuv420sp_combined.data();
  unsigned char *uvPlane = yPlane + ySize;

  // Process each quadrant
  for (int q = 0; q < 4; ++q) {
    // Calculate starting positions
    int yStartX = (q % 2) * yHalfWidth;
    int yStartY = (q / 2) * yHalfHeight;
    int uvStartX = yStartX;
    int uvStartY = yStartY / 2; // Because UV height is half of Y height

    // Pointers to the output Y and UV data
    const unsigned char *ySrc = v_yuv420sp_4[q].data();
    const unsigned char *uvSrc = v_yuv420sp_4[q].data() + yHalfSize;

    // Copy Y plane data
    for (int i = 0; i < yHalfHeight; ++i) {
      std::memcpy(yPlane + (yStartY + i) * width + yStartX,
                  ySrc + i * yHalfWidth, yHalfWidth);
    }

    // Copy UV plane data
    for (int i = 0; i < uvHalfHeight; ++i) {
      std::memcpy(uvPlane + (uvStartY + i) * width + uvStartX,
                  uvSrc + i * uvHalfWidth, uvHalfWidth);
    }
  }
}

std::string getIPAddressUsingIfconfig() {
  FILE *pipe = popen("ifconfig", "r");
  if (!pipe) {
    std::cerr << "popen failed" << std::endl;
    return "";
  }

  std::stringstream buffer;
  char ch;
  while (fread(&ch, 1, 1, pipe) > 0) {
    buffer.put(ch);
  }

  std::string output = buffer.str();
  pclose(pipe);

  std::size_t inetPos = output.find("inet ");
  if (inetPos == std::string::npos) {
    return "";
  }

  std::size_t addrStart = output.find_first_not_of(" \t", inetPos + 5);
  std::size_t addrEnd = output.find_first_of(" \t\n", addrStart);
  return output.substr(addrStart, addrEnd - addrStart);
}

uint8_t getCameraId() {
  std::string ipAddress = getIPAddressUsingIfconfig();
  std::size_t lastDotPos = ipAddress.find_last_of('.');
  if (lastDotPos == std::string::npos) {
    return -1;
  }
  std::string lastOctetStr = ipAddress.substr(lastDotPos + 1);
  return static_cast<uint8_t>(std::stoi(lastOctetStr));
}
