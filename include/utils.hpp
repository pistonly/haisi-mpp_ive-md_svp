#ifndef IVE_utils_HPP
#define IVE_utils_HPP
#include "ot_common_ive.h"
#include "ot_common_video.h"
#include "ot_type.h"
#include <arpa/inet.h>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <mutex>

// 定义日志级别
enum LogLevel { DEBUG, INFO, WARNING, ERROR };

// 简单的日志类
class Logger {
public:
  Logger(LogLevel level) : log_level(level) {}

  void setLogLevel(LogLevel level) { log_level = level; }

  template <typename... Args> void log(LogLevel level, Args &&...args) {
    if (level >= log_level) {
      print(level, std::forward<Args>(args)...);
    }
  }

private:
  LogLevel log_level;

  template <typename... Args> void print(LogLevel level, Args &&...args) {
    std::string level_str;
    switch (level) {
    case DEBUG:
      level_str = "[DEBUG]";
      break;
    case INFO:
      level_str = "[INFO]";
      break;
    case WARNING:
      level_str = "[WARNING]";
      break;
    case ERROR:
      level_str = "[ERROR]";
      break;
    }
    std::lock_guard<std::mutex> guard(mtx);
    std::cout << level_str << " ";
    (std::cout << ... << args) << std::endl;
  }

  std::mutex mtx; // 保证多线程环境下日志输出的原子性
};

extern Logger logger;

/**
 * @brief serialize data for sending to TCP server
 * @details Description
 * @param[in] data Description
 * @return Description
 */
std::vector<char> serialize(const std::vector<std::vector<float>> &data);

/**
 * @brief send filename and data to TCP server
 * @details Description
 * @param[in] sock Description
 * @param[in] filename Description
 * @param[in] data Description
 */
void send_file_and_data(int sock, const std::string &filename,
                        const std::vector<std::vector<float>> &data);

/**
 * @brief merge rois to big Canvas.
 * @details Description
 * @param[in] img: YUV format image
 * @param[in] p_blob Pointer to blob
 * @param[out] merged_rois: YUV format image
 * @param[in] top_lefts The lefts of top
 * @param[in] scale_x Description
 * @param[in] scale_y Description
 * @param[in] imgH Description
 * @param[in] imgW Description
 * @param[in] merged_roi_H Description
 * @param[in] merged_roi_W Description
 */
void merge_rois(const unsigned char *img, ot_ive_ccblob *p_blob,
                std::vector<unsigned char> &merged_rois,
                std::vector<std::pair<int, int>> &top_lefts, float scale_x,
                float scale_y, int imgH, int imgW, int merged_roi_H,
                int merged_roi_W);

/**
 * @brief Debug tool. save data to binary files.
 * @details Description
 * @param[in] merged_roi The roi of merged
 * @param[in] out_dir The dir of out
 * @param[in] imgId Description
 * @return Description
 */
int save_merged_rois(const std::vector<unsigned char> &merged_roi,
                     const std::string &out_dir, const int imgId);

/**
 * @brief copy yuv420 from frame
 * @details Description
 * @param[out] yuv420 Description
 * @param[in] frame Description
 */
void copy_yuv420_from_frame(char *yuv420, ot_video_frame_info *frame);

void save_detect_results(const std::vector<std::vector<float>> &decs,
                         const std::string &out_dir, const int imageId,
                         const std::string &prefix = "");

void save_detect_results(const std::vector<std::vector<float>> &bbox,
                         const std::vector<float> &conf,
                         const std::vector<int> &cls,
                         const std::string &out_dir, const int imageId,
                         const std::string &prefix = "");

void save_detect_results(const std::vector<std::vector<float>> &decs,
                         const std::string &out_dir, 
                         const std::string &filename);

#endif
