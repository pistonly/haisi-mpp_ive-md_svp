#ifndef IVE_utils_HPP
#define IVE_utils_HPP
#include "ot_common_ive.h"
#include "ot_common_video.h"
#include "ot_type.h"
#include <arpa/inet.h>
#include <cstring>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <cstdio>
#include <sstream>
#include <string>

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

void send_dection_results(int sock, const std::vector<std::vector<float>> &decs,
                          uint8_t cameraId, uint64_t timestamp);

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
/**
 * @brief 将YUV420格式的图像帧分割为四个宽高均为一半的小图像。
 *
 * @param outputImageDatas 用于存储四个输出图像数据的指针数组。
 * 每个指针应指向已分配足够内存的缓冲区，用于存储分割后的小图像数据。
 * @param frame 指向输入的YUV420格式的图像帧。
 *
 * 该函数首先映射输入帧的数据，然后将其分割为四个象限，
 * 将每个象限的数据复制到提供的输出图像数据缓冲区中。
 *
 * @note 输入的帧应为YUV420格式。
 *
 * @warning 在调用此函数之前，必须为 outputImageDatas
 * 数组中的每个指针分配足够的内存。 每个缓冲区的大小应为： (width/2) *
 * (height/2) * 3 / 2 字节。
 */

void copy_split_yuv420_from_frame(unsigned char *outputImageDatas[],
                                  ot_video_frame_info *frame);

void copy_split_yuv420_from_frame(
    std::vector<std::vector<unsigned char>> &outputImageDatas,
    ot_video_frame_info *frame);

/**
 * @brief 分割YUV420sp（NV12/NV21）格式的图像为四个宽高均为一半的小图像。
 *
 * @param inputImageData 输入的YUV420sp图像数据指针。
 * @param width 输入图像的宽度。
 * @param height 输入图像的高度。
 * @param outputImageDatas 用于存储四个输出图像数据的指针数组。
 *
 * 该函数将输入图像分为四个象限，并将数据复制到提供的输出图像数据缓冲区中。
 * 每个输出图像都将以YUV420sp格式存储，尺寸为 (width/2) x (height/2)。
 *
 * @note 调用此函数前，必须为 outputImageDatas
 * 数组中的每个指针分配足够的内存。 每个缓冲区的大小应为： (width/2) *
 * (height/2) * 3 / 2 字节。
 *
 * @warning 该函数假设输入的图像数据为YUV420sp格式（NV12）。
 * 如果您的数据为NV21格式，需要在复制UV平面数据时交换U和V分量。
 */
void splitYUV420sp(const unsigned char *inputImageData, int width, int height,
                   unsigned char *outputImageDatas[4]);

/**
 * @brief 将四个 YUV420sp（NV12/NV21）格式的小图像合成为一个原始尺寸的大图像。
 *
 * @param v_yuv420sp_4
 * 存储四个输入小图像数据的向量，每个元素是一个字节向量，表示一张小图像。
 * 每个小图像应当是原始图像的四个象限之一，顺序为：
 * - 第 0 个：左上角
 * - 第 1 个：右上角
 * - 第 2 个：左下角
 * - 第 3 个：右下角
 * @param width 合成后大图像的宽度。
 * @param height 合成后大图像的高度。
 * @param yuv420sp_combined 存储合成后大图像数据的向量，函数会将数据写入其中。
 * 该向量应当预先分配足够的空间，大小至少为 `width * height * 3 / 2` 字节。
 *
 * 该函数将四个小的 YUV420sp 图像按照象限合并到一个大图像中。
 * Y（亮度）和平面和 UV（色度）平面分别处理，使用 `memcpy` 进行高效的数据复制。
 *
 * @note 输入的小图像应当是由分割函数得到的，尺寸为 `(width/2) x (height/2)`。
 *
 * @warning `yuv420sp_combined` 向量必须预先分配，并且大小至少为 `width * height
 * * 3 / 2` 字节。
 */
void combine_YUV420sp(
    const std::vector<std::vector<unsigned char>> &v_yuv420sp_4, int width,
    int height, std::vector<unsigned char> &yuv420sp_combined);

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

void save_detect_results(const std::vector<std::vector<float>> &decs,
                         const uint8_t cameraId, const uint64_t timestamp,
                         const std::string &out_dir,
                         const std::string &filename);

void save_detect_results_csv(const std::vector<std::vector<float>> &decs,
                             const std::string &out_dir,
                             const std::string &filename);
// 在编译时定义 ENABLE_TIMER 即可启用 Timer 功能
// 可以在编译时通过 -DENABLE_TIMER 来打开

#ifdef ENABLE_TIMER
class Timer {
public:
  Timer(const std::string &name)
      : name_(name), start_(std::chrono::high_resolution_clock::now()) {}
  ~Timer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start_)
            .count();
    logger.log(DEBUG, name_, " took ", duration, " ms");
  }

private:
  std::string name_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};
#else
// 当没有启用 Timer 时，Timer 是一个空类，不做任何操作
class Timer {
public:
  Timer(const std::string &) {}
};
#endif

std::string getIPAddressUsingIfconfig(); 

uint8_t getCameraId(); 

#endif
