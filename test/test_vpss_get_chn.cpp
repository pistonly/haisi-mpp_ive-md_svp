#include "ot_defines.h"
#include "ot_type.h"
#include "sample_comm.h"
#include "ss_mpi_vpss.h"
#include "utils.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

ot_video_frame_info frame_H;

int main(int argc, char *argv[]) {
  td_s32 vpss_grp = 0;
  td_s32 vpss_chn = 1;
  if (argc > 2){
    vpss_grp = std::stoi(argv[1]);
    vpss_chn = std::stoi(argv[2]);
  }
  td_s32 ret = ss_mpi_vpss_get_chn_frame(vpss_grp, vpss_chn, &frame_H, 100);
  if (ret != TD_SUCCESS) {
    sample_print("get chn frame-0 failed for Err(%#x)\n", ret);
    return 1;
  }

  //  copy to frame
  td_u32 height = frame_H.video_frame.height;
  td_u32 width = frame_H.video_frame.width;
  td_u32 size = height * width * 3 / 2; // 对于YUV420格式，大小为宽*高*1.5

  td_void *yuv = ss_mpi_sys_mmap_cached(frame_H.video_frame.phys_addr[0], size);
  if (yuv == NULL) {
    sample_print("mmap failed!\n");
    return false;
  }

  std::vector<unsigned char> img(size);
  copy_yuv420_from_frame(reinterpret_cast<char *>(img.data()), &frame_H);

  ss_mpi_vpss_release_chn_frame(vpss_grp, vpss_chn, &frame_H);

  // save image
  std::stringstream ss;
  ss << "image_from_vpss_grp-" << vpss_grp << "_chn-" << vpss_chn << "_"
     << width << "x" << height << ".bin";
  std::ofstream outFile(ss.str(), std::ios::binary);
  if (!outFile) {
    sample_print("Error opening file for save");
    return 1;
  }

  outFile.write(reinterpret_cast<const char *>(img.data()), img.size());
  outFile.close();
  return 0;
}
