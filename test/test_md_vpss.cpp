#include "ive_md.hpp"
#include "ot_common_ive.h"
#include "ot_defines.h"
#include "ot_type.h"
#include "sample_comm.h"
#include "ss_mpi_vpss.h"
#include "utils.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>


ot_video_frame_info frame;

int main(int argc, char *argv[]) {
  td_s32 vpss_grp = 0;
  td_s32 vpss_chn = 1;
  if (argc > 2) {
    vpss_grp = std::stoi(argv[1]);
    vpss_chn = std::stoi(argv[2]);
  }
  bool sys_init = false;
  IVE_MD md(sys_init);

  ot_ive_ccblob blob = {0};

  int img_id = 0;
  while (img_id++ < 10) {
    td_s32 ret = ss_mpi_vpss_get_chn_frame(vpss_grp, vpss_chn, &frame, 100);
    if (ret != TD_SUCCESS) {
      sample_print("get chn frame-0 failed for Err(%#x)\n", ret);
      return 1;
    }
    md.process(frame, &blob);
    std::cout << "instance number: " << static_cast<int>(blob.info.bits.rgn_num) << std::endl;

    ret = ss_mpi_vpss_release_chn_frame(vpss_grp, vpss_chn, &frame);
  }

  return 0;
}
