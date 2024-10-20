#ifndef IVE_MD_HPP
#define IVE_MD_HPP

#include "ot_common_ive.h"
#include "ot_common_video.h"

#define OT_SAMPLE_MD_SRC_NUM 2
#define OT_SAMPLE_MD_WIDTH 1920
#define OT_SAMPLE_MD_HEIGHT 1080

typedef struct {
  ot_svp_src_img img[OT_SAMPLE_MD_SRC_NUM];
  ot_svp_dst_img diff;
  ot_svp_src_img sad_zeros;
  ot_svp_dst_img sad;
  ot_svp_dst_img sad_thres;
  ot_svp_dst_mem_info blob;
} ot_sample_MDStep_info;

class IVE_MD {
public:
  IVE_MD(bool sys_init=true);
  ~IVE_MD();
  IVE_MD(const IVE_MD&) = delete;
  IVE_MD &operator=(const IVE_MD &) = delete;
  IVE_MD(IVE_MD &&) = default;
  IVE_MD& operator=(IVE_MD&&) = default;
  int process(const unsigned char* p_image, ot_ive_ccblob *blob);
  int process(ot_video_frame_info &frame, ot_ive_ccblob *blob);
  int process(ot_video_frame_info *frame, ot_ive_ccblob *blob);

private:
  ot_sample_MDStep_info m_mdstep;
  td_s32 m_ret;
  ot_ive_sub_ctrl m_sub_ctrl;
  ot_ive_handle m_handle;
  ot_ive_threshold_ctrl m_thre_ctrl;
  ot_ive_ccl_ctrl m_ccl_ctrl;
  ot_ive_sad_ctrl m_sad_ctrl;
  td_u8 m_current_img;
  bool m_isFirstFrame;

  int process_core(ot_ive_ccblob *blob);
};
#endif
