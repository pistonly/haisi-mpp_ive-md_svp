
#include "ot_common_ive.h"
#include "ot_common_svp.h"
#include "ot_type.h"
#include "sample_common_ive.h"
#include "sample_common_svp.h"
#include "ss_mpi_ive.h"
#include "ive_md.hpp"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


IVE_MD::IVE_MD() {

  // ive mpi init
  m_ret = sample_common_ive_check_mpi_init();
  sample_svp_check_failed_trace(m_ret != TD_TRUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                "ive_check_mpi_init failed!\n");

  // md init
  (td_void)memset_s(&m_mdstep, sizeof(ot_sample_MDStep_info), 0,
                    sizeof(ot_sample_MDStep_info));

  for (int i = 0; i < OT_SAMPLE_MD_SRC_NUM; ++i) {
    m_ret =
      sample_common_ive_create_image(&(m_mdstep.img[i]), OT_SVP_IMG_TYPE_U8C1,
                                     OT_SAMPLE_MD_WIDTH, OT_SAMPLE_MD_HEIGHT);
    sample_svp_check_failed_trace(m_ret != TD_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                  "sample_common_ive_create_image failed!\n");
  }

  m_ret =
      sample_common_ive_create_image(&m_mdstep.diff, OT_SVP_IMG_TYPE_U8C1,
                                     OT_SAMPLE_MD_WIDTH, OT_SAMPLE_MD_HEIGHT);
  sample_svp_check_failed_trace(m_ret != TD_SUCCESS,
                                SAMPLE_SVP_ERR_LEVEL_ERROR,
                                "sample_common_ive_create_image failed!\n");
  m_ret =
      sample_common_ive_create_image(&m_mdstep.sad_zeros, OT_SVP_IMG_TYPE_U8C1,
                                     OT_SAMPLE_MD_WIDTH, OT_SAMPLE_MD_HEIGHT);
  sample_svp_check_failed_trace(m_ret != TD_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                "sample_common_ive_create_image failed!\n");

  m_ret = sample_common_ive_create_image(&m_mdstep.sad, OT_SVP_IMG_TYPE_U16C1,
                                         OT_SAMPLE_MD_WIDTH / 4,
                                         OT_SAMPLE_MD_HEIGHT / 4);

  sample_svp_check_failed_trace(m_ret != TD_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                "sample_common_ive_create_image failed!\n");

  m_ret = sample_common_ive_create_image(
      &m_mdstep.sad_thres, OT_SVP_IMG_TYPE_U8C1, OT_SAMPLE_MD_WIDTH / 4,
      OT_SAMPLE_MD_HEIGHT / 4);
  sample_svp_check_failed_trace(m_ret != TD_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                "sample_common_ive_create_image failed!\n");

  // init blob
  m_ret =
      sample_common_ive_create_mem_info(&m_mdstep.blob, sizeof(ot_ive_ccblob));
  sample_svp_check_failed_trace(m_ret != TD_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                "sample_common_ive_create_mem_info "
                                "failed!\n");

  m_sub_ctrl.mode = OT_IVE_SUB_MODE_ABS;
  m_thre_ctrl.mode = OT_IVE_THRESHOLD_MODE_BINARY;
  m_thre_ctrl.low_threshold = 15;
  m_thre_ctrl.min_val = 0;
  m_thre_ctrl.max_val = 255;

  m_ccl_ctrl.mode = OT_IVE_CCL_MODE_4C;
  m_ccl_ctrl.init_area_threshold = 0;
  m_ccl_ctrl.step = 10;

  m_sad_ctrl.mode = OT_IVE_SAD_MODE_MB_4X4;
  m_sad_ctrl.out_ctrl = OT_IVE_SAD_OUT_CTRL_16BIT_BOTH;
  m_sad_ctrl.max_val = 255;
  m_sad_ctrl.min_val = 0;
  m_sad_ctrl.threshold = 100;

  m_ret = sample_common_ive_init_zeros_img(&m_mdstep.sad_zeros);
  m_current_img = 0;
  m_isFirstFrame = true;
}

IVE_MD::~IVE_MD(){
  for (td_u16 i=0; i< OT_SAMPLE_MD_SRC_NUM; ++i) {
    sample_svp_mmz_free(m_mdstep.img[i].phys_addr[0], m_mdstep.img[i].virt_addr[0]);
  }
  sample_svp_mmz_free(m_mdstep.diff.phys_addr[0], m_mdstep.diff.virt_addr[0]);
  sample_svp_mmz_free(m_mdstep.sad_zeros.phys_addr[0], m_mdstep.sad_zeros.virt_addr[0]);
  sample_svp_mmz_free(m_mdstep.sad.phys_addr[0], m_mdstep.sad.virt_addr[0]);
  sample_svp_mmz_free(m_mdstep.sad_thres.phys_addr[0], m_mdstep.sad_thres.virt_addr[0]);
  sample_svp_mmz_free(m_mdstep.blob.phys_addr, m_mdstep.blob.virt_addr);
}

int IVE_MD::process_core(ot_ive_ccblob *blob) {
  if (m_isFirstFrame) {
    m_isFirstFrame = false;
    blob->info.bits.rgn_num = 0;
    return m_ret;
  }

  m_ret = ss_mpi_ive_sub(&m_handle, &m_mdstep.img[m_current_img],
                         &m_mdstep.img[1 - m_current_img], &m_mdstep.diff,
                         &m_sub_ctrl, TD_TRUE);
  sample_svp_check_exps_trace(m_ret != TD_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
                              "ss_mpi_ive_sub failed!\n");

  m_ret = ss_mpi_ive_threshold(&m_handle, &m_mdstep.diff, &m_mdstep.diff,
                               &m_thre_ctrl, TD_TRUE);
  sample_svp_check_exps_trace(m_ret != TD_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
                              "ss_mpi_ive_threshold failed!\n");

  m_ret =
      ss_mpi_ive_sad(&m_handle, &m_mdstep.diff, &m_mdstep.sad_zeros,
                     &m_mdstep.sad, &m_mdstep.sad_thres, &m_sad_ctrl, TD_TRUE);
  sample_svp_check_exps_trace(m_ret != TD_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
                              "ss_mpi_ive_sad failed!\n");

  m_ret = ss_mpi_ive_ccl(&m_handle, &m_mdstep.sad_thres, &m_mdstep.blob,
                         &m_ccl_ctrl, TD_TRUE);
  sample_svp_check_exps_trace(m_ret != TD_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
                              "ss_mpi_ive_ccl failed!\n");

  m_current_img = 1 - m_current_img;

  // output blob
  ot_ive_ccblob *blob_tmp = sample_svp_convert_addr_to_ptr(ot_ive_ccblob, m_mdstep.blob.virt_addr);

  blob->info.bits.rgn_num = blob_tmp->info.bits.rgn_num;
  blob->info.bits.cur_area_threshold = blob_tmp->info.bits.cur_area_threshold;
  blob->info.bits.label_status = blob_tmp->info.bits.label_status;
  for(td_u16 i=0; i< blob_tmp->info.bits.rgn_num; ++i) {
    blob->rgn[i].area = blob_tmp->rgn[i].area;
    blob->rgn[i].left = blob_tmp->rgn[i].left;
    blob->rgn[i].right = blob_tmp->rgn[i].right;
    blob->rgn[i].top = blob_tmp->rgn[i].top;
    blob->rgn[i].bottom = blob_tmp->rgn[i].bottom;
  }
  return m_ret;
}

int IVE_MD::process(const unsigned char *p_image, ot_ive_ccblob *blob) {
  m_ret = sample_common_ive_set_img(&m_mdstep.img[m_current_img], p_image);
  sample_svp_check_exps_trace(m_ret != TD_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
                              "sample_common_ive_set_img failed!\n");
  return process_core(blob);
}

int IVE_MD::process(ot_video_frame_info &frame, ot_ive_ccblob *blob) {
  td_bool is_instant = TD_TRUE;
  m_ret = sample_common_ive_dma_image(
      &frame, &m_mdstep.img[m_current_img], is_instant);
  return process_core(blob);
}

int IVE_MD::process(ot_video_frame_info *frame, ot_ive_ccblob *blob) {
  td_bool is_instant = TD_TRUE;
  m_ret = sample_common_ive_dma_image(
      frame, &m_mdstep.img[m_current_img], is_instant);
  return process_core(blob);
}
