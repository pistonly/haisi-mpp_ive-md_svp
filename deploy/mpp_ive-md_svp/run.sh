#!/bin/sh
export LD_LIBRARY_PATH=/mnt/data/lib:/mnt/data/lib/npu:$LD_LIBRARY_PATH
cd /mnt/data/mpp_ive-md_svp/out
sleep 10
nohup ./md4k_yoloThread_merge_rois_async_vpss_nnn_1chn > /mnt/data/log.log 2>&1 &
