cmake_minimum_required(VERSION 3.5.1)

project(MD_Infer)
# set(CMAKE_CXX_COMPILER /opt/linux/x86-arm/aarch64-mix210-linux/bin/aarch64-mix210-linux-gcc)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
add_definitions(-DSENSOR0_TYPE=OV_OS08A20_MIPI_8M_30FPS_12BIT)

# 添加一个选项用于控制 ENABLE_TIMER 是否被定义
option(ENABLE_TIMER "Enable the Timer class" ON)

# 根据选项定义宏 ENABLE_TIMER
if(ENABLE_TIMER)
  add_definitions(-DENABLE_TIMER)
endif()

# Specify target generation path
set(CMAKE_C_COMPILER "aarch64-mix210-linux-gcc")
set(CMAKE_CXX_COMPILER "aarch64-mix210-linux-g++")


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -s")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie")
set(CMAKE_SKIP_RPATH TRUE)

# Compile options
add_compile_options(
        -fPIC
        -fstack-protector-all
        -D_FORTIFY_SOURCE=2 -O2
        -lstdc++
)

set(SDK_PATH /home/liuyang/Documents/haisi/ss928_sdk_021)
set(MPP_PATH ${SDK_PATH}/smp/a55_linux/mpp)
set(SAMPLE_PATH ${MPP_PATH}/sample)
set(YOLO_PATH /home/liuyang/Documents/haisi/ai-sd3403/ai-sd3403/mpp_inference)
set(Ascend_DIR /usr/local/Ascend)
set(DDK_PATH ${Ascend_DIR}/ascend-toolkit/svp_latest)
set(DDK_NNN_PATH ${Ascend_DIR}/ascend-toolkit/5.20)

# FFmpeg路径
set(FFMPEG_PATH /home/liuyang/Documents/haisi/ffmpeg/install)

include_directories(
  ./include/
  ./include/svp/common/
  ${MPP_PATH}/out/include/
  ${MPP_PATH}/out/include/svp_npu/
  ${SAMPLE_PATH}/common/
  ${SAMPLE_PATH}/audio/adp/
  ${YOLO_PATH}/include/
  ${DDK_PATH}/acllib/include/
  ${DDK_NNN_PATH}/acllib/include/
  ${FFMPEG_PATH}/include
)

add_library(ive_md STATIC src/ive_md.cpp)
add_library(hardware_decode STATIC src/ffmpeg_vdec_vpss.cpp)

set(LIB_PATH ${Ascend_DIR}/ascend-toolkit/svp_latest/acllib_linux.x86_64/acllib/lib64/stub/)
set(LIB_NNN_PATH ${DDK_NNN_PATH}/acllib/lib64/stub/)
link_directories(${LIB_PATH} ${LIB_NNN_PATH} ${FFMPEG_PATH}/lib)


set(SYSTEM_LIBS pthread m dl stdc++)
set(SECURITY_LIBS ${MPP_PATH}/out/lib/libsecurec.a)
set(ACL_LIBS ${MPP_PATH}/out/lib/svp_npu/libsvp_acl.a ${MPP_PATH}/out/lib/svp_npu/libprotobuf-c.a)
set(YOLO_LIB ${YOLO_PATH}/lib/libMPP_YOLOV8.a)

FILE(GLOB SAMPLE_COMMON_SRC ${SAMPLE_PATH}/common/*.c)
FILE(GLOB SVP_COMMON_SRC ./src/svp/common/*.c)
FILE(GLOB UTILS_SRC ./src/utils.cpp)
add_executable(md_yolo_rois ./examples/md_yolo_rois.cpp ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC})

add_executable(md_yolo_merge_roi ./examples/md_yolo_merge_rois.cpp ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC})

add_executable(md_yolo_merge_rois_async ./examples/md_yolo_merge_rois_async.cpp ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC})

add_executable(md_yolo_merge_rois_async_nnn ./examples/md_yolo_merge_rois_async_nnn.cpp ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC})

add_executable(md_yolo_merge_rois_async_vdec_vpss ./examples/md_yolo_merge_rois_async_vdec_vpss.cpp ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC}
  ${UTILS_SRC})

FILE(GLOB YOLOV8_NNN_SRC ./src/yolov8_nnn.cpp)
add_executable(md_yolo_merge_rois_async_vdec_vpss_nnn ./examples/md_yolo_merge_rois_async_vdec_vpss_nnn.cpp
  ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC} ${UTILS_SRC} ${YOLOV8_NNN_SRC})

add_executable(md_yolo_merge_rois_async_vpss_nnn_2chns ./examples/md_yolo_merge_rois_async_vpss_nnn_2chns.cpp
  ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC} ${UTILS_SRC} ${YOLOV8_NNN_SRC})

add_executable(md4k_yolo_merge_rois_async_vdec_vpss_nnn ./examples/md4k_yolo_merge_rois_async_vdec_vpss_nnn.cpp
  ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC} ${UTILS_SRC} ${YOLOV8_NNN_SRC})

add_executable(md4k_yoloThread_merge_rois_async_vdec_vpss_nnn ./examples/md4k_yoloThread_merge_rois_async_vdec_vpss_nnn.cpp
  ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC} ${UTILS_SRC} ${YOLOV8_NNN_SRC})

add_executable(md4k_yoloThread_merge_rois_async_vdec_vpss_nnn_with_sky ./examples/md4k_yoloThread_merge_rois_async_vdec_vpss_nnn_with_sky.cpp
  ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC} ${UTILS_SRC} ${YOLOV8_NNN_SRC})

add_executable(md4k_yoloThread_merge_rois_async_vpss_nnn_2chns ./examples/md4k_yoloThread_merge_rois_async_vpss_nnn_2chns.cpp
  ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC} ${UTILS_SRC} ${YOLOV8_NNN_SRC})

add_executable(md4k_yoloThread_merge_rois_async_vpss_nnn_2chns_v2 ./examples/md4k_yoloThread_merge_rois_async_vpss_nnn_2chns_v2.cpp
  ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC} ${UTILS_SRC} ${YOLOV8_NNN_SRC})

add_executable(md4k_yoloThread_merge_rois_async_vpss_nnn_1chn-1 ./examples/md4k_yoloThread_merge_rois_async_vpss_nnn_1chn-1.cpp
  ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC} ${UTILS_SRC} ${YOLOV8_NNN_SRC})

add_executable(md4k_yoloThread_merge_rois_async_vpss_nnn_1chn-1_with_sky ./examples/md4k_yoloThread_merge_rois_async_vpss_nnn_1chn-1_with_sky.cpp
  ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC} ${UTILS_SRC} ${YOLOV8_NNN_SRC})

add_executable(md4k_yoloThread_merge_rois_async_vpss_nnn_1chn-2 ./examples/md4k_yoloThread_merge_rois_async_vpss_nnn_1chn-2.cpp
  ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC} ${UTILS_SRC} ${YOLOV8_NNN_SRC})

add_executable(md4k_yoloThread_merge_rois_async_vpss_nnn_1chn-2_with_sky ./examples/md4k_yoloThread_merge_rois_async_vpss_nnn_1chn-2_with_sky.cpp
  ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC} ${UTILS_SRC} ${YOLOV8_NNN_SRC})

add_executable(md_yoloThread_merge_rois_async_vdec_vpss_nnn ./examples/md_yoloThread_merge_rois_async_vdec_vpss_nnn.cpp
  ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC} ${UTILS_SRC} ${YOLOV8_NNN_SRC})


add_executable(md_yolo_merge_rois_sync_vdec_vpss_nnn ./examples/md_yolo_merge_rois_sync_vdec_vpss_nnn.cpp
  ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC} ${UTILS_SRC} ${YOLOV8_NNN_SRC})

add_executable(test_vpss_get_chn ./test/test_vpss_get_chn.cpp  ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC} ${UTILS_SRC})
add_executable(test_md_vpss ./test/test_md_vpss.cpp  ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC} ${UTILS_SRC})


include(cmake/ive_link_libs.cmake)
target_link_libraries(md_yolo_rois
  ive_md
  svp_acl
  ascendcl
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  stdc++ # note this line is important
)

target_link_libraries(md_yolo_merge_roi
  ive_md
  svp_acl
  ascendcl
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  stdc++ # note this line is important
)

target_link_libraries(md_yolo_merge_rois_async
  ive_md
  svp_acl
  ascendcl
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  stdc++ # note this line is important
)

target_link_libraries(md_yolo_merge_rois_async_nnn
  ive_md
  svp_acl
  ascendcl
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  stdc++ # note this line is important
)

target_link_libraries(md_yolo_merge_rois_async_vdec_vpss
  ive_md
  svp_acl
  ascendcl
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  hardware_decode
  avformat
  avcodec
  avutil
  swresample
  swscale
  stdc++ # note this line is important
)

target_link_libraries(md_yolo_merge_rois_async_vdec_vpss_nnn
  ive_md
  ascendcl
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  hardware_decode
  avformat
  avcodec
  avutil
  swresample
  swscale
  stdc++ # note this line is important
)

target_link_libraries(md_yoloThread_merge_rois_async_vdec_vpss_nnn
  ive_md
  ascendcl
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  hardware_decode
  avformat
  avcodec
  avutil
  swresample
  swscale
  stdc++ # note this line is important
)


target_link_libraries(md_yolo_merge_rois_sync_vdec_vpss_nnn
  ive_md
  ascendcl
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  hardware_decode
  avformat
  avcodec
  avutil
  swresample
  swscale
  stdc++ # note this line is important
)


target_link_libraries(md_yolo_merge_rois_async_vpss_nnn_2chns
  ive_md
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  ascendcl
  stdc++ # note this line is important
)

target_link_libraries(md4k_yoloThread_merge_rois_async_vpss_nnn_2chns
  ive_md
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  ascendcl
  stdc++ # note this line is important
)

target_link_libraries(md4k_yoloThread_merge_rois_async_vpss_nnn_2chns_v2
  ive_md
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  ascendcl
  stdc++ # note this line is important
)


target_link_libraries(md4k_yoloThread_merge_rois_async_vpss_nnn_1chn-1
  ive_md
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  ascendcl
  stdc++ # note this line is important
)

target_link_libraries(md4k_yoloThread_merge_rois_async_vpss_nnn_1chn-1_with_sky
  ive_md
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  ascendcl
  stdc++ # note this line is important
)

target_link_libraries(md4k_yoloThread_merge_rois_async_vpss_nnn_1chn-2
  ive_md
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  ascendcl
  stdc++ # note this line is important
)

target_link_libraries(md4k_yoloThread_merge_rois_async_vpss_nnn_1chn-2_with_sky
  ive_md
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  ascendcl
  stdc++ # note this line is important
)

target_link_libraries(md4k_yolo_merge_rois_async_vdec_vpss_nnn
  ive_md
  ascendcl
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  hardware_decode
  avformat
  avcodec
  avutil
  swresample
  swscale
  stdc++ # note this line is important
)

target_link_libraries(md4k_yoloThread_merge_rois_async_vdec_vpss_nnn
  ive_md
  ascendcl
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  hardware_decode
  avformat
  avcodec
  avutil
  swresample
  swscale
  stdc++ # note this line is important
)

target_link_libraries(md4k_yoloThread_merge_rois_async_vdec_vpss_nnn_with_sky
  ive_md
  ascendcl
  pthread
  ${YOLO_LIB}
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  hardware_decode
  avformat
  avcodec
  avutil
  swresample
  swscale
  stdc++ # note this line is important
)

target_link_libraries(test_vpss_get_chn
  pthread
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${IVE_LINK_LIBS}
  stdc++ # note this line is important
)

target_link_libraries(test_md_vpss
  ive_md
  pthread
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${IVE_LINK_LIBS}
  stdc++ # note this line is important
)
