cmake_minimum_required(VERSION 2.8.3)
# set project's name
project(drt_vio)

include(CheckCXXCompilerFlag)  # 改动
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

set(CMAKE_BUILD_TYPE "Release")  # Debug
set(CMAKE_CXX_FLAGS "-std=c++17")   # 改动
#set(CMAKE_CXX_STANDARD 17)          # 改动
#set(CMAKE_CXX_STANDARD_REQUIRED ON) # 改动
#set(CMAKE_CXX_EXTENSIONS OFF)       # 改动
#-DEIGEN_USE_MKL_ALL")
# eigen 3.3以上版本加-mavx -mfma
#set(CMAKE_CXX_FLAGS_RELEASE "-O3  -g -DEIGEN_NO_DEBUG -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DEIGEN_NO_DEBUG -g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -fPIC")

set(STD_CXX_FS stdc++fs)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_definitions("-D SOPHUS_USE_BASIC_LOGGING")
#add_definitions(-DBOOST_ERROR_CODE_HEADER_ONLY)
add_definitions(-DBOOST_LOG_DYN_LINK)  # 改动

add_subdirectory(thirdparty/basalt EXCLUDE_FROM_ALL) # 改动

find_package(Gflags)
find_package(Glog)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(TBB REQUIRED)
find_package(Boost REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Boost COMPONENTS filesystem log log_setup REQUIRED) # 改动
find_package(SuiteSparse) # 改动

include_directories(
  include
  include/odometry
  include/optimization
  include/common
  include/frames
  include/io
  include/features
  include/features/tracking
  include/rel_pose_estimation
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
  ${TBB_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}        # 改动
  ${SUITESPARSE_INCLUDE_DIRS}   # 改动
  ${Boost_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/thirdparty/basalt/include                  # 以下都是改动
  ${PROJECT_SOURCE_DIR}/thirdparty/basalt/thirdparty/basalt-headers/include
  ${PROJECT_SOURCE_DIR}/third_party/basalt/thirdparty/basalt-headers/thirdparty/Sophus
  ${PROJECT_SOURCE_DIR}/thirdparty/basalt/thirdparty/basalt-headers/thirdparty/cereal/include
  )

add_library(camera_model SHARED
        src/camera_models/chessboard/Chessboard.cc
        src/camera_models/calib/CameraCalibration.cc
        src/camera_models/camera_models/Camera.cc
        src/camera_models/camera_models/CameraFactory.cc
        src/camera_models/camera_models/CostFunctionFactory.cc
        src/camera_models/camera_models/PinholeCamera.cc
        src/camera_models/camera_models/CataCamera.cc
        src/camera_models/camera_models/EquidistantCamera.cc
        src/camera_models/camera_models/ScaramuzzaCamera.cc
        src/camera_models/sparse_graph/Transform.cc
        src/camera_models/gpl/gpl.cc
        src/camera_models/gpl/EigenQuaternionParameterization.cc)

target_link_libraries(camera_model ${Boost_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

ADD_LIBRARY(MyVio SHARED
        src/featureTracker/featureTracker.cpp
        src/featureTracker/parameters.cpp
        src/IMU/basicTypes.cpp
        src/IMU/imuPreintegrated.cpp
        src/initMethod/drtVioInit.cpp
        src/initMethod/drtLooselyCoupled.cpp
        src/initMethod/drtTightlyCoupled.cpp
        src/initMethod/polynomial.cc
        src/visualization/visualize.cpp
        src/factor/imuIntegFactor.cc
        src/features/base_matcher.cc          # 以下都是改动
        src/features/tracking_matcher.cc
        src/frames/base_frame.cc
        src/frames/keypoints.cc
        src/frames/tracking_frame.cc
        src/odometry/frame_processing.cc
        src/odometry/view_graph.cc
        src/optimization/nec_ceres.cc      
        src/optimization/pnec_ceres.cc
        src/optimization/scf.cc
        src/common/camera.cc
        src/common/common.cc
        src/common/converter.cc
        src/common/timing.cc
        src/common/visualization.cc
        src/rel_pose_estimation/essential_matrix_methods.cc
        src/rel_pose_estimation/frame2frame.cc
        src/rel_pose_estimation/pnec.cc        # 这里出问题
        src/visualization/visualize.cpp
        src/io/config.cc
        src/io/dataset_loader.cc
        src/io/odometry_output.cc
        )

target_link_libraries(MyVio
        ${TBB_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        ${Boost_LIBRARIES}
        ${STD_CXX_FS}
        ${SUITESPARSE_LIBRARIES}    # 以下都是改动
        ${Boost_FILESYSTEM_LIBRARY} 
        ${Boost_LOG_LIBRARY}
        ${Boost_LOG_SETUP_LIBRARY}
        ${OpenCV_LIBRARIES}
        camera_model
        #umfpack
        -lboost_system
        opengv            # 只是因为顺序不一样？
        basalt
        )



add_executable(run_euroc app/main.cpp)
#add_executable(run_euroc app/imu_aided_vp_detection2_static.cpp)


target_link_libraries(run_euroc
        MyVio
        ${Pangolin_LIBRARIES}
        -lpthread)
