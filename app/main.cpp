//
// Created by ubuntu on 2020/9/1.
//

#include "io/datasetIO.h"
#include "io/datasetIOEuroc.h"
#include "featureTracker/featureTracker.h"
#include "featureTracker/parameters.h"
#include "IMU/imuPreintegrated.hpp"
#include "initMethod/drtVioInit.h"
#include "initMethod/drtLooselyCoupled.h"
#include "initMethod/drtTightlyCoupled.h"
#include "utils/eigenUtils.hpp"
#include "utils/ticToc.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <glog/logging.h>
#include <string>

#include "config.h"
#include "klt_patch_optical_flow.h"
#include "base_matcher.h"
#include "tracking_matcher.h"
#include "frame2frame.h"
#include "timing.h"
#include "base_frame.h"
#include "frame_processing.h"
#include "nec_ceres.h"
#include "pnec_ceres.h"
#include "tracking_frame.h"
#include "view_graph.h"
#include "dataset_loader.h"

#include <boost/filesystem.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <unordered_map>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <sophus/se3.hpp>



using namespace std;
using namespace cv;

struct PNECObservation1 {
  PNECObservation1()
      : transform(Eigen::AffineCompact2d()), covariance(Eigen::Matrix2d()),
        hessian(Eigen::Matrix3d()) {}
  PNECObservation1(const Eigen::AffineCompact2d &t, const Eigen::Matrix2d &cov,
                  const Eigen::Matrix3d &h)
      : transform(t), covariance(cov), hessian(h) {}
  Eigen::AffineCompact2d transform;
  Eigen::Matrix2d covariance;
  Eigen::Matrix3d hessian;
};

Eigen::Matrix3d UnscentedTransform(const Eigen::Vector3d &mu,
                                   const Eigen::Matrix3d &cov,
                                   const Eigen::Matrix3d &K_inv, double kappa) {
  // Pass mu[0], mu[1], 1.0
  int n = 2;
  int m = 2 * n + 1;

  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
  C.topLeftCorner(2, 2) = cov.topLeftCorner(2, 2).llt().matrixL();
  
  std::vector<Eigen::Vector3d> points;
  points.reserve(m);
  std::vector<double> weights;
  weights.reserve(m);
  points.push_back(mu);
  weights.push_back(kappa / ((float)n + kappa));
  for (int i = 0; i < n; i++) {
    Eigen::Vector3d c_col = C.col(i);
    points.push_back(mu + c_col);
    weights.push_back(0.5 / ((float)n + kappa));
  }
  for (int i = 0; i < n; i++) {
    Eigen::Vector3d c_col = C.col(i);
    points.push_back(mu - c_col);
    weights.push_back(0.5 / ((float)n + kappa));
  }

  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> transformed_points;
  transformed_points.reserve(m);
  for (int i = 0; i < m; i++) {
    Eigen::Vector3d t_point;
    t_point = (K_inv * points[i]).normalized();
    transformed_points.push_back(t_point);
    mean = mean + (weights[i] * t_point);
  }
  Eigen::Matrix3d sigma = Eigen::Matrix3d::Zero();
  for (int i = 0; i < m; i++) {
    sigma = sigma + weights[i] * (transformed_points[i] - mean) *
                        (transformed_points[i] - mean).transpose();
  }
  return sigma;
}

basalt::OpticalFlowInput::Ptr OpticalFlowFromOpenCV(const cv::Mat &image,
                                                    const int64_t timestamp) {
  std::vector<basalt::ImageData> res(1);
  res[0].img.reset(new basalt::ManagedImage<uint16_t>(image.cols, image.rows));

  const uint8_t *data_in = image.ptr();
  uint16_t *data_out = res[0].img->ptr;

  size_t full_size = image.cols * image.rows;
  for (size_t i = 0; i < full_size; i++) {
    int val = data_in[i];
    val = val << 8;
    data_out[i] = val;
  }

  basalt::OpticalFlowInput::Ptr data(new basalt::OpticalFlowInput);

  data->t_ns = timestamp;
  data->img_data = res;

  return data;
}

int main(int argc, char **argv) {


    bool use_single_ligt = false;
    bool use_ligt_vins = false;

    if (argc != 3) {
        std::cout << "Usage: code" << " code type" << " data type"  << "\n";
        return -1;
    }

    char *codeType = argv[1];
    char *dataType = argv[2];
    std::ofstream save_file("../result/" + string(codeType) + "_" + string(dataType) + ".txt");

    auto dataset_io = struct_vio::EurocIO();
    dataset_io.read("/home/jadeting/muchangshi/drt_pnec/config/euroc.yaml");
    auto vio_dataset = dataset_io.get_data();
    readParameters("/home/jadeting/muchangshi/drt_pnec/config/euroc.yaml");
    FeatureTracker trackerData;
    trackerData.readIntrinsicParameter("/home/jadeting/muchangshi/drt_pnec/config/euroc.yaml");
    PUB_THIS_FRAME = true;
     if(FISHEYE)
    {
        trackerData.fisheye_mask = cv::imread(FISHEYE_MASK, 0);
        if(!trackerData.fisheye_mask.data)
        {
            cout << "load mask fail" << endl;
        }
        else
        {
            cout << "load mask success" << endl;
        }
    }
    double sf = std::sqrt(double(IMU_FREQ));

    
    // new code 
    const std::string tracking_config_filename = "/home/jadeting/muchangshi/drt_pnec/config/tracking_config.json";
    const std::string tracking_calib_filename = "/home/jadeting/muchangshi/drt_pnec/config/tracking_calib.json";
    const std::string camera_config_filename = "/home/jadeting/muchangshi/drt_pnec/config/config_euroc.yaml";
    const std::string pnec_config_filename = "/home/jadeting/muchangshi/drt_pnec/config/pnec_config.yaml";

    pnec::CameraParameters cam_parameters = pnec::input::LoadCameraConfig(camera_config_filename);
    pnec::rel_pose_estimation::Options pnec_options = pnec::input::LoadPNECConfig(pnec_config_filename);
    basalt::Calibration<double> tracking_calib = pnec::input::LoadTrackingCalib(tracking_calib_filename);
    basalt::VioConfig tracking_config = pnec::input::LoadTrackingConfig(tracking_config_filename);
    
    // =============================== TRACKING =============================== 
    basalt::KLTPatchOpticalFlow<float, basalt::Pattern52> tracking(
    tracking_config, tracking_calib, true, false);

    // =============================== MATCHER ================================ 
    pnec::features::BaseMatcher::Ptr matcher;
    pnec::features::TrackingMatcher::Ptr tracking_matcher =
        std::make_shared<pnec::features::TrackingMatcher>(50, 5.0);
    matcher = tracking_matcher;

    // ========================= REL POSE ESTIMATION ========================== 
    pnec::rel_pose_estimation::Frame2Frame::Ptr rel_pose_estimation;
    rel_pose_estimation =
        std::make_shared<pnec::rel_pose_estimation::Frame2Frame>(pnec_options);

    pnec::Camera::instance().init(cam_parameters);
    pnec::common::Timing timing;

    // ============================== VIEW GRAPH ============================== 
    int max_graph_size = 20;
    const std::string results_path = "/home/jadeting/muchangshi/drt_pnec/result/pnec";
    pnec::odometry::ViewGraph::Ptr view_graph(
        new pnec::odometry::ViewGraph(max_graph_size, results_path));


    // =========================== FRAME PROCESSING =========================== 
    const std::string visualization_path = "/home/jadeting/muchangshi/drt_pnec/result/visualization";
    pnec::visualization::Options vis_options(
        visualization_path, pnec::visualization::Options::NO,
        pnec::visualization::Options::INLIER);
    pnec::odometry::FrameProcessing frame_processor(
        view_graph, rel_pose_estimation, matcher, true, vis_options);


    for (int i = 0; i < vio_dataset->get_image_timestamps().size() - 100; i += 10) {

        DRT::drtVioInit::Ptr  pDrtVioInit;
        if (string(codeType) == "drtTightly")
        {
            pDrtVioInit.reset(new DRT::drtTightlyCoupled(RIC[0], TIC[0]));
        }

        if (string(codeType) == "drtLoosely")
        {
            pDrtVioInit.reset(new DRT::drtLooselyCoupled(RIC[0], TIC[0]));
        }


        std::vector<int> idx;

        // 40 4HZ, 0.25s
        for (int j = i; j < 100 + i; j += 1)
            idx.push_back(j);                                                  

        double last_img_t_s, cur_img_t_s;
        bool first_img = true;
        bool init_feature = true;


        trackerData.reset();

        std::vector<double> idx_time;
        std::vector<cv::Mat> imgs;

        for (int i: idx) {

            int64_t t_ns = vio_dataset->get_image_timestamps()[i];
            

            cur_img_t_s = t_ns * 1e-9;
            

            cv::Mat img = vio_dataset->get_image_data(t_ns)[0].image;

            trackerData.readImage(img, t_ns * 1e-9);

            auto tic = std::chrono::high_resolution_clock::now(),
                 toc = std::chrono::high_resolution_clock::now();

            pnec::common::FrameTiming frame_timing(t_ns);
            pnec::frames::BaseFrame::Ptr f;
            f.reset(new pnec::frames::TrackingFrame(t_ns, cur_img_t_s, tracking, frame_timing, img));   
            toc = std::chrono::high_resolution_clock::now();
            frame_timing.feature_creation_ = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic) - frame_timing.frame_loading_;
            
          
            for (unsigned int i = 0;; i++) {
                bool completed = false;
                completed |= trackerData.updateID(i);
                if (!completed)
                    break;
            }

            

            // auto &un_pts = trackerData.cur_un_pts;
            // auto &cur_pts = trackerData.cur_pts;
            // auto &ids = trackerData.ids;                   
            // auto &pts_velocity = trackerData.pts_velocity;
            // new code
            auto &reserved_forw_pts = trackerData.reserved_forw_pts;
            auto &reserved_cur_pts = trackerData.prev_pts;
            //auto &pre_un_pts = trackerData.prev_un_pts;

            pnec::features::KeyPoints cur_keypoints = f->keypoints();

            Eigen::aligned_map<pnec::features::KeyPointID, Eigen::aligned_vector<pair<int, Eigen::Matrix<double, 3, 1 >> >>
                    image;
            // Eigen::aligned_map<int, Eigen::aligned_vector<pair<int, Eigen::Matrix<double, 3, 1 >> >>
            //         image;
            // for (unsigned int i = 0; i < ids.size(); i++) {
            //     if (trackerData.track_cnt[i] > 1) {                         
            //         int v = ids[i];
            //         int feature_id = v / NUM_OF_CAM;
            //         int camera_id = v % NUM_OF_CAM;
            //         double x = un_pts[i].x;
            //         double y = un_pts[i].y;
            //         double z = 1;
            //         // double p_u = cur_pts[i].x;
            //         // double p_v = cur_pts[i].y;
            //         // double velocity_x = pts_velocity[i].x;
            //         // double velocity_y = pts_velocity[i].y;
            //         assert(camera_id == 0);
            //         Eigen::Matrix<double, 3, 1> xyz_uv_velocity;
            //         //xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            //         xyz_uv_velocity << x, y, z;
            //         image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            //     }
            // }

             if (init_feature) {
                init_feature = false;
                continue;
            }

            for (const auto& pair : cur_keypoints) {        
                pnec::features::KeyPointID v = pair.first;
                pnec::features::KeyPointID feature_id = v / NUM_OF_CAM;
                pnec::features::KeyPointID camera_id = v % NUM_OF_CAM;
                Eigen::Vector3d bv = pair.second.bearing_vector_;
                assert(camera_id == 0);
                Eigen::Matrix<double, 3, 1> xyz_uv_velocity;
                xyz_uv_velocity << bv(0), bv(1), bv(2);
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }


            if (pDrtVioInit->addFeatureCheckParallax(cur_img_t_s, image, 0.0)) {

                idx_time.push_back(cur_img_t_s);

                std::cout << "add image is: " << fixed << cur_img_t_s << " image number is: " << idx_time.size()
                          << std::endl;

                // cov hessian
                // std::vector<PNECObservation1> cur_observations;  
                std::vector<cv::Point2f> src_pts = reserved_cur_pts; // reserved_cur_pts = pre_pts
                src_pts.resize(reserved_forw_pts.size());
                
                cv::Mat point_transform = cv::estimateAffine2D(src_pts, reserved_forw_pts);
                Eigen::Matrix3d eigen_mat;
                eigen_mat << point_transform.at<double>(0, 0), point_transform.at<double>(0, 1), point_transform.at<double>(0, 2),
                            point_transform.at<double>(1, 0), point_transform.at<double>(1, 1), point_transform.at<double>(1, 2),
                            0, 0, 1;
                Eigen::AffineCompact2d eigen_transform(eigen_mat);

                basalt::KLTPatchOpticalFlow<float, basalt::Pattern52> tracking(true, false);
                
                basalt::OpticalFlowInput::Ptr img_ptr = OpticalFlowFromOpenCV(img, t_ns);
            
                Eigen::Matrix3d K;
                K << 458.654, 0, 367.215,
                    0, 457.296, 248.375,
                    0, 0, 1;
                Eigen::Matrix3d K_inv = K.inverse();
                for (unsigned int i = 0; i < reserved_forw_pts.size(); i++) {
                    if (trackerData.track_cnt[i] > 1) {                         
                        int v = ids[i];
                        int feature_id = v / NUM_OF_CAM;
                
                        Eigen::Vector2d pts;
                        pts << static_cast<double>(reserved_forw_pts[i].x) , static_cast<double>(reserved_forw_pts[i].y);
                        basalt::PNECObservation result = tracking.processFrame(img_ptr, pts);

                        PNECObservation1 obs(eigen_transform, result.covariance / 10.0, result.hessian * 10.0);

                        //cur_observations.push_back(obs);
                        // fx: 458.654
                        // fy: 457.296
                        // cx: 367.215
                        // cy: 248.375
                        Eigen::Vector3d mu(cur_pts[i].x, cur_pts[i].y, 1.0);
                        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
                        cov.topLeftCorner(2, 2) = obs.covariance;
                        Eigen::Matrix3d bv_covariance_ = UnscentedTransform(mu, cov, K_inv, 1.0);
                        pDrtVioInit->SFMConstruct[feature_id].obs[cur_img_t_s].bv_covariance = bv_covariance_;
                        
                    }
                }
                

                if (first_img) {
                    last_img_t_s = cur_img_t_s;
                    first_img = false;
                    continue;
                }

                auto GyroData = vio_dataset->get_gyro_data();
                auto AccelData = vio_dataset->get_accel_data();

                std::vector<MotionData> imu_segment;

                for (size_t i = 0; i < GyroData.size(); i++) {
                    double timestamp = GyroData[i].timestamp_ns * 1e-9;

                    MotionData imu_data;
                    imu_data.timestamp = timestamp;
                    imu_data.imu_acc = AccelData[i].data;
                    imu_data.imu_gyro = GyroData[i].data;

                    if (timestamp > last_img_t_s && timestamp <= cur_img_t_s) {
                        imu_segment.push_back(imu_data);
                    }
                    if (timestamp > cur_img_t_s) {
                        imu_segment.push_back(imu_data);
                        break;
                    }
                }

                vio::IMUBias bias;
                vio::IMUCalibParam
                        imu_calib(RIC[0], TIC[0], GYR_N * sf, ACC_N * sf, GYR_W / sf, ACC_W / sf);
                vio::IMUPreintegrated imu_preint(bias, &imu_calib, last_img_t_s, cur_img_t_s);

                int n = imu_segment.size() - 1;

                for (int i = 0; i < n; i++) {
                    double dt;
                    Eigen::Vector3d gyro;
                    Eigen::Vector3d acc;

                    if (i == 0 && i < (n - 1))               // [start_time, imu[0].time]
                    {
                        float tab = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                        float tini = imu_segment[i].timestamp - last_img_t_s;
                        CHECK(tini >= 0);
                        acc = (imu_segment[i + 1].imu_acc + imu_segment[i].imu_acc -
                               (imu_segment[i + 1].imu_acc - imu_segment[i].imu_acc) * (tini / tab)) * 0.5f;
                        gyro = (imu_segment[i + 1].imu_gyro + imu_segment[i].imu_gyro -
                                (imu_segment[i + 1].imu_gyro - imu_segment[i].imu_gyro) * (tini / tab)) * 0.5f;
                        dt = imu_segment[i + 1].timestamp - last_img_t_s;
                    } else if (i < (n - 1))      // [imu[i].time, imu[i+1].time]
                    {
                        acc = (imu_segment[i].imu_acc + imu_segment[i + 1].imu_acc) * 0.5f;
                        gyro = (imu_segment[i].imu_gyro + imu_segment[i + 1].imu_gyro) * 0.5f;
                        dt = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                    } else if (i > 0 && i == n - 1) {
                        // std::cout << " n : " << i + 1 << " " << n << " " << imu_segment[i + 1].timestamp << std::endl;
                        float tab = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                        float tend = imu_segment[i + 1].timestamp - cur_img_t_s;
                        CHECK(tend >= 0);
                        acc = (imu_segment[i].imu_acc + imu_segment[i + 1].imu_acc -
                               (imu_segment[i + 1].imu_acc - imu_segment[i].imu_acc) * (tend / tab)) * 0.5f;
                        gyro = (imu_segment[i].imu_gyro + imu_segment[i + 1].imu_gyro -
                                (imu_segment[i + 1].imu_gyro - imu_segment[i].imu_gyro) * (tend / tab)) * 0.5f;
                        dt = cur_img_t_s - imu_segment[i].timestamp;
                    } else if (i == 0 && i == (n - 1)) {
                        acc = imu_segment[i].imu_acc;
                        gyro = imu_segment[i].imu_gyro;
                        dt = cur_img_t_s - last_img_t_s;
                    }

                    CHECK(dt >= 0);
                    imu_preint.integrate_new_measurement(gyro, acc, dt);
                }
                // std::cout << fixed << "cur time: " << cur_img_t_s << " " << "last time: " << last_img_t_s << std::endl;

                pDrtVioInit->addImuMeasure(imu_preint);

                last_img_t_s = cur_img_t_s;


            }

            if (idx_time.size() >= 10) break;

            if (SHOW_TRACK) {
                cv::Mat show_img;
                cv::cvtColor(img, show_img, CV_GRAY2RGB);
                for (unsigned int j = 0; j < trackerData.cur_pts.size(); j++) {
                    double len = min(1.0, 1.0 * trackerData.track_cnt[j] / WINDOW_SIZE);
                    cv::circle(show_img, trackerData.cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                }

                cv::namedWindow("IMAGE", WINDOW_AUTOSIZE);
                cv::imshow("IMAGE", show_img);
                cv::waitKey(1);
            }

            cv::Mat show_img;
            cv::cvtColor(img, show_img, CV_GRAY2RGB);
            for (unsigned int j = 0; j < trackerData.cur_pts.size(); j++) {
                double len = min(1.0, 1.0 * trackerData.track_cnt[j] / WINDOW_SIZE);
                cv::circle(show_img, trackerData.cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
            }

            imgs.push_back(show_img);
        }

        if (idx_time.size() < 10) continue;

        bool is_good = pDrtVioInit->checkAccError();

        if (!is_good)
        {
            std::cout << "not good" << std::endl;
            continue;
        }

        if ( !pDrtVioInit->process()) {
            save_file << "time: " << fixed << idx_time[0] << " other_reason" << std::endl;
            save_file << "scale_error: " << "nan" << std::endl;
            save_file << "pose_error: " << "nan" << std::endl;
            save_file << "biasg_error: " << "nan" << std::endl;
            save_file << "biasa_error: " << "nan" << std::endl;
            save_file << "velo_error: " << "nan" << std::endl;
            save_file << "gravity_error: " << "nan" << std::endl;
            save_file << "rot_error: " << "nan"  << std::endl;
            save_file << "v0_error: "  << "nan" << std::endl;
            save_file << "gt_vel_rot: " << "nan" << " " << "nan" << std::endl;
            LOG(INFO) << "---scale: ";
            std::cout << "time: " << fixed << idx_time[0] << std::endl;
            std::cout << "scale_error: " << 100 << std::endl;
            std::cout << "pose_error: " << 100 << std::endl;
            std::cout << "biasg_error: " << 100 << std::endl;
            std::cout << "biasa_error: " << 100 << std::endl;
            std::cout << "velo_error: " << 100 << std::endl;
            std::cout << "rot_error: " << 100 << std::endl;
            continue;
        }

        // 获取真实值
        std::vector<Eigen::Vector3d> gt_pos;
        std::vector<Eigen::Matrix3d> gt_rot;
        std::vector<Eigen::Vector3d> gt_vel;
        std::vector<Eigen::Vector3d> gt_g_imu;
        std::vector<Eigen::Vector3d> gt_angluar_vel;
        Eigen::Vector3d avgBg;
        Eigen::Vector3d avgBa;
        avgBg.setZero();
        avgBa.setZero();

        auto get_traj = [&](double timeStamp, struct_vio::GtData &rhs) -> bool {
            Eigen::map<double, struct_vio::GtData> gt_data = vio_dataset->get_gt_state_data();

            for (const auto &traj: gt_data) {
                if (std::abs((traj.first - timeStamp)) < 1e-3) {
                    rhs = traj.second;
                    return true;
                }
            }
            return false;
        };

        try {
            for (auto &t: idx_time) {
                struct_vio::GtData rhs;
                if (get_traj(t, rhs)) {
                    gt_pos.emplace_back(rhs.position);
                    gt_vel.emplace_back(rhs.velocity);
                    gt_rot.emplace_back(rhs.rotation.toRotationMatrix());
                    gt_g_imu.emplace_back(rhs.rotation.inverse() * G);

                    avgBg += rhs.bias_gyr;
                    avgBa += rhs.bias_acc;
                    
                } else {
                    std::cout << "no gt pose,fail" << std::endl;
                    throw -1;
                }
            }
        } catch (...) {
            save_file << "time: " << fixed << idx_time[0] << " other_reason" << std::endl;
            save_file << "scale_error: " << "nan" << std::endl;
            save_file << "pose_error: " << "nan" << std::endl;
            save_file << "biasg_error: " << "nan" << std::endl;
            save_file << "biasa_error: " << "nan" << std::endl;
            save_file << "velo_error: " << "nan" << std::endl;
            save_file << "gravity_error: " << "nan" << std::endl;
            save_file << "rot_error: " << "nan"  << std::endl;
            save_file << "v0_error: " << "nan" << std::endl;
            LOG(INFO) << "---scale: ";
            std::cout << "time: " << fixed << idx_time[0] << std::endl;
            std::cout << "scale_error: " << 100 << std::endl;
            std::cout << "pose_error: " << 100 << std::endl;
            std::cout << "biasg_error: " << 100 << std::endl;
            std::cout << "biasa_error: " << 100 << std::endl;
            std::cout << "velo_error: " << 100 << std::endl;
            std::cout << "rot_error: " << 100 << std::endl;
            continue;
        }

        avgBg /= idx_time.size();
        avgBa /= idx_time.size();

        double rot_rmse = 0;

        // rotation accuracy estimation
        for (int i = 0; i < idx_time.size() - 1; i++) {
            int j = i + 1;
            Eigen::Matrix3d rij_est = pDrtVioInit->rotation[i].transpose() * pDrtVioInit->rotation[j];
            Eigen::Matrix3d rij_gt = gt_rot[i].transpose() * gt_rot[j];
            Eigen::Quaterniond qij_est = Eigen::Quaterniond(rij_est);
            Eigen::Quaterniond qij_gt = Eigen::Quaterniond(rij_gt);
            double error =
                    std::acos(((qij_gt * qij_est.inverse()).toRotationMatrix().trace() - 1.0) / 2.0) * 180.0 / M_PI;
            rot_rmse += error * error;
        }
        rot_rmse /= (idx_time.size() - 1);
        rot_rmse = std::sqrt(rot_rmse);

        // translation accuracy estimation
        Eigen::Matrix<double, 3, Eigen::Dynamic> est_aligned_pose(3, idx_time.size());
        Eigen::Matrix<double, 3, Eigen::Dynamic> gt_aligned_pose(3, idx_time.size());

        for (int i = 0; i < idx_time.size(); i++) {
            est_aligned_pose(0, i) = pDrtVioInit->position[i](0);
            est_aligned_pose(1, i) = pDrtVioInit->position[i](1);
            est_aligned_pose(2, i) = pDrtVioInit->position[i](2);

            gt_aligned_pose(0, i) = gt_pos[i](0);
            gt_aligned_pose(1, i) = gt_pos[i](1);
            gt_aligned_pose(2, i) = gt_pos[i](2);
        }


        Eigen::Matrix4d Tts = Eigen::umeyama(est_aligned_pose, gt_aligned_pose, true);
        Eigen::Matrix3d cR = Tts.block<3, 3>(0, 0);
        Eigen::Vector3d t = Tts.block<3, 1>(0, 3);
        double s = cR.determinant();
        s = pow(s, 1.0 / 3);
        Eigen::Matrix3d R = cR / s;

        double pose_rmse = 0;
        for (int i = 0; i < idx_time.size(); i++) {
            Eigen::Vector3d target_pose = R * est_aligned_pose.col(i) + t;
            pose_rmse += (target_pose - gt_aligned_pose.col(i)).dot(target_pose - gt_aligned_pose.col(i));

        }
        pose_rmse /= idx_time.size();
        pose_rmse = std::sqrt(pose_rmse);

        std::cout << "vins sfm pose rmse: " << pose_rmse << std::endl;

        // gravity accuracy estimation
        double gravity_error =
                180. * std::acos(pDrtVioInit->gravity.normalized().dot(gt_g_imu[0].normalized())) / EIGEN_PI;

        // gyroscope bias and accelerometor bias accuracy estimation
        Eigen::Vector3d Bgs = pDrtVioInit->biasg;
        Eigen::Vector3d Bac = pDrtVioInit->biasa;

        LOG(INFO) << "calculate biasg: " << Bgs.x() << " " << Bgs.y() << " " << Bgs.z();
        LOG(INFO) << "gt biasg: " << avgBg.x() << " " << avgBg.y() << " " << avgBg.z();
        
        LOG(INFO) << "calculate biasa: " << Bac.x() << " " << Bac.y() << " " << Bac.z();
        LOG(INFO) << "gt biasa: " << avgBa.x() << " " << avgBa.y() << " " << avgBa.z();
        

        const double scale_error = std::abs(s - 1.);
        const double gyro_bias_error = 100. * std::abs(Bgs.norm() - avgBg.norm()) / avgBg.norm();
        const double gyro_bias_error2 = 180. * std::acos(Bgs.normalized().dot(avgBg.normalized())) / EIGEN_PI;
        const double acc_bias_error = 100. * std::abs(Bac.norm() - avgBa.norm()) / avgBa.norm();
        //const double acc_bias_error2 = 180. * std::acos(Bac.normalized().dot(avgBa.normalized())) / EIGEN_PI;
        const double pose_error = pose_rmse;
        const double rot_error = rot_rmse;


        // velocity accuracy estimation
        double velo_norm_rmse = 0;
        double mean_velo = 0;
        for (int i = 0; i < idx_time.size(); i++) {
            velo_norm_rmse += (gt_vel[i].norm() - pDrtVioInit->velocity[i].norm()) *
                              (gt_vel[i].norm() - pDrtVioInit->velocity[i].norm());
            mean_velo += gt_vel[i].norm();
        }

        velo_norm_rmse /= idx_time.size();
        velo_norm_rmse = std::sqrt(velo_norm_rmse);
        mean_velo = mean_velo / idx_time.size();

        // the initial velocity accuracy estimation
        double v0_error = std::abs(gt_vel[0].norm() - pDrtVioInit->velocity[0].norm());

        std::cout << "integrate time: " << fixed << *idx_time.begin() << " " << *idx_time.rbegin() << " "
                  << *idx_time.rbegin() - *idx_time.begin() << std::endl;
        std::cout << "pose error: " << pose_error << " m" << std::endl;
        std::cout << "biasg error: " << gyro_bias_error << " %" << std::endl;
        std::cout << "biasa_error: " << acc_bias_error << " %" << std::endl;
        std::cout << "gravity_error: " << gravity_error << std::endl;
        std::cout << "scale error: " << scale_error * 100 << " %" << std::endl;
        std::cout << "velo error: " << velo_norm_rmse << " m/s" << std::endl;
        std::cout << "v0_error: " << v0_error << std::endl;
        std::cout << "rot error: " << rot_error << std::endl;

        if (std::abs(s - 1) > 0.5 or std::abs(gravity_error) > 10) {
            LOG(INFO) << "===scale: " << s << " " << "gravity error: " << gravity_error;
            save_file << "time: " << fixed << idx_time[0] << " scale_gravity_fail" << std::endl;
            save_file << "scale_error: " << scale_error << std::endl;
            save_file << "pose_error: " << pose_error << std::endl;
            save_file << "biasg_error: " << gyro_bias_error << std::endl;
            save_file << "biasa_error: " << acc_bias_error << std::endl;
            save_file << "velo_error: " << velo_norm_rmse << std::endl;
            save_file << "gravity_error: " << gravity_error << std::endl;
            save_file << "rot_error: " << rot_error << std::endl;
            save_file << "v0_error: " << v0_error << std::endl;
        } else {
            LOG(INFO) << "***scale: " << s << " " << "gravity error: " << gravity_error;
            save_file << "time: " << fixed << idx_time[0] << " good" << std::endl;
            save_file << "scale_error: " << scale_error << std::endl;
            save_file << "pose_error: " << pose_error << std::endl;
            save_file << "biasg_error: " << gyro_bias_error << std::endl;
            save_file << "biasa_error: " << acc_bias_error << std::endl;
            save_file << "velo_error: " << velo_norm_rmse << std::endl;
            save_file << "gravity_error: " << gravity_error << std::endl;
            save_file << "rot_error: " << rot_error << std::endl;
            save_file << "v0_error: " << v0_error << std::endl;
        }

    }

}


