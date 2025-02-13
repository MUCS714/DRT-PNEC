/**
 BSD 3-Clause License

 This file is part of the PNEC project.
 https://github.com/tum-vision/pnec

 Copyright (c) 2022, Dominik Muhle.
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "frame2frame.h"

#include "math.h"
#include <Eigen/Core>
#include <boost/filesystem.hpp>
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/CentralRelativeWeightingAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Lmeds.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/EigensolverSacProblem.hpp>
#include <opengv/types.hpp>

#include "common/common.h"
#include "essential_matrix_methods.h"
#include "odometry_output.h"
#include "pnec.h"

namespace pnec {
namespace rel_pose_estimation {

Frame2Frame::Frame2Frame(const pnec::rel_pose_estimation::Options &options)
    : options_{options} {}
Frame2Frame::~Frame2Frame() {}


Sophus::SE3d Frame2Frame::Align(pnec::frames::BaseFrame::Ptr frame1,
                                pnec::frames::BaseFrame::Ptr frame2,
                                pnec::FeatureMatches &matches,
                                Sophus::SE3d prev_rel_pose,
                                std::vector<int> &inliers,
                                pnec::common::FrameTiming &frame_timing,
                                bool ablation, std::string ablation_folder) {
  if (ablation) {
    if (!boost::filesystem::exists(ablation_folder)) {
      boost::filesystem::create_directory(ablation_folder);
    }
  }

  curr_timestamp_ = frame2->Timestamp();

  opengv::bearingVectors_t bvs1;
  opengv::bearingVectors_t bvs2;
  std::vector<Eigen::Matrix3d> proj_covs;
  GetFeatures(frame1, frame2, matches, bvs1, bvs2, proj_covs);

  // return PNECAlign(bvs1, bvs2, proj_covs, prev_rel_pose, inliers, frame_timing,
  //                  ablation_folder);
  return Sophus::SE3d();
}

const std::map<std::string, std::vector<std::pair<double, Sophus::SE3d>>> &
Frame2Frame::GetAblationResults() const {
  return ablation_rel_poses_;
}

Sophus::SE3d Frame2Frame::PNECAlign(
    const opengv::bearingVectors_t &bvs1, const opengv::bearingVectors_t &bvs2,
    const std::vector<Eigen::Matrix3d> &projected_covs,
    Sophus::SE3d prev_rel_pose, std::vector<int> &inliers,
    pnec::common::FrameTiming &frame_timing, std::string ablation_folder) {
  options_.use_nec_ = true;
  options_.use_ceres_ = false;
  pnec::rel_pose_estimation::PNEC pnec(options_);
  
  Sophus::SE3d rel_pose =
      pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, inliers, frame_timing);

  if (ablation_rel_poses_.count("PNEC") == 0) {
    ablation_rel_poses_["PNEC"] = {
        std::make_pair<double, Sophus::SE3d>(0.0, Sophus::SE3d())};
  }
  ablation_rel_poses_["PNEC"].push_back(
      std::make_pair(curr_timestamp_, rel_pose));
  return rel_pose;
}


void Frame2Frame::GetFeatures(pnec::frames::BaseFrame::Ptr host_frame,
                              pnec::frames::BaseFrame::Ptr target_frame,
                              pnec::FeatureMatches &matches,
                              opengv::bearingVectors_t &host_bvs,
                              opengv::bearingVectors_t &target_bvs,
                              std::vector<Eigen::Matrix3d> &proj_covs) {
  std::vector<size_t> host_matches;
  std::vector<size_t> target_matches;
  for (const auto &match : matches) {
    host_matches.push_back(match.queryIdx);
    target_matches.push_back(match.trainIdx);
  }
  pnec::features::KeyPoints host_keypoints =
      host_frame->keypoints(host_matches);
  pnec::features::KeyPoints target_keypoints =
      target_frame->keypoints(target_matches);

  std::vector<Eigen::Matrix3d> host_covs;
  std::vector<Eigen::Matrix3d> target_covs;
  for (auto const &[id, keypoint] : host_keypoints) {
    host_bvs.push_back(keypoint.bearing_vector_);
    host_covs.push_back(keypoint.bv_covariance_);
  }
  for (auto const &[id, keypoint] : target_keypoints) {
    target_bvs.push_back(keypoint.bearing_vector_);
    target_covs.push_back(keypoint.bv_covariance_);
  }

  if (options_.noise_frame_ == pnec::common::Host) {
    proj_covs = host_covs;
  } else {
    proj_covs = target_covs;
  }
}
} // namespace rel_pose_estimation
} // namespace pnec