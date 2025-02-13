/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
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

// There are changes compared to the original file at
// https://gitlab.com/VladyslavUsenko/basalt/-/blob/master/include/basalt/optical_flow/patch_optical_flow.h

#pragma once

#include <boost/log/trivial.hpp>
#include <iostream>
#include <thread>

#include <sophus/se2.hpp>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>

#include "pnec_patch.h"
#include <basalt/optical_flow/optical_flow.h>

#include <basalt/image/image_pyr.h>
#include <basalt/utils/keypoints.h>

namespace basalt {

struct PNECObservation {
  PNECObservation()
      : covariance(Eigen::Matrix2d()), hessian(Eigen::Matrix3d()) {}
  PNECObservation(const Eigen::Matrix2d &cov, const Eigen::Matrix3d &h)
      : covariance(cov), hessian(h) {}
  
  Eigen::Matrix2d covariance;
  Eigen::Matrix3d hessian;
};

struct PNECOpticalFlowResult : OpticalFlowResult {
  using Ptr = std::shared_ptr<PNECOpticalFlowResult>;

  std::vector<Eigen::aligned_map<KeypointId, PNECObservation>> observations;
};

template <typename Map1, typename Map2>
bool key_compare(Map1 const &lhs, Map2 const &rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                    [](auto a, auto b) { return a.first == b.first; });
}

template <typename Scalar, template <typename> typename Pattern>
class KLTPatchOpticalFlow : public OpticalFlowBase {
public:
  typedef POpticalFlowPatch<Scalar, Pattern<Scalar>> PatchT;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 2, 2> Matrix2;

  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;

  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
  typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;

  typedef Sophus::SE2<Scalar> SE2;


  //onst VioConfig &config,
  //const basalt::Calibration<double> &calib
  KLTPatchOpticalFlow(const bool use_mahalanobis = true,
                      const bool numerical_cov = true)
      : use_mahalanobis_{use_mahalanobis}, numerical_cov_(numerical_cov){
    input_queue.set_capacity(10);
    patch_coord = PatchT::pattern2.template cast<float>();
  }

  ~KLTPatchOpticalFlow() {}

  PNECObservation
  processFrame(OpticalFlowInput::Ptr &new_img_vec, const Eigen::Vector2d &pts) {
  
      pyramid.reset(new std::vector<basalt::ManagedImagePyr<u_int16_t>>);
      pyramid->resize(1);
      pyramid->at(0).setFromImage(*new_img_vec->img_data[0].img,1 );

      PatchT p;
      Vector2 pos = pts.cast<Scalar>();
      for (int l = 0; l < 1; l++) {
        Scalar scale = 1 << l;
        Vector2 pos_scaled = pos / scale;
        p = PatchT(pyramid->at(0).lvl(l), pos_scaled);
      }
      
      PNECObservation obs(p.Cov.template cast<double>() / 10.0, p.H_se2.template cast<double>() * 10.0);
      return obs;

  }  

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  private:

  bool use_mahalanobis_;

  bool numerical_cov_;



  // Eigen::aligned_unordered_map<KeypointId, Eigen::aligned_vector<PatchT>>
  //     patches;

  std::shared_ptr<std::vector<basalt::ManagedImagePyr<u_int16_t>>> pyramid;

};

} // namespace basalt
