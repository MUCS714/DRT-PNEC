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

#ifndef FRAMES_TRACKING_FRAME_H_
#define FRAMES_TRACKING_FRAME_H_

#include <exception>
#include <iostream>
#include <stdio.h>

#include "opencv2/opencv.hpp"

#include "base_frame.h"
#include "camera.h"
#include "klt_patch_optical_flow.h"
#include "timing.h"
#include <basalt/optical_flow/optical_flow.h>

namespace pnec {
namespace frames {
class TrackingFrame : public BaseFrame {

public:
  using Ptr = std::shared_ptr<TrackingFrame>;
  TrackingFrame(int64_t id, double timestamp, basalt::KLTPatchOpticalFlow<float, basalt::Pattern52> &tracking,
                pnec::common::FrameTiming &frame_timing, const cv::Mat &image)
      : BaseFrame(id, timestamp), tracking_{tracking} {
    FindFeatures(frame_timing, image);    // 这里运行FindFeatures，对图像找出匹配点
  }

private:
  void FindFeatures(pnec::common::FrameTiming &frame_timing, const cv::Mat &image);
  //void FindFeatures2(const cv::Mat &image, const Eigen::Vector2d &offset);

  basalt::KLTPatchOpticalFlow<float, basalt::Pattern52> &tracking_;
};

} // namespace frames
} // namespace pnec

#endif // FRAMES_TRACKING_FRAME_H_
