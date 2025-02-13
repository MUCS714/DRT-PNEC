//
// Created by xubo on 23-8-30.
//
#include "initMethod/drtLooselyCoupled.h"

namespace DRT
{
    drtLooselyCoupled::drtLooselyCoupled(const Eigen::Matrix3d &Rbc, const Eigen::Vector3d &pbc)
    : drtVioInit(Rbc, pbc){}

    bool drtLooselyCoupled::process()
    {

        cout << "drt loosely process" << endl;

        ticToc t_solve;
        // solve gyroscope bias
        ticToc t_biasg;

        if(!gyroBiasEstimator())
            return false;


        double time_biasg = t_biasg.toc();

        ticToc t_ligt;
        // reintegrate the imu data with solved gyroscope bias
        vio::IMUBias solved_bias(biasg, biasa);

        // vio::IMUBias solved_bias(Eigen::Vector3d(0., 0., 0.), Eigen::Vector3d(0, 0, 0));

        // LOG(INFO) << "frame number: " << int_frameid2_time_frameid.size() << " imu number: " << imu_meas.size()
        //          << std::endl;

        for (int i = 0; i < imu_meas.size(); i++) {
            imu_meas[i].reintegrate(solved_bias);
        }

        // set first rotation to Identity
        // convert relative rotation of IMU to the global rotation of camera
        frame_rot[int_frameid2_time_frameid.at(0)] = Eigen::Matrix3d::Identity();

        Eigen::Matrix3d accumRot = Eigen::Matrix3d::Identity();

        for (int i = 1; i < int_frameid2_time_frameid.size(); i++) {
            Eigen::Matrix3d dRcicj = Rbc_.transpose() * imu_meas[i - 1].dR_.matrix() * Rbc_;
            // R_0_1 * R_1_2 * R_2_3....
            accumRot = accumRot * dRcicj;
            frame_rot[int_frameid2_time_frameid.at(i)] = accumRot;
        }

        // number frames
        int num_view_ = int_frameid2_time_frameid.size();
        // number points
        int num_pts_ = 0;

        for (const auto &pt: SFMConstruct) {
            if (pt.second.obs.size() < 3) continue;
            ++num_pts_;
        }

        // allocate memory for LTL matrix where Lt=0, expect for the reference view
        Eigen::MatrixXd LTL = Eigen::MatrixXd::Zero(num_view_ * 3 - 3, num_view_ * 3 - 3);

        // use d = A_lr * t > 0 to identify the correct sign of the translation result
        Eigen::MatrixXd A_lr = Eigen::MatrixXd::Zero(num_pts_, 3 * num_view_);

        // construct LTL and A_lr matrix from 3D points
        // std::cout << "build_LTL" << std::endl;
        build_LTL(LTL, A_lr);

        Eigen::VectorXd evectors = Eigen::VectorXd::Zero(3 * num_view_);

        //[Step.4 in Pose-only Algorithm]: obtain the translation solution by using SVD
        // std::cout << "solve_LTL" << std::endl;
        if (!solve_LTL(LTL, evectors)) {
            return false;
        }

        //[Step.5 in Pose-only Algorithm]: identify the right global translation solution
        identify_sign(A_lr, evectors);

        rotation.resize(int_frameid2_time_frameid.size());
        position.resize(int_frameid2_time_frameid.size());
        velocity.resize(int_frameid2_time_frameid.size());


        for (int i = 0; i < local_active_frames.size(); i++) {
            rotation[i] = frame_rot.at(int_frameid2_time_frameid.at(i)) * Rbc_.transpose();  //Rc0cj * Rcb = Rc0bj
            position[i] = evectors.middleRows<3>(3 * i);
        }
        double time_ligt = t_ligt.toc();

        ticToc t_velocity_gravity;
        if (linearAlignment()) {
            return true;
        } else {
            printf("solve g failed!\n");
            return false;
        }
    }

    Eigen::Matrix<double,3,1> drtLooselyCoupled::toVector3d(const cv::Mat &cvVector)
    {
        Eigen::Matrix<double,3,1> v;
        v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

        return v;
    }

    cv::Mat drtLooselyCoupled::toCvMat(const Eigen::Matrix<double,3,1> &m)
    {
        cv::Mat cvMat(3,1,CV_32F);
        for(int i=0;i<3;i++)
                cvMat.at<float>(i)=m(i);

        return cvMat.clone();
    }

    cv::Mat drtLooselyCoupled::toCvMat(const Eigen::Matrix3d &m)
    {
        cv::Mat cvMat(3,3,CV_32F);
        for(int i=0;i<3;i++)
            for(int j=0; j<3; j++)
                cvMat.at<float>(i,j)=m(i,j);

        return cvMat.clone();
    }

    cv::Mat drtLooselyCoupled::SkewSymmetricMatrix(const cv::Mat &v)
    {
        return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
                v.at<float>(2),               0,-v.at<float>(0),
                -v.at<float>(1),  v.at<float>(0),              0);
    }

    bool drtLooselyCoupled::linearAlignment() {
        int all_frame_count = int_frameid2_time_frameid.size();
        int n_state = all_frame_count * 3 + 3 + 1;

        MatrixXd A{n_state, n_state};
        A.setZero();
        VectorXd b{n_state};
        b.setZero();
        double Q = 0.;

        for (int i = 0; i < int_frameid2_time_frameid.size() - 1; i++) {
            int j = i + 1;
            MatrixXd tmp_A(6, 10);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = imu_meas[i].sum_dt_;

            CHECK(imu_meas[i].start_t_ns == int_frameid2_time_frameid.at(i)) << "imu meas error";
            CHECK(imu_meas[i].end_t_ns == int_frameid2_time_frameid.at(j)) << "imu meas error";

            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 1>(0, 6) = rotation[i].transpose() * (position[j] - position[i]) / 100.0;

            tmp_A.block<3, 3>(0, 7) = rotation[i].transpose() * dt * dt / 2 * Matrix3d::Identity() * G.norm();

            tmp_b.block<3, 1>(0, 0) = imu_meas[i].dP_ + rotation[i].transpose() * rotation[j] * pbc_ - pbc_;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = rotation[i].transpose() * rotation[j];

            tmp_A.block<3, 3>(3, 7) = rotation[i].transpose() * dt * Matrix3d::Identity() * G.norm();

            tmp_b.block<3, 1>(3, 0) = imu_meas[i].dV_;

            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();

            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
            b.tail<4>() += r_b.tail<4>();

            A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
            A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();

            Q += tmp_b.transpose() * cov_inv * tmp_b;
        }

        Eigen::VectorXd x;
        double s;
        Eigen::MatrixXd M_k2TM_k2 = A.bottomRightCorner<3, 3>();


        double mean_value = (M_k2TM_k2(0, 0) + M_k2TM_k2(1, 1) + M_k2TM_k2(2, 2)) / 3.0;

        double scale = 1 / mean_value;
        A = A * scale;
        b = b * scale;
        Q = Q * scale;

        if (!gravityRefine(A, -2. * b, Q, 1, x))
        {
            return false;
        }

        gravity = x.tail(3) * G.norm();
        s = x(n_state - 4) / 100.0;
        x(n_state - 4) = s;

        // new
        cv::Mat gI = cv::Mat::zeros(3,1,CV_32F);
        gI.at<float>(2) = 1;
        // Normalized approx. gravity vecotr in world frame
        cv::Mat gwstar = toCvMat(gravity);
        cv::Mat gwn = gwstar/cv::norm(gwstar);
        // Debug log
        //cout<<"gw normalized: "<<gwn<<endl;

        // vhat = (gI x gw) / |gI x gw|
        cv::Mat gIxgwn = gI.cross(gwn);
        double normgIxgwn = cv::norm(gIxgwn);
        cv::Mat vhat = gIxgwn/normgIxgwn;
        double theta = std::atan2(normgIxgwn,gI.dot(gwn));
        // Debug log
        //cout<<"vhat: "<<vhat<<", theta: "<<theta*180.0/M_PI<<endl;

        Eigen::Vector3d vhateig = toVector3d(vhat);
        Eigen::Matrix3d RWIeig = Sophus::SO3::exp(vhateig*theta).matrix();
        cv::Mat Rwi = toCvMat(RWIeig);
        cv::Mat GI = gI*toCvMat(G);//9.8012;
        // Solve C*x=D for x=[s,dthetaxy] (1+2)x1 vector
        cv::Mat C = cv::Mat::zeros(3*(N-2),3,CV_32F);
        cv::Mat D = cv::Mat::zeros(3*(N-2),1,CV_32F);

        for(int i=0; i<int_frameid2_time_frameid.size()-2; i++)
        {
            int j = i + 1;
            int k = i + 2;
            // Delta time between frames
            double dt12 = imu_meas[i].sum_dt_;
            double dt23 = imu_meas[j].sum_dt_;
            // Pre-integrated measurements
            cv::Mat dp12 = toCvMat(imu_meas[i].dP_);
            cv::Mat dv12 = toCvMat(imu_meas[i].dV_);
            cv::Mat dp23 = toCvMat(imu_meas[j].dP_);
            // Position of camera center
            cv::Mat pc1 = toCvMat(position[i]);
            cv::Mat pc2 = toCvMat(position[j]);
            cv::Mat pc3 = toCvMat(position[k]);
            // Rotation of camera, Rwc
            cv::Mat Rc1 = toCvMat(rotation[i]);
            cv::Mat Rc2 = toCvMat(rotation[j]);
            cv::Mat Rc3 = toCvMat(rotation[k]);
            // Stack to C/D matrix
            // lambda*s + phi*dthetaxy + zeta*ba = psi
            cv::Mat lambda = (pc2-pc1)*dt23 + (pc2-pc3)*dt12;
            cv::Mat phi = 0.5*(dt12*dt12*dt23 + dt12*dt23*dt23)*Rwi*SkewSymmetricMatrix(GI);  // note: this has a '-', different to paper
            cv::Mat psi = (Rc2-Rc1)*pbc_*dt23 + Rc1*dp12*dt23 - (Rc3-Rc2)*pbc_*dt12
                        - Rc2*dp23*dt12 - Rc1*dv12*dt23*dt12 - 0.5*Rwi*GI*(dt12*dt12*dt23 + dt12*dt23*dt23); // note:  - paper
            lambda.copyTo(C.rowRange(3*i+0,3*i+3).col(0));
            phi.colRange(0,2).copyTo(C.rowRange(3*i+0,3*i+3).colRange(1,3)); //only the first 2 columns, third term in dtheta is zero, here compute dthetaxy 2x1.
            psi.copyTo(D.rowRange(3*i+0,3*i+3));

            // Debug log
            //cout<<"iter "<<i<<endl;
        }

        // Use svd to compute C*x=D, x=[s,dthetaxy,ba] 6x1 vector
        // C = u*w*vt, u*w*vt*x=D
        // Then x = vt'*winv*u'*D
        cv::Mat w2,u2,vt2;
        // Note w2 is 3x1 vector by SVDecomp()
        // C is changed in SVDecomp() with cv::SVD::MODIFY_A for speed
        cv::SVDecomp(C,w2,u2,vt2,cv::SVD::MODIFY_A);
        // Debug log
        //cout<<"u2:"<<endl<<u2<<endl;
        //cout<<"vt2:"<<endl<<vt2<<endl;
        //cout<<"w2:"<<endl<<w2<<endl;

        // Compute winv
        cv::Mat w2inv=cv::Mat::eye(3,3,CV_32F);
        for(int i=0;i<3;i++)
        {
            if(fabs(w2.at<float>(i))<1e-10)
            {
                w2.at<float>(i) += 1e-10;
                // Test log
                cerr<<"w2(i) < 1e-10, w="<<endl<<w2<<endl;
            }

            w2inv.at<float>(i,i) = 1./w2.at<float>(i);
        }
        // Then y = vt'*winv*u'*D
        cv::Mat y = vt2.t()*w2inv*u2.t()*D;

        double s_ = y.at<float>(0);
        cv::Mat dthetaxy = y.rowRange(1,3);

        // dtheta = [dx;dy;0]
        cv::Mat dtheta = cv::Mat::zeros(3,1,CV_32F);
        dthetaxy.copyTo(dtheta.rowRange(0,2));
        Eigen::Vector3d dthetaeig = toVector3d(dtheta);
        // Rwi_ = Rwi*exp(dtheta)
        Eigen::Matrix3d Rwieig_ = RWIeig*Sophus::SO3::exp(dthetaeig).matrix();
        gravity = Rwieig_* toVector3d(GI);
        s = s_;

        for (int i = int_frameid2_time_frameid.size() - 1; i >= 0; i--) {
            position[i] = s * position[i] - rotation[i] * pbc_;
            velocity[i] = rotation[i] * x.segment<3>(i * 3);
        }
        
        Eigen::Matrix3d rot0 = rotation[0].transpose();  // Rb0c0
        gravity = rot0 * gravity;
        for (int i = 0; i < int_frameid2_time_frameid.size(); i++) {
            rotation[i] = rot0 * rotation[i];   // Rb0c0 * Rc0bi
            position[i] = rot0 * position[i] + pbc_;
            velocity[i] = rot0 * velocity[i];
        }

        cout << "refine: " << gravity.norm() << " " << G.norm() << endl;
        return true;
    }




    void drtLooselyCoupled::build_LTL(Eigen::MatrixXd &LTL, Eigen::MatrixXd &A_lr) {
        // #pragma omp parallel for shared(A_lr, LTL)

        int num_view_ = int_frameid2_time_frameid.size();
        int track_id = 0;
        for (const auto &pt: SFMConstruct) {

            const auto &obs = pt.second.obs;

            // the number of obs must greater than 3
            if (obs.size() < 3) continue;

            TimeFrameId lbase_view_id = 0;
            TimeFrameId rbase_view_id = 0;

            select_base_views(obs,
                              lbase_view_id,
                              rbase_view_id);


            //原本中的L矩阵，[B, C, D, ...]
            Eigen::MatrixXd tmp_LiGT_vec = Eigen::MatrixXd::Zero(3, num_view_ * 3);
            // [Step.3 in Pose-only algorithm]: calculate local L matrix,

            for (const auto &frame: obs) {
                // the current view id
                TimeFrameId i_view_id = frame.first;
                //使用X_i和X_l，可以计算t_il，引入r view计算l view的深度，对应公式(11)
                if (i_view_id != lbase_view_id) {
                    //公式(21)
                    Eigen::Matrix3d xi_cross = cross_product_matrix(frame.second.normalpoint);

                    Eigen::Matrix3d R_cicl =
                            frame_rot.at(i_view_id).transpose() * frame_rot.at(lbase_view_id);

                    Eigen::Matrix3d R_crcl =
                            frame_rot.at(rbase_view_id).transpose() * frame_rot.at(lbase_view_id);

                    Eigen::Vector3d a_lr_tmp_t =
                            cross_product_matrix(R_crcl * obs.at(lbase_view_id).normalpoint) *
                            obs.at(rbase_view_id).normalpoint;

                    Eigen::RowVector3d a_lr_t =
                            a_lr_tmp_t.transpose() * cross_product_matrix(obs.at(rbase_view_id).normalpoint);

                    // combine all a_lr vectors into a matrix form A, i.e., At > 0
                    // a_lr * trl = 0 -> alr * Rrw *(twl - twr) = 0 公式写trl对应原文tlr, 为了把相对的t转为全局的，所以乘了个R
                    A_lr.row(track_id).block<1, 3>(0, time_frameid2_int_frameid.at(lbase_view_id) * 3) =
                            a_lr_t * frame_rot.at(rbase_view_id).transpose();
                    A_lr.row(track_id).block<1, 3>(0, time_frameid2_int_frameid.at(rbase_view_id) * 3) =
                            -a_lr_t * frame_rot.at(rbase_view_id).transpose();
                    // theta_lr
                    Eigen::Vector3d theta_lr_vector = cross_product_matrix(obs.at(rbase_view_id).normalpoint)
                                                      * R_crcl
                                                      * obs.at(lbase_view_id).normalpoint;

                    double theta_lr = theta_lr_vector.squaredNorm();

                    // calculate matrix B [rbase_view_id]
                    // 对应公(18) 也能看出来变量里把transpose省略掉了，B对应right view的全局t
                    Eigen::Matrix3d Coefficient_B =
                            xi_cross * R_cicl * obs.at(lbase_view_id).normalpoint * a_lr_t *
                            frame_rot.at(rbase_view_id).transpose();

                    // calculate matrix C [i_view_id]
                    Eigen::Matrix3d Coefficient_C = theta_lr * cross_product_matrix(obs.at(i_view_id).normalpoint) *
                                                    frame_rot.at(i_view_id).transpose();

                    // calculate matrix D [lbase_view_id]
                    Eigen::Matrix3d Coefficient_D = -(Coefficient_B + Coefficient_C);
                    // calculate temp matrix L for a single 3D matrix
                    tmp_LiGT_vec.setZero();
                    tmp_LiGT_vec.block<3, 3>(0, time_frameid2_int_frameid.at(rbase_view_id) * 3) += Coefficient_B;
                    tmp_LiGT_vec.block<3, 3>(0, time_frameid2_int_frameid.at(i_view_id) * 3) += Coefficient_C;
                    tmp_LiGT_vec.block<3, 3>(0, time_frameid2_int_frameid.at(lbase_view_id) * 3) += Coefficient_D;


                    // calculate LtL submodule
                    Eigen::MatrixXd LTL_l_row = Coefficient_D.transpose() * tmp_LiGT_vec;
                    Eigen::MatrixXd LTL_r_row = Coefficient_B.transpose() * tmp_LiGT_vec;
                    Eigen::MatrixXd LTL_i_row = Coefficient_C.transpose() * tmp_LiGT_vec;

                    // assignment for LtL (except for the reference view id)
                    // #pragma omp critical
                    {
                        if (time_frameid2_int_frameid.at(lbase_view_id) > 0)
                            LTL.middleRows<3>(
                                    time_frameid2_int_frameid.at(lbase_view_id) * 3 - 3) += LTL_l_row.rightCols(
                                    LTL_l_row.cols() - 3);

                        if (time_frameid2_int_frameid.at(rbase_view_id) > 0)
                            LTL.middleRows<3>(
                                    time_frameid2_int_frameid.at(rbase_view_id) * 3 - 3) += LTL_r_row.rightCols(
                                    LTL_r_row.cols() - 3);

                        if (time_frameid2_int_frameid.at(i_view_id) > 0)
                            LTL.middleRows<3>(time_frameid2_int_frameid.at(i_view_id) * 3 - 3) += LTL_i_row.rightCols(
                                    LTL_i_row.cols() - 3);
                    }
                }
            }

            ++track_id;
        }
    }

    bool drtLooselyCoupled::solve_LTL(const Eigen::MatrixXd &LTL, Eigen::VectorXd &evectors) {
        // ========================= Solve Problem by Eigen's SVD =======================
        // std::cout << "Solve Problem by Eigen's SVD" << std::endl;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(LTL, Eigen::ComputeFullU | Eigen::ComputeFullV);
        //  if (svd.info() != Eigen::Success)
        //    OPENMVG_LOG_ERROR << "SVD solver failure - expect to have invalid output";
        Eigen::MatrixXd V = svd.matrixV();
        evectors.bottomRows(V.rows()) = V.col(V.cols() - 1);
        return true;

    }

    void drtLooselyCoupled::identify_sign(const Eigen::MatrixXd &A_lr, Eigen::VectorXd &evectors) {
        const Eigen::VectorXd judgeValue = A_lr * evectors;
        const int positive_count = (judgeValue.array() > 0.0).cast<int>().sum();
        const int negative_count = judgeValue.rows() - positive_count;
        if (positive_count < negative_count) {
            evectors = -evectors;
        }
        // std::cout << "positive_count: " << positive_count << "  negative_count: " << negative_count << std::endl;
    }


    void drtLooselyCoupled::select_base_views(const Eigen::aligned_map<TimeFrameId, FeaturePerFrame> &track,
                                              TimeFrameId &lbase_view_id,
                                              TimeFrameId &rbase_view_id) {
        double best_criterion_value = -1.;
        std::vector<int> track_id;
        // track_id.reserve(track.size());

        for (const auto &frame: track) {
            int id = time_frameid2_int_frameid.at(frame.first);
            track_id.push_back(id);
        }

        size_t track_size = track_id.size(); //num_pts_

        // [Step.2 in Pose-only Algorithm]: select the left/right-base views
        for (int i = 0; i < track_size - 1; ++i) {
            for (int j = i + 1; j < track_size; ++j) {

                const TimeFrameId &i_view_id = int_frameid2_time_frameid.at(track_id[i]);
                const TimeFrameId &j_view_id = int_frameid2_time_frameid.at(track_id[j]);

                const Eigen::Vector3d &i_coord = track.at(i_view_id).normalpoint;
                const Eigen::Vector3d &j_coord = track.at(j_view_id).normalpoint;

                // R_i is world to camera i
                const Eigen::Matrix3d &R_i = frame_rot.at(i_view_id);
                const Eigen::Matrix3d &R_j = frame_rot.at(j_view_id);
                // camera i to camera j
                // Rcjw *  Rwci
                const Eigen::Matrix3d R_ij = R_j.transpose() * R_i;
                const Eigen::Vector3d theta_ij = j_coord.cross(R_ij * i_coord);

                double criterion_value = theta_ij.norm();

                if (criterion_value > best_criterion_value) {

                    best_criterion_value = criterion_value;

                    if (i_view_id < j_view_id) {
                        lbase_view_id = i_view_id;
                        rbase_view_id = j_view_id;
                    } else {
                        lbase_view_id = j_view_id;
                        rbase_view_id = i_view_id;
                    }

                }
            }
        }
    }

}