/**
 * @file optimization.hpp
 * @author ouyangzhanpeng
 * @brief 
 * @version 0.1
 * @date 2022-03-26
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP

#include <cmath>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Eigen>
#include <memory>


#include "IMU/imuPreintegrated.hpp"
#include "geometry.hpp"
#include "opengvMethod.hpp"

struct BiasSolverCostFunctor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BiasSolverCostFunctor(const std::vector<Eigen::Vector3d> &bearings1,
                          const std::vector<Eigen::Vector3d> &bearings2,
                          const std::vector<Eigen::Matrix3d> &b2_cov,
                          const Eigen::Quaterniond &qic,
                          const vio::IMUPreintegrated &integrate,
                          const vio::IMUPreintegrated &integrate2) : _qic(qic){


        jacobina_q_bg = integrate.JRg_;
        
        qjk_imu = integrate.dR_.unit_quaternion();
        Eigen::Vector3d Translation = integrate2.dP_;
        Eigen::Matrix3d Rotation = integrate2.dR_.matrix();
        Eigen::Quaterniond qcjk = _qic.inverse() * qjk_imu;
        
        //create F things
        for (int i = 0; i < bearings1.size(); i++) {
            
            Eigen::Vector3d f1 = bearings1[i].normalized();

            Eigen::Vector3d f2 = bearings2[i].normalized();
            
            f1 = qcjk.inverse() * f1;
            f2 = _qic * f2;
           
            Eigen::Matrix3d bv_1_hat = Skew(f1);
            Eigen::Vector3d transformed_translation =
            (Translation.transpose() * bv_1_hat * Rotation).transpose();

            double bv_weight = 1 / (transformed_translation.transpose() * b2_cov[i] *transformed_translation)(0, 0);
            bv_weight = bv_weight * 1.0e-8;
            f2 = f2 * std::sqrt(bv_weight);           
            
            Eigen::Matrix3d F = f2 * f2.transpose();

            double weight = 1.0;
            xxF_ = xxF_ + weight * f1[0] * f1[0] * F;
            yyF_ = yyF_ + weight * f1[1] * f1[1] * F;
            zzF_ = zzF_ + weight * f1[2] * f1[2] * F;
            xyF_ = xyF_ + weight * f1[0] * f1[1] * F;
            yzF_ = yzF_ + weight * f1[1] * f1[2] * F;
            // zxF_ = zxF_ + weight * f1[2] * f1[0] * F; //opengv
            xzF_ = xzF_ + weight * f1[0] * f1[2] * F;   //ROBA
        }
    }

    template<typename T>
    bool operator()(const T *const parameter1, T *residual) const {


        Eigen::Map<const Eigen::Matrix<T, 3, 1>> deltaBg(parameter1);
        
        Eigen::Matrix<T, 3, 1> jacobian_bg = jacobina_q_bg.cast<T>() * deltaBg;

        Eigen::Matrix<T, 4, 1> qij_tmp;
        

        ceres::AngleAxisToQuaternion(jacobian_bg.data(), qij_tmp.data());

        Eigen::Quaternion<T> qij(qij_tmp(0), qij_tmp(1), qij_tmp(2), qij_tmp(3));

        Eigen::Matrix<T, 3, 1> cayley = Quaternion2Cayley<T>(qij);

        Eigen::Matrix<T, 1, 3> jacobian;

        T EV = opengv::GetSmallestEVwithJacobian(
                xxF_, yyF_, zzF_, xyF_, yzF_, xzF_, cayley, jacobian);
        
        residual[0] = EV;      
        
        return true;
    }


    static ceres::CostFunction *
    Create(const std::vector<Eigen::Vector3d> &bearings1,
           const std::vector<Eigen::Vector3d> &bearings2,
           const std::vector<Eigen::Matrix3d> &b2_cov,
           const Eigen::Quaterniond &qic,
           const vio::IMUPreintegrated &integratePtr,
           const vio::IMUPreintegrated &integrate2) {
        return (new ceres::AutoDiffCostFunction<BiasSolverCostFunctor, 1, 3>(
                new BiasSolverCostFunctor(bearings1, bearings2, b2_cov, qic, integratePtr, integrate2)));
    }

private:

    Eigen::Matrix3d xxF_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d yyF_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d zzF_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d xyF_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d yzF_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d xzF_ = Eigen::Matrix3d::Zero();

    Eigen::Matrix3d jacobina_q_bg;
    Eigen::Quaterniond qjk_imu;
    Eigen::Quaterniond _qic;

};
#endif //OPTIMIZATION_HPP

