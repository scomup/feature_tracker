#ifndef VISUAL_ODOMETRY_COMMON_MATH_H_
#define VISUAL_ODOMETRY_COMMON_MATH_H_

#include <cmath>
#include <vector>

#include "Eigen/Core"
#include "ceres/ceres.h"

namespace feature_tracker{
namespace common {

// Clamps 'value' to be in the range ['min', 'max'].
template <typename T>
T Clamp(const T value, const T min, const T max) {
  if (value > max) {
    return max;
  }
  if (value < min) {
    return min;
  }
  return value;
}

// Calculates 'base'^'exponent'.
template <typename T>
constexpr T Power(T base, int exponent) {
  return (exponent != 0) ? base * Power(base, exponent - 1) : T(1);
}

// Calculates a^2.
template <typename T>
constexpr T Pow2(T a) {
  return Power(a, 2);
}

// Converts from degrees to radians.
constexpr double DegToRad(double deg) { return M_PI * deg / 180.; }

// Converts form radians to degrees.
constexpr double RadToDeg(double rad) { return 180. * rad / M_PI; }

// Bring the 'difference' between two angles into [-pi; pi].
template <typename T>
T NormalizeAngleDifference(T difference) {
  while (difference > M_PI) {
    difference -= T(2. * M_PI);
  }
  while (difference < -M_PI) {
    difference += T(2. * M_PI);
  }
  return difference;
}

template <typename T>
T atan2(const Eigen::Matrix<T, 2, 1>& vector) {
  return ceres::atan2(vector.y(), vector.x());
}

template <typename T>
inline void QuaternionProduct(const double* const z, const T* const w,
                              T* const zw) {
  zw[0] = z[0] * w[0] - z[1] * w[1] - z[2] * w[2] - z[3] * w[3];
  zw[1] = z[0] * w[1] + z[1] * w[0] + z[2] * w[3] - z[3] * w[2];
  zw[2] = z[0] * w[2] - z[1] * w[3] + z[2] * w[0] + z[3] * w[1];
  zw[3] = z[0] * w[3] + z[1] * w[2] - z[2] * w[1] + z[3] * w[0];
}

template <typename T>
inline Eigen::Matrix<T, 3, 1> Matrix3dtoEuler(const Eigen::Matrix<T, 3, 3> &R)
{
  const T sy = ceres::sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));

  const bool singular = sy < (1e-6); // If

  Eigen::Matrix<T, 3, 1> euler;
  if (!singular)
  {
    euler(0) = ceres::atan2(R(2, 1), R(2, 2));
    euler(1) = ceres::atan2(-R(2, 0), sy);
    euler(2) = ceres::atan2(R(1, 0), R(0, 0));
  }
  else
  {
    euler(0) = ceres::atan2(-R(1, 2), R(1, 1));
    euler(1) = ceres::atan2(-R(2, 0), sy);
    euler(2) = T(0);
  }
  return euler;
}

template <typename T>
inline Eigen::Matrix<T, 3, 3> EulertoMatrix3d(const Eigen::Matrix<T, 3, 1> &theta)
{
  const Eigen::Matrix<T, 3, 3> R_x = (Eigen::Matrix<T, 3, 3>() << T(1), T(0), T(0), T(0), cos(theta[0]), -sin(theta[0]), 0, sin(theta[0]), cos(theta[0])).finished();
  const Eigen::Matrix<T, 3, 3> R_y = (Eigen::Matrix<T, 3, 3>() << cos(theta[1]), 0, sin(theta[1]), 0, 1, 0, -sin(theta[1]), 0, cos(theta[1])).finished();
  const Eigen::Matrix<T, 3, 3> R_z = (Eigen::Matrix<T, 3, 3>() << cos(theta[2]), -sin(theta[2]), 0, sin(theta[2]), cos(theta[2]), 0, 0, 0, 1).finished();

  // Combined rotation matrix
  Eigen::Matrix<T, 3, 3> R = R_z * R_y * R_x;
  return R;
}

}  // namespace common
}  // namespace visual_odometry

#endif  // VISUAL_ODOMETRY_COMMON_MATH_H_
