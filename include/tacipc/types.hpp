#pragma once
#include <variant>
#include <Eigen/Core>
#include <nlohmann/json.hpp>

namespace eigen
{
    using matXXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 3>;
    using matX3d = Eigen::Matrix<double, Eigen::Dynamic, 3>;
    using matXXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, 3>;
    using matX2i = Eigen::Matrix<int, Eigen::Dynamic, 2>;
    using matX3i = Eigen::Matrix<int, Eigen::Dynamic, 3>;
    using matX4i = Eigen::Matrix<int, Eigen::Dynamic, 4>;
    using mat4d = Eigen::Matrix<double, 4, 4>;
    using vec3d = Eigen::Vector<double, 3>;
    using vecXd = Eigen::Vector<double, Eigen::Dynamic>;
    using vec2i = Eigen::Vector<int, 2>;
    using vec3i = Eigen::Vector<int, 3>;
    using vec4i = Eigen::Vector<int, 4>;
    using vecXi = Eigen::Vector<int, Eigen::Dynamic>; 

    template <class T>
    using matX3 = Eigen::Matrix<T, Eigen::Dynamic, 3>;
    template <class T>
    using mat4x3 = Eigen::Matrix<T, 4, 3>;
    template <class T>
    using vec3 = Eigen::Vector<T, 3>;
    template <class T>
    using vec12 = Eigen::Vector<T, 12>;
}; // namespace eigen

namespace tacipc
{
    using floating_point_t = std::variant<float, double, long double>;

    /// @brief enum class for body type, rigid or soft
    enum BodyType : int
    {
        Rigid = 0,
        Soft = 1
    };
}; // namespace tacipc

using json = nlohmann::json;