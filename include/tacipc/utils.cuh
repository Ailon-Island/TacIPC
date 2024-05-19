#pragma once
#include <iostream>
#include <tacipc/macros.hpp>
#include <tacipc/meshio/mesh.hpp>
#include <tacipc/dynamicBuffer.cuh>
#include <tacipc/solver/Properties.cuh>
// #include <tacipc/solver/VecWrapper.cuh>
#include <tacipc/solver/TVWrapper.cuh>
#include <zensim/zpc_tpls/fmt/color.h>
#include <zensim/zpc_tpls/fmt/format.h>
#include <zensim/container/Bvh.hpp>
#include <zensim/container/Bvs.hpp>
#include <zensim/container/Bvtt.hpp>
#include <zensim/math/Vec.h>
#include <zensim/ZpcMathUtils.hpp>
#include <zensim/math/MathUtils.h>
#include <zensim/types/SmallVector.hpp>
#include <zensim/container/Vector.hpp>
#include <zensim/container/TileVector.hpp>
#include <zensim/cuda/execution/ExecutionPolicy.cuh>
#include <zensim/execution/Atomics.hpp>

namespace tacipc
{
struct Logger
{
    FILE *file;
    Logger(const char *filename) : file(fopen(filename, "w")) {}
    Logger(const std::string &filename) : Logger(filename.c_str()) {}
    ~Logger() { fclose(file); }
    template <typename... Args>
    void log(Args... args)
    {
        fmt::print(file, args...);
    }
};

#define tacipc_assert_default(expr) tacipc_assert(expr, "default")
#define tacipc_assert(expr, msg)                                                 \
    {                                                                          \
        if (!(expr))                                                             \
        {                                                                      \
            printf(                                                            \
                "[tacipc]\tassertion failed!\tmsg[%s],\tline(%d),\tfile(%s)\n",  \
                msg, __LINE__, __FILE__);                                      \
        }                                                                      \
    }
// TODO: zs::SmallString::operator+ seems buggy
// #define tacipc_assert_args(expr, msg, ...)                                       \
//     {                                                                          \
//         if (!expr)                                                             \
//         {                                                                      \
//             zs::SmallString s0{"[tacipc]\tassertion failed!\tmsg["};             \
//             zs::SmallString s1{msg};                                           \
//             zs::SmallString s2{"],\tline(%d),\tfile(%s)\n"};                   \
//             auto str = s0 + s1 + s2;                                           \
//             printf(str.asChars(), __VA_ARGS__, __LINE__, __FILE__);            \
//         }                                                                      \
//     }


// stl utils
template <bool Move = false, class T>
constexpr void concat(std::vector<T> &dst, std::vector<T> &src) {
    if constexpr (Move)
        dst.insert(std::end(dst), std::make_move_iterator(std::begin(src)), std::make_move_iterator(std::end(src)));
    else
        dst.insert(std::end(dst), std::begin(src), std::end(src));
}

// warp utils
constexpr std::size_t count_warps(std::size_t n) noexcept {
    return (n + 31) / 32;
}
constexpr int warp_index(int n) noexcept {
    return n / 32;
}
constexpr auto warp_mask(int i, int n) noexcept {
    int k = n % 32;
    const int tail = n - k;
    if (i < tail)
        return zs::make_tuple(0xFFFFFFFFu, 32);
    return zs::make_tuple(((unsigned)(1ull << k) - 1), k);
}

template <typename T> __forceinline__ __device__ void reduce_to(int i, int n, T val, T &dst) {
    auto [mask, numValid] = warp_mask(i, n);
    __syncwarp(mask);
    auto locid = threadIdx.x & 31;
    for (int stride = 1; stride < 32; stride <<= 1) {
        auto tmp = __shfl_down_sync(mask, val, stride);
        if (locid + stride < numValid)
            val += tmp;
    }
    if (locid == 0)
        zs::atomic_add(zs::exec_cuda, &dst, val);
}

// reduce
template <typename T, typename Op = std::plus<T>>
inline T reduce(zs::CudaExecutionPolicy &cudaPol, const zs::Vector<T> &res, Op op = {}) {
    using namespace zs;
    Vector<T> ret{res.get_allocator(), 1};
    zs::reduce(cudaPol, std::begin(res), std::end(res), std::begin(ret), (T)0, op);
    return ret.getVal();
}

template <typename T, typename Op = std::plus<T>>
inline T reduce(zs::OmpExecutionPolicy &ompPol, const zs::Vector<T> &res, Op op = {}) {
    using namespace zs;
    Vector<T> ret{res.get_allocator(), 1};
    zs::reduce(ompPol, std::begin(res), std::end(res), std::begin(ret), (T)0, op);
    return ret.getVal();
}

template <typename PropsT, typename TagT0, typename TagT1>
inline typename PropsT::value_t dot(zs::CudaExecutionPolicy &cudaPol, TVWrapper<PropsT> &vertData, const TagT0 &tag0,
             const TagT1 &tag1) {
    using namespace zs;
    using T = typename PropsT::value_t;
    constexpr auto space = execspace_e::cuda;
    // Vector<double> res{vertData.get_allocator(), vertData.size()};
    Vector<T> res{vertData.get_allocator(), count_warps(vertData.size())};
    zs::memset(zs::mem_device, res.data(), 0, sizeof(T) * count_warps(vertData.size()));
    cudaPol(range(vertData.size()), [data = view<space>(vertData), res = view<space>(res), tag0, tag1,
                                     n = vertData.size()] __device__(int pi) mutable {
        auto v0 = data.pack(tag0, pi);
        auto v1 = data.pack(tag1, pi);
        auto v = v0.dot(v1);
        // res[pi] = v;
        reduce_to(pi, n, v, res[pi / 32]);
    });
    return reduce(cudaPol, res, thrust::plus<T>{});
}
template <typename PropsT, typename TagT>
inline typename PropsT::value_t infNorm(zs::CudaExecutionPolicy &cudaPol, TVWrapper<PropsT> &vertData,
                                     const TagT tag = PropsT::dir) {
    using namespace zs;
    using T = typename PropsT::value_t;
    constexpr auto space = execspace_e::cuda;
    constexpr int codim = tag.numChannels;
    Vector<T> res{vertData.get_allocator(), count_warps(vertData.size())};
    zs::memset(zs::mem_device, res.data(), 0, sizeof(T) * count_warps(vertData.size()));
    cudaPol(range(vertData.size()), [data = view<space>(vertData), res = view<space>(res), tag,
                                     n = vertData.size()] __device__(int pi) mutable {
        auto v = data.pack(tag, pi);
        auto val = v.abs().max();

        auto [mask, numValid] = warp_mask(pi, n);
        auto locid = threadIdx.x & 31;
        for (int stride = 1; stride < 32; stride <<= 1) {
            auto tmp = __shfl_down_sync(mask, val, stride);
            if (locid + stride < numValid)
                val = zs::max(val, tmp);
        }
        if (locid == 0)
            res[pi / 32] = val;
    });
    return reduce(cudaPol, res, thrust::maximum<T>{});
}

// constitutive model utils
template <class VecTA, class VecTB, class VViewT>
constexpr auto deformation_gradient(const VViewT &vData, const VecTA &IB, const VecTB &inds) {
    using VProps = typename VViewT::props_t;
    using value_t = typename VProps::value_t;
    constexpr int codim = VecTA::template range_t<0>::value;
    constexpr int dim = VProps::xn.numChannels;
    using vec_t = zs::vec<value_t, dim>;
    using mat_t = zs::vec<value_t, dim, codim>;
    vec_t xs[codim+1]{};
    for (int ei = 0; ei < codim+1; ++ei)
        xs[ei] = vData.pack(VProps::xn, inds[ei]);
    vec_t x_x0[codim]{};
    for (int ei = 0; ei < codim; ++ei)
        x_x0[ei] = xs[ei+1] - xs[0];
    mat_t Ds{};
    for (int di = 0; di < dim; ++di)
        for (int ei = 0; ei < codim; ++ei)
            Ds(di, ei) = x_x0[ei](di);
                //   {x1x0[0], x2x0[0], x3x0[0], 
                //    x1x0[1], x2x0[1], x3x0[1], 
                //    x1x0[2], x2x0[2], x3x0[2]};
    mat_t F = Ds * IB;
    return F;
}

// transform utils
template <class T = double, class VecT>
constexpr inline auto to_point4(const VecT& v)
{
    return zs::vec<T, 4>{v[0], v[1], v[2], 1.}; 
}

template <class T = double, class VecT>
constexpr inline auto to_vec4(const VecT& v)
{
    return zs::vec<T, 4>{v[0], v[1], v[2], 0.}; 
}

template <class T = double, class VecT>
constexpr inline auto to_vec3(const VecT& v)
{
    return zs::vec<T, 3>{v[0], v[1], v[2]}; 
}

template <class VecT>
constexpr inline auto get_mat3(const VecT& mat)
{
    zs::vec<typename VecT::value_type, 3, 3> ret; 
    for (int di = 0; di < 3; di++)
        for (int dj = 0; dj < 3; dj++)
            ret(di, dj) = mat(di, dj); 
    return ret; 
}

template <class VecA, class VecB>
constexpr inline auto getRestPoint(VecA point, VecB q)
{
    using vec3 = zs::vec<typename VecA::value_type, 3>;
    using mat3 = zs::vec<typename VecA::value_type, 3, 3>;

    mat3 A; 
    for (int di = 0; di < 3; di++)
        for (int dj = 0; dj < 3; dj++)
            A(di, dj) = q(3 + di * 3 + dj); 
    return zs::inverse(A) * (point - vec3{q(0), q(1), q(2)}); 
}

template <class T>
auto getOrthoVec(zs::vec<T, 3> v, double eps_c = 1e-6)
{
    using vec3 = zs::vec<T, 3>;

    v = v.normalized(); 
    auto ret = v.cross(vec3{1., 0., 0.}); 
    if (ret.l2NormSqr() > eps_c)
        return ret.normalized(); 
    return v.cross(vec3{0., 1., 0.}).normalized(); 
}

enum euler_angle_convention_e { roe = 0, ypr };
constexpr auto roe_c = zs::wrapv<euler_angle_convention_e::roe>{};
constexpr auto ypr_c = zs::wrapv<euler_angle_convention_e::ypr>{};

enum angle_unit_e { radian = 0, degree };
constexpr auto radian_c = zs::wrapv<angle_unit_e::radian>{};
constexpr auto degree_c = zs::wrapv<angle_unit_e::degree>{};

template <typename T = float, int dim_ = 3> 
struct Rotation : zs::vec<T, dim_, dim_> {
    // 3d rotation can be viewed as a series of three successive rotations about coordinate axes
    using value_type = T;
    static constexpr int dim = dim_;  // do not let TM's dim fool you!
    using TV = zs::vec<value_type, dim>;
    using TM = zs::vec<value_type, dim, dim>;

    constexpr auto &self() noexcept { return static_cast<TM &>(*this); }
    constexpr const auto &self() const noexcept { return static_cast<const TM &>(*this); }

    constexpr Rotation() noexcept : TM{TM::identity()} {}

    constexpr Rotation(const TM &m) noexcept : TM{m} {}

    constexpr Rotation &operator=(const TM &o) noexcept {
      this->self() = o.self();
      return *this;
    }

    template <auto d = dim, zs::enable_if_t<d == 2> = 0> constexpr Rotation(value_type theta) noexcept
        : TM{TM::identity()} {
      value_type sinTheta = zs::sin(theta);
      value_type cosTheta = zs::cos(theta);
      (*this)(0, 0) = cosTheta;
      (*this)(0, 1) = sinTheta;
      (*this)(1, 0) = -sinTheta;
      (*this)(1, 1) = cosTheta;
    }
    /// axis + rotation
    template <typename VecT, auto unit = angle_unit_e::radian, auto d = dim,
              zs::enable_if_all<d == 3, std::is_convertible_v<typename VecT::value_type, T>,
                            VecT::dim == 1, VecT::template range_t<0>::value == 3> = 0>
    constexpr Rotation(const zs::VecInterface<VecT> &p_, value_type alpha, zs::wrapv<unit> = {}) noexcept
        : TM{} {
      if constexpr (unit == angle_unit_e::degree) alpha *= ((value_type)zs::g_pi / (value_type)180);
      auto p = p_.normalized();
      TM P{0, -p(2), p(1), p(2), 0, -p(0), -p(1), p(0), 0};
      value_type sinAlpha = zs::sin(alpha);
      value_type cosAlpha = zs::cos(alpha);
      self() = cosAlpha * TM::identity() + (1 - cosAlpha) * dyadic_prod(p, p) - sinAlpha * P;
    }
    // template <auto unit = angle_unit_e::radian, auto d = dim, zs::enable_if_t<d == 3> = 0>
    // constexpr auto extractAxisRotation(zs::wrapv<unit> = {}) const noexcept {
    //   const auto cosAlpha = (value_type)0.5 * (trace(self()) - 1);
    //   value_type alpha = zs::acos(cosAlpha);
    //   if (zs::math::near_zero(cosAlpha - 1)) return zs::make_tuple(TV{0, 1, 0}, (value_type)0);

    //   TV p{};
    //   if (zs::math::near_zero(cosAlpha + 1)) {
    //     p(0) = zs::math::sqrtNewtonRaphson(((*this)(0, 0) + 1) * (value_type)0.5);
    //     if (zs::math::near_zero(p(0))) {
    //       p(1) = zs::math::sqrtNewtonRaphson(((*this)(1, 1) + 1) * (value_type)0.5);
    //       p(2) = zs::math::sqrtNewtonRaphson(((*this)(2, 2) + 1) * (value_type)0.5);
    //       if ((*this)(1, 2) < (value_type)0) p(2) = -p(2);
    //     } else {
    //       p(1) = (*this)(0, 1) * (value_type)0.5 / p(0);
    //       p(2) = (*this)(0, 2) * (value_type)0.5 / p(0);
    //     }
    //   } else {
    //     const auto sinAlpha = zs::math::sqrtNewtonRaphson((value_type)1 - cosAlpha * cosAlpha);
    //     p(0) = ((*this)(2, 1) - (*this)(1, 2)) * (value_type)0.5 / sinAlpha;
    //     p(1) = ((*this)(0, 2) - (*this)(2, 0)) * (value_type)0.5 / sinAlpha;
    //     p(2) = ((*this)(1, 0) - (*this)(0, 1)) * (value_type)0.5 / sinAlpha;
    //   }
    //   if constexpr (unit == angle_unit_e::radian)
    //     return zs::make_tuple(p, alpha);
    //   else if constexpr (unit == angle_unit_e::degree)
    //     return zs::make_tuple(p, alpha * (value_type)180 / (value_type)zs::g_pi);
    // }

    /// euler angles
    template <auto unit = angle_unit_e::radian, auto convention = euler_angle_convention_e::roe,
              auto d = dim, zs::enable_if_t<d == 3> = 0>
    constexpr Rotation(value_type psi, value_type theta, value_type phi, zs::wrapv<unit> = {},
                       zs::wrapv<convention> = {}) noexcept
        : TM{} {
      if constexpr (unit == angle_unit_e::degree) {
        psi *= ((value_type)zs::g_pi / (value_type)180);
        theta *= ((value_type)zs::g_pi / (value_type)180);
        phi *= ((value_type)zs::g_pi / (value_type)180);
      }
      auto sinPsi = zs::sin(psi);
      auto cosPsi = zs::cos(psi);
      auto sinTheta = zs::sin(theta);
      auto cosTheta = zs::cos(theta);
      auto sinPhi = zs::sin(phi);
      auto cosPhi = zs::cos(phi);
      if constexpr (convention == euler_angle_convention_e::roe) {
        // Roe convention (successive rotations)
        // ref: https://www.continuummechanics.org/rotationmatrix.html
        // [z] psi -> [y'] theta -> [z'] phi
        auto cosPsi_cosTheta = cosPsi * cosTheta;
        auto sinPsi_cosTheta = sinPsi * cosTheta;
        (*this)(0, 0) = cosPsi_cosTheta * cosPhi - sinPsi * sinPhi;
        (*this)(0, 1) = sinPsi_cosTheta * cosPhi + cosPsi * sinPhi;
        (*this)(0, 2) = -sinTheta * cosPhi;
        (*this)(1, 0) = -cosPsi_cosTheta * sinPhi - sinPsi * cosPhi;
        (*this)(1, 1) = -sinPsi_cosTheta * sinPhi + cosPsi * cosPhi;
        (*this)(1, 2) = sinTheta * sinPhi;
        (*this)(2, 0) = cosPsi * sinTheta;
        (*this)(2, 1) = sinPsi * sinTheta;
        (*this)(2, 2) = cosTheta;
      } else if constexpr (convention == euler_angle_convention_e::ypr) {
        // navigation (yaw, pitch, roll)
        // axis [x, y, z] = direction [north, east, down] = body [front, right, bottom]
        // ref:
        // http://personal.maths.surrey.ac.uk/T.Bridges/SLOSH/3-2-1-Eulerangles.pdf
        // [z] psi -> [y'] theta -> [x'] phi
        auto sinPhi_sinTheta = sinPhi * sinTheta;
        auto cosPhi_sinTheta = cosPhi * sinTheta;
        (*this)(0, 0) = cosTheta * cosPsi;
        (*this)(0, 1) = sinPhi_sinTheta * cosPsi - cosPhi * sinPsi;
        (*this)(0, 2) = cosPhi_sinTheta * cosPsi + sinPhi * sinPsi;
        (*this)(1, 0) = cosTheta * sinPsi;
        (*this)(1, 1) = sinPhi_sinTheta * sinPsi + cosPhi * cosPsi;
        (*this)(1, 2) = cosPhi_sinTheta * sinPsi - sinPhi * cosPsi;
        (*this)(2, 0) = -sinTheta;
        (*this)(2, 1) = sinPhi * cosTheta;
        (*this)(2, 2) = cosPhi * cosTheta;
      }
    }
    // // ref: http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
    // template <auto unit = angle_unit_e::radian, auto convention = euler_angle_convention_e::roe,
    //           auto d = dim, zs::enable_if_t<d == 3> = 0>
    // constexpr auto extractAngles(zs::wrapv<unit> = {}, zs::wrapv<convention> = {}) const noexcept {
    //   value_type psi{}, theta{}, phi{};
    //   if constexpr (convention == euler_angle_convention_e::roe) {
    //     const auto cosTheta = (*this)(2, 2);
    //     if (zs::math::near_zero(cosTheta - 1)) {
    //       theta = 0;
    //       psi = zs::atan2((*this)(1, 0), (*this)(0, 0));
    //       phi = (value_type)0;
    //     } else if (zs::math::near_zero(cosTheta + 1)) {
    //       theta = zs::g_pi;
    //       psi = zs::atan2(-(*this)(1, 0), -(*this)(0, 0));
    //       phi = (value_type)0;
    //     } else {
    //       theta = zs::acos(cosTheta);  // another solution (-theta)
    //       /// theta [0, zs::g_pi], thus (sinTheta > 0) always holds true
    //       psi = zs::atan2((*this)(1, 2), (*this)(0, 2));  // no need to divide sinTheta
    //       phi = zs::atan2((*this)(2, 1), -(*this)(2, 0));
    //     }
    //   } else if constexpr (convention == euler_angle_convention_e::ypr) {
    //     const auto sinTheta = -(*this)(0, 2);
    //     if (zs::math::near_zero(sinTheta - 1)) {
    //       theta = zs::g_half_pi;
    //       psi = zs::atan2((*this)(2, 1), (*this)(2, 0));
    //       phi = (value_type)0;
    //     } else if (zs::math::near_zero(sinTheta + 1)) {
    //       theta = -zs::g_half_pi;
    //       psi = zs::atan2(-(*this)(2, 1), -(*this)(2, 0));
    //       phi = (value_type)0;
    //     } else {
    //       theta = std::asin(sinTheta);  // another solution: (zs::g_pi - theta)
    //       /// theta [-zs::g_pi/2, zs::g_pi/2], thus (cosTheta > 0) always holds true
    //       psi = std::atan2((*this)(0, 1), (*this)(0, 0));  // no need to divide cosTheta
    //       phi = std::atan2((*this)(1, 2), (*this)(2, 2));
    //     }
    //   }
    //   if constexpr (unit == angle_unit_e::radian)
    //     return zs::make_tuple(psi, theta, phi);
    //   else if constexpr (unit == angle_unit_e::degree)
    //     return zs::make_tuple(psi * (value_type)180 / (value_type)zs::g_pi,
    //                           theta * (value_type)180 / (value_type)zs::g_pi,
    //                           phi * (value_type)180 / (value_type)zs::g_pi);
    // }
    template <typename VecT,
              zs::enable_if_all<std::is_convertible_v<typename VecT::value_type, T>, VecT::dim == 1,
                            VecT::template range_t<0>::value == 4> = 0>
    constexpr Rotation(const zs::VecInterface<VecT> &q) noexcept : TM{} {
      if constexpr (dim == 2) {
        /// Construct a 2D counter clock wise rotation from the angle \a a in
        /// radian.
        T sinA = zs::sin(q(0)), cosA = zs::cos(q(0));
        (*this)(0, 0) = cosA;
        (*this)(0, 1) = sinA;
        (*this)(1, 0) = -sinA;
        (*this)(1, 1) = cosA;
      } else if constexpr (dim == 3) {
        /// The quaternion is required to be normalized, otherwise the result is
        /// undefined.
        self() = quaternion2matrix(q);
      }
    }
    template <
        typename VecTA, typename VecTB,
        zs::enable_if_all<std::is_convertible_v<typename VecTA::value_type, value_type>,
                      std::is_convertible_v<typename VecTB::value_type, value_type>,
                      VecTA::dim == 1, VecTB::dim == 1,
                      VecTA::template range_t<0>::value == VecTB::template range_t<0>::value> = 0>
    constexpr Rotation(const zs::VecInterface<VecTA> &a, const zs::VecInterface<VecTB> &b) noexcept : TM{} {
      if constexpr (dim == 2 && VecTA::template range_t<0>::value == 2) {
        auto aa = a.normalized();
        auto bb = b.normalized();
        (*this)(0, 0) = aa(0) * bb(0) + aa(1) * bb(1);
        (*this)(0, 1) = aa(0) * bb(1) - bb(0) * aa(1);
        (*this)(1, 0) = -(aa(0) * bb(1) - bb(0) * aa(1));
        (*this)(1, 1) = aa(0) * bb(0) + aa(1) * bb(1);
      } else if constexpr (dim == 3 && VecTA::template range_t<0>::value == 3) {
        T k_cos_theta = a.dot(b);
        T k = zs::sqrt(a.l2NormSqr() * b.l2NormSqr());
        zs::vec<T, 4> q{};
        if (k_cos_theta / k == -1) {
          // 180 degree rotation around any orthogonal vector
          q(3) = 0;
          auto c = a.orthogonal().normalized();
          q(0) = c(0);
          q(1) = c(1);
          q(2) = c(2);
        } else {
          q(3) = k_cos_theta + k;
          auto c = a.cross(b);
          q(0) = c(0);
          q(1) = c(1);
          q(2) = c(2);
          q = q.normalized();
        }
        self() = quaternion2matrix(q);
      }
    }

    template <typename VecT, int d = dim,
              zs::enable_if_all<d == 3, std::is_convertible_v<typename VecT::value_type, T>,
                            VecT::dim == 1, VecT::template range_t<0>::value == 4> = 0>
    static constexpr TM quaternion2matrix(const zs::VecInterface<VecT> &q) noexcept {
      /// (0, 1, 2, 3)
      /// (x, y, z, w)
      const T tx = T(2) * q(0);
      const T ty = T(2) * q(1);
      const T tz = T(2) * q(2);
      const T twx = tx * q(3);
      const T twy = ty * q(3);
      const T twz = tz * q(3);
      const T txx = tx * q(0);
      const T txy = ty * q(0);
      const T txz = tz * q(0);
      const T tyy = ty * q(1);
      const T tyz = tz * q(1);
      const T tzz = tz * q(2);
      TM rot{};
      rot(0, 0) = T(1) - (tyy + tzz);
      rot(0, 1) = txy + twz;
      rot(0, 2) = txz - twy;
      rot(1, 0) = txy - twz;
      rot(1, 1) = T(1) - (txx + tzz);
      rot(1, 2) = tyz + twx;
      rot(2, 0) = txz + twy;
      rot(2, 1) = tyz - twx;
      rot(2, 2) = T(1) - (txx + tyy);
      return rot;
    }
  };

template <typename T_, int dim_> 
struct Transform : zs::vec<T_, dim_ + 1, dim_ + 1> {
    using value_type = T_;
    static constexpr int dim = dim_;

    using mat_type = zs::vec<value_type, dim + 1, dim + 1>;

    constexpr decltype(auto) self() const { return static_cast<const mat_type &>(*this); }
    constexpr decltype(auto) self() { return static_cast<mat_type &>(*this); }

    /// translation
    template <typename VecT, zs::enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr void setToTranslation(const zs::VecInterface<VecT> &v) noexcept {
      self() = mat_type::identity();
      for (int i = 0; i != dim; ++i) self()(i, dim) = v[i];
    }

    template <typename VecT, zs::enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr void preTranslate(const zs::VecInterface<VecT> &v) noexcept {
      Transform tr{};
      tr.setToTranslation(v);
      self() = tr * self();
    }

    template <typename VecT, zs::enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr void postTranslate(const zs::VecInterface<VecT> &v) noexcept {
      Transform tr{};
      tr.setToTranslation(v);
      self() = self() * tr;
    }

    /// scale
    template <typename VecT, zs::enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr void setToScale(const zs::VecInterface<VecT> &v) noexcept {
      self() = mat_type::identity();
      for (int d = 0; d != dim; ++d) self()(d, d) = v[d];
    }

    template <typename VecT, zs::enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr void preScale(const zs::VecInterface<VecT> &v) noexcept {
      for (int i = 0; i != dim; ++i)
        for (int j = 0; j != dim + 1; ++j) self()(i, j) *= v[i];
    }

    template <typename VecT, zs::enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr void postScale(const zs::VecInterface<VecT> &v) noexcept {
      for (int i = 0; i != dim + 1; ++i)
        for (int j = 0; j != dim; ++j) self()(i, j) *= v[j];
    }

    /// rotation
    template <typename T, zs::enable_if_t<zs::is_convertible_v<T, value_type>> = 0>
    void setToRotation(const Rotation<T, dim> &r) noexcept {
      self() = mat_type::zeros();
      for (int i = 0; i != dim; ++i)
        for (int j = 0; j != dim; ++j) self()(i, j) = r(j, i);
      self()(dim, dim) = 1;
    }

    template <typename T, zs::enable_if_t<zs::is_convertible_v<T, value_type>> = 0>
    void preRotate(const Rotation<T, dim> &v) noexcept {
      Transform rot{};
      rot.setToRotation(v);
      self() = rot * self();
    }

    template <typename T, zs::enable_if_t<zs::is_convertible_v<T, value_type>> = 0>
    void postRotate(const Rotation<T, dim> &v) noexcept {
      Transform rot{};
      rot.setToRotation(v);
      self() = self() * rot;
    }

    template <typename VecT, zs::enable_if_all<
        VecT::dim == 2, 
        VecT::template range_t<0>::value == 4,
        VecT::template range_t<1>::value == 4> = 0>
    constexpr void preTransform(const zs::VecInterface<VecT> &t) noexcept {
        self() = t * self();
    }

    template <typename VecT, zs::enable_if_all<
        VecT::dim == 2, 
        VecT::template range_t<0>::value == 4,
        VecT::template range_t<1>::value == 4> = 0>
    constexpr void postTransform(const zs::VecInterface<VecT> &t) noexcept {
        self() = self() * t;
    }
};

template <class T>
auto getTransform(zs::vec<T, 3> p, zs::vec<T, 4> quat)
{
    using vec3 = zs::vec<T, 3>;
    using mat4 = zs::vec<T, 4, 4>;

    mat4 ret = mat4::identity(); 
    for (int di = 0; di < 3; di++)
        ret(di, 3) = p[di]; 
    for (int di = 0; di < 3; di++)
        for (int dj = 0; dj < 3; dj++)
            ret(di, dj) = quat[3 + di * 3 + dj]; 
    return ret; 
}

// eigen-zpc
template <class T, int Rows, int Cols>
constexpr zs::Vector<zs::vec<T, Cols>> eigen2Vector(const Eigen::Matrix<T, Rows, Cols> &mat) {
    using namespace zs;
    using VecT = vec<T, Cols>;
    Vector<vec<T, Cols>> ret {mat.rows(), memsrc_e::host, -1};
    for (int i = 0; i < mat.rows(); i++)
    {
        auto &currentRow = mat.row(i);
        for (int j = 0; j < mat.cols(); j++)
            ret[i][j] = currentRow[j];
    }
    return ret;
}

template <class T, int Rows>
constexpr zs::Vector<T> eigen2Vector(const Eigen::Matrix<T, Rows, 1> &mat) {
    using namespace zs;
    Vector<T> ret {mat.rows(), memsrc_e::host, -1};
    for (int i = 0; i < mat.rows(); i++)
        ret[i] = mat[i];
    return ret;
}

template <class T, int Rows, std::enable_if_t<Rows != Eigen::Dynamic, int> = 0>
constexpr auto eigen2Vec(const Eigen::Matrix<T, Rows, 1> & vec) {
    using vecT = zs::vec<T, Rows>;
    vecT ret;
    for (int i = 0; i < Rows; i++)
        ret[i] = vec[i];
    return ret;
}

template <class T, int Rows>
constexpr auto vec2Eigen(const zs::vec<T, Rows> &vec) {
    Eigen::Matrix<T, Rows, 1> ret;
    for (int i = 0; i < Rows; i++)
        ret[i] = vec[i];
    return ret;
}

template <class T, int Rows, int Cols>
constexpr auto vec2Eigen(const zs::vec<T, Rows, Cols> &mat) {
    Eigen::Matrix<T, Rows, Cols> ret;
    for (int i = 0; i < Rows; i++)
        for (int j = 0; j < Cols; j++)
            ret(i, j) = mat(i, j);
    return ret;
}

template <class T>
auto applyTransform(zs::vec<T, 3> point, zs::vec<T, 4, 4> transform)
{
    return to_vec3(transform * to_point4(point));
}

template <class T, int Rows>
Eigen::Matrix<T, Rows, 3> applyTransform(Eigen::Matrix<T, Rows, 3> const &points, Eigen::Matrix4<T> const &transform)
{
    return (points * transform.template block<3, 3>(0, 0).transpose()).rowwise() + transform.template block<3, 1>(0, 3).transpose();
}

inline void applyTransform(meshio::TriMesh &mesh, const eigen::mat4d &transform)
{
    mesh.setVerts(applyTransform(mesh.getVerts(), transform));
}


inline void applyTransform(meshio::TetMesh &mesh, const eigen::mat4d &transform)
{
    mesh.setVerts(applyTransform(mesh.getVerts(), transform));
}

template <typename T>
void applyTransform(meshio::TriMesh &mesh, const zs::vec<T, 4, 4> &transform)
{
    applyTransform(mesh, vec2Eigen(transform));
}

template <typename T>
void applyTransform(meshio::TetMesh &mesh, const zs::vec<T, 4, 4> &transform)
{
    applyTransform(mesh, vec2Eigen(transform));
}

// affine body data utils

// TODO: add enable_if to check dim 
template<class VecTA, class VecTB> 
constexpr auto ABD_q2x(const VecTA& pos0, const VecTB& q)
{
    using T = typename VecTA::value_type; 
    zs::vec<T, 3> ret{q(0), q(1), q(2)}; 
    for (int di = 0; di < 3; di++)
        for (int dj = 0; dj < 3; dj++)
            ret(di) += q(3 + di * 3 + dj) * pos0(dj); 
    return ret; 
}

// TODO: add enable_if to check dim 
template<class VecTA, class VecTB>
constexpr auto ABD_x2q(const VecTA& pos0, const VecTB& x)
{
    using T = typename VecTA::value_type;  
    zs::vec<T, 12> ret{}; 
    for (int d = 0; d < 3; d++)
        ret(d) = x(d); 
    for (int ci = 0; ci < 3; ci++)
        for (int d = 0; d < 3; d++)
            ret(ci * 3 + 3 + d) = pos0(d) * x(ci); 
    return ret; 
}

template<std::size_t start, std::size_t end, class VecT>
constexpr auto sliceVec(const VecT& q)
{
    using T = typename VecT::value_type; 
    using RetT = zs::vec<T, end - start>; 
    RetT ret{}; 
    for (int d = start; d < end; d++)
        ret[d - start] = q[d]; 
    return ret; 
}

template<class VecT>
constexpr auto q2T(const VecT& q)
{
    using T = typename VecT::value_type;
    using mat4 = zs::vec<T, 4, 4>;

    auto trans = mat4::identity();
    for (int di = 0; di < 3; di++)
        trans(di, 3) = q[di];
    for (int di = 0; di < 3; di++)
        for (int dj = 0; dj < 3; dj++)
            trans(di, dj) = q[3 + di * 3 + dj];
    return trans;
}

template<class VecT>
constexpr auto T2q(const VecT& trans)
{
    using T = typename VecT::value_type;
    using vec12 = zs::vec<T, 12>;

    vec12 q{};
    for (int di = 0; di < 3; di++)
    {
        q[di] = trans(di, 3);
        for (int dj = 0; dj < 3; dj++)
            q[3 + di * 3 + dj] = trans(di, dj);
    }
    return q;
}

template<class VecT>
constexpr auto q2A(const VecT& q)
{
    using T = typename VecT::value_type;
    using mat3 = zs::vec<T, 3, 3>;
    
    mat3 A{};
    for (int di = 0; di < 3; di++)
        for (int dj = 0; dj < 3; dj++)
            A(di, dj) = q[3 + di * 3 + dj]; 
    return A; 
}

template<class VecT>
constexpr auto Ahess2qhess(const VecT& Ahess)
{
    using T = typename VecT::value_type;
    using mat12 = zs::vec<T, 12, 12>;

    mat12 qhess{mat12::zeros()};
    for (int di = 0; di < 9; di++)
        for (int dj = 0; dj < 9; dj++)
            qhess(di+3, dj+3) = Ahess(di, dj);
    return qhess;
}

/// gradient and hessian utils
template<class VecInt, class VecT, class VViewT>
constexpr __device__ void scatterContactForce(VecInt& inds, VecT& grad, VViewT& vData)
{
    using namespace zs;
    using vec3 = vec<typename VecT::value_type, 3>;
    using VProps = typename VViewT::props_t;
    constexpr auto inds_N = VecInt::template range_t<0>::value;
    for (int i = 0; i <inds_N; i++)
    {
        int vi = inds[i];
        for (int d = 0; d < 3; d++)
            atomic_add(exec_cuda, &vData(VProps::contact, d, vi), grad[i * 3 + d]);
    }
}

/// gradient and hessian utils
template<class VecInt, class VecT, class VecMatT, class DOFViewT, class RBViewT, class VViewT, class SysHess>
constexpr __device__ void scatterGradientAndHessian(VecInt& segStart, VecInt& segLen, VecInt& segIsKin, 
                             VecT& grad, VecMatT& hess,
                             DOFViewT& dofData, RBViewT& rbData, VViewT& vData, int rbDofs, int sb_vOffset,  
                             SysHess& sysHess, bool kinematicSatisfied=true, 
                             bool includeHessian=true, bool includeForce=true, bool includePre = true)
{
    using namespace zs;
    using mat3 = vec<typename VecT::value_type, 3, 3>;
    using DOFProps = typename DOFViewT::props_t;
    using RBProps = typename RBViewT::props_t;
    using VProps = typename VViewT::props_t;
    constexpr int blockN = VecMatT::template range_t<0>::value / 3;
    using inds_t = vec<int, blockN>;
    int segN = VecInt::template range_t<0>::value;

    int qiOffset = 0, qjOffset = 0;
    for (int blockI = 0; blockI < segN; blockI++) {
        if (includeForce) 
        {
            if (segLen[blockI] == 12) // rigid body
                for (int d = 0; d < 12; d++) {
                    atomic_add(exec_cuda, 
                            &rbData(RBProps::contact, d, segStart[blockI] / 12),
                            grad[qiOffset + d]);
                }
            // no longer needed because point force is scattered beforehand
            // else // soft body
            //     for (int d = 0; d < 3; d++) 
            //         atomic_add(exec_cuda, 
            //             &vData(VProps::contact, d, (segStart[blockI] - rbDofs) / 3 + sb_vOffset), 
            //             grad[qiOffset + d]); 
        }
        if (segIsKin[blockI] && kinematicSatisfied)
        {
            qiOffset += segLen[blockI];
            continue; 
        }
        qjOffset = 0;
        if (!segIsKin[blockI])
            for (int d = 0; d < segLen[blockI]; d++) {
                atomic_add(exec_cuda, &dofData(DOFProps::grad, segStart[blockI] + d),
                        grad[qiOffset + d]);
            }

        qiOffset += segLen[blockI];
    }

    qiOffset = 0;
    if (includeHessian)
    {
        auto inds = inds_t::constant(-1);
        int top = 0;
        for (int seg = 0; seg < segN; ++seg)
        {
            if (segIsKin[seg] && kinematicSatisfied)
            {
                // if kinematic satisfied, we can skip the kinematic block since they are default to be -1
                top += segLen[seg] / 3;
                qiOffset += segLen[seg];
                continue;
            }
            for (int I = segStart[seg] / 3; I < (segStart[seg] + segLen[seg]) / 3; ++I)
            {
                inds[top++] = I;
            }
// #if s_enableCholmodDirectSolver
#if !s_enableAutoPrecondition
            if (includePre) // TODO: precondition
                if (segLen[seg] == 3) // soft body 
                {
                    int vi = (segStart[seg] - rbDofs) / 3 + sb_vOffset; 
                    // printf("scatter precondition for soft body vertex %d\n", (int)(vi - sb_vOffset));
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++)
                            atomic_add(exec_cuda, &vData(VProps::Pre, 3 * i + j, vi), hess(qiOffset + i, qiOffset + j));
                } else { // rigid body
                    int bi = segStart[seg] / 12; 
                    // printf("scatter precondition for rigid body %d\n", bi);
                    for (int k = 0; k < 4; k++)
                        for (int i = 0; i < 3; i++)
                            for (int j = 0; j < 3; j++)
                                atomic_add(exec_cuda, &rbData(RBProps::Pre, 9 * k + i * 3 + j, bi), hess(qiOffset + k * 3 + i, qiOffset + k * 3 + j));
                }
#endif
// #endif     
            qiOffset += segLen[seg];
        }

        // printf("od hess inds (size = %d): %d %d %d %d %d %d %d %d\n", blockN, inds[0], inds[1], inds[2], inds[3], inds[4], inds[5], inds[6], inds[7]);
        sysHess.addOffDiagHessNoTile(inds, inds, hess, includePre, rbData, vData, sb_vOffset);
        // sysHess.addOffDiagHess(inds, inds, hess); // TODO: still buggy
                       
    }
        
}

template <typename T> inline T computeHb(const T d2, const T dHat2) {
    if (d2 >= dHat2)
        return 0;
    T t2 = d2 - dHat2;
    return ((std::log(d2 / dHat2) * -2 - t2 * 4 / d2) + (t2 / d2) * (t2 / d2));
}

/// collision utils
template <int dim, typename T>
constexpr zs::AABBBox<dim, T> bv_empty() noexcept {
    zs::AABBBox<dim, T> box;
    for (int d = 0; d != dim; ++d) {
        box._min[d] = zs::limits<T>::max();
        box._max[d] = zs::limits<T>::lowest();
    }
    return box;
}

template <typename bv_t>
constexpr bv_t bv_empty() noexcept {
    bv_t box;
    for (int d = 0; d != bv_t::dim; ++d) {
        box._min[d] = zs::limits<typename bv_t::value_type>::max();
        box._max[d] = zs::limits<typename bv_t::value_type>::lowest();
    }
    return box;
}

template <int dim, typename T>
constexpr bool is_empty(const zs::AABBBox<dim, T> &box) noexcept {
    for (int d = 0; d != dim; ++d)
        if (box._min[d] > box._max[d])
            return true;
    return false;
}

// merge two bounding boxes
template <int dim, typename T>
constexpr zs::AABBBox<dim, T> merge(const zs::AABBBox<dim, T> &box1, const zs::AABBBox<dim, T> &box2) noexcept {
    if (is_empty(box1))
    {
        fmt::print("box1 is empty\n");
        return box2;
    }
    if (is_empty(box2))
    {
        fmt::print("box2 is empty\n");
        return box1;
    }
    zs::AABBBox<dim, T> box;
    for (int d = 0; d != dim; ++d) {
        box._min[d] = zs::min(box1._min[d], box2._min[d]);
        box._max[d] = zs::max(box1._max[d], box2._max[d]);
    }
    return box;
}

template <typename VProps, typename EProps>
inline void
retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, const TVWrapper<VProps> &vData,
                          const TVWrapper<EProps> &eles, int sbEOffset, 
                          zs::Vector<zs::AABBBox<3, typename VProps::value_t>> &rbBvs, zs::Vector<zs::AABBBox<3, typename VProps::value_t>> &sbBvs) {
    using namespace zs;
    using T = typename VProps::value_t;
    using Ti = zs::conditional_t<zs::is_same_v<T, float>, zs::i32, zs::i64>;
    using bv_t = AABBBox<3, T>;
    constexpr int codim = EProps::inds.numChannels;
    static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
    constexpr auto space = execspace_e::cuda;
    rbBvs.resize(sbEOffset);
    sbBvs.resize((int)eles.size() - sbEOffset);
    fmt::print("rb eles size: {}, sb eles size: {}\n", sbEOffset, (int)eles.size() - sbEOffset);
    pol(range(eles.size()), [eles = view<space>(eles, false_c, "eles"), vData = view<space>(vData, false_c, "vData"), 
                             rbBvs = view<space>(rbBvs, false_c, "rbBvs"), sbBvs = view<space>(sbBvs, false_c, "sbBvs"), 
                             sbEOffset] ZS_LAMBDA(int ei) mutable {
        constexpr int dim = EProps::inds.numChannels;
        auto inds = eles.pack(EProps::inds, ei).template reinterpret_bits<Ti>();
        auto x0 = vData.pack(VProps::xn, (int)inds[0]);
        bv_t bv{x0, x0};
        for (int d = 1; d != dim; ++d)
            merge(bv, vData.pack(VProps::xn, (int)inds[d]));
        if (ei < sbEOffset) // rigid body
            rbBvs(ei) = bv;
        else // soft body
            sbBvs(ei - sbEOffset) = bv;
    });
}


template <typename VProps, typename EProps>
inline void 
retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, const TVWrapper<VProps> &verts,
                          const TVWrapper<EProps> &eles, const TVWrapper<VProps> &vData, typename VProps::value_t stepSize, 
                          int sbEOffset, 
                          zs::Vector<zs::AABBBox<3, typename VProps::value_t>> &rbBvs, zs::Vector<zs::AABBBox<3, typename VProps::value_t>> &sbBvs) {
    using namespace zs;
    using T = typename VProps::value_t;
    using Ti = zs::conditional_t<zs::is_same_v<T, float>, zs::i32, zs::i64>;
    using bv_t = AABBBox<3, T>;
    constexpr int codim = EProps::inds.numChannels;
    static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
    constexpr auto space = execspace_e::cuda;
    rbBvs.resize(sbEOffset);
    sbBvs.resize(eles.size() - sbEOffset);
    pol(zs::range(eles.size()), [eles = view<space>(eles, false_c, "eles"), 
                                 rbBvs = view<space>(rbBvs), sbBvs = view<space>(sbBvs),
                                 verts = view<space>(verts, false_c, "verts"), vData = view<space>(vData, false_c, "vData"),
                                 stepSize, sbEOffset] ZS_LAMBDA(int ei) mutable {
        constexpr int dim = EProps::inds.numChannels;
        auto inds = eles.pack(EProps::inds, ei).template reinterpret_bits<Ti>();
        auto x0 = verts.pack(VProps::xn, inds[0]);
        auto dir0 = vData.pack(VProps::dir, inds[0]);
        bv_t bv{get_bounding_box(x0, x0 + stepSize * dir0)};
        for (int d = 1; d != dim; ++d) {
            auto x = verts.pack(VProps::xn, inds[d]);
            auto dir = vData.pack(VProps::dir, inds[d]);
            merge(bv, x);
            merge(bv, x + stepSize * dir);
        }
        if (ei < sbEOffset) // rigid body
            rbBvs[ei] = bv;
        else // soft body
            sbBvs[ei - sbEOffset] = bv;
    });
}

template <typename RBProps, typename VProps, typename EProps>
inline void pure_rigid_retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, TVWrapper<VProps> &vData, TVWrapper<RBProps> &rbData, typename VProps::value_t dHat, bool with_aug, const TVWrapper<EProps> &eles, int voffset, const zs::Vector<zs::AABBBox<3, typename VProps::value_t>>& bodyBoxes, zs::Vector<int>& culledInds, zs::Vector<zs::AABBBox<3, typename VProps::value_t>> &ret) {
    using namespace zs;
    using T = typename VProps::value_t;
    using Ti = zs::conditional_t<zs::is_same_v<T, double>, zs::i64, zs::i32>;
    using bv_t = AABBBox<3, T>;
    constexpr int codim = EProps::inds.numChannels;
    static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
    constexpr auto space = execspace_e::cuda;
    zs::Vector<int> cnt{ret.get_allocator(), 1}; 
    cnt.setVal(0); 
    ret.resize(eles.size()); 
    culledInds.resize(eles.size()); 
    pol(zs::range(eles.size()), [eles = view<space>(eles), bvs = proxy<space>(ret),
                                 rbData = view<space>(rbData), 
                                 vData = view<space>(vData), bodyBoxes = proxy<space>(bodyBoxes), 
                                 culledInds = proxy<space>(culledInds), 
                                 cnt = proxy<space>(cnt), 
                                 ret = proxy<space>(ret), 
                                 voffset, 
                                 dHat, with_aug] ZS_LAMBDA(int ei) mutable {
        constexpr int dim = EProps::inds.numChannels;
        auto inds = eles.pack(EProps::inds, ei).template reinterpret_bits<Ti>() + voffset;
        auto x0 = vData.pack(VProps::xn, inds[0]);
        bv_t bv{x0, x0};
        for (int d = 1; d != dim; ++d) {
            auto x = vData.pack(VProps::xn, inds[d]);
            merge(bv, x);
        }
        bv_t aug_bv{bv._min - dHat, bv._max + dHat}; 
        int bi = reinterpret_bits<Ti>(vData(VProps::body, inds[0])); 
        auto excl_i = rbData(RBProps::exclGrpIdx, bi); 
        for (int bj = 0; bj <  bodyBoxes.size(); bj++)
        {
            if (bj == bi)
                continue; 
            if (rbData(RBProps::isGhost, bj))
                continue; 
            if (excl_i >= 0)
            {
                auto excl_j = rbData(RBProps::exclGrpIdx, bj);
                if (excl_i == excl_j)
                    continue;  
            }
            auto r_bv = bodyBoxes[bj]; 
            if (overlaps(aug_bv, r_bv))
            {
                auto no = atomic_add(exec_cuda, &cnt[0], 1); 
                ret[no] = with_aug ? bv_t{bv._min - dHat * 0.5, bv._max + dHat * 0.5} : bv; 
                culledInds[no] = ei; 
                return; 
            }
        }
    });
    auto cntVal = cnt.getVal(); 
    ret.resize(cntVal);
    culledInds.resize(cntVal); 
}

template <typename RBProps, typename VProps, typename EProps>
inline void pure_rigid_retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, TVWrapper<VProps> &vData, TVWrapper<RBProps> &rbData, typename VProps::value_t dHat, typename VProps::value_t stepSize, const TVWrapper<EProps> &eles, int voffset, const zs::Vector<zs::AABBBox<3, typename VProps::value_t>>& bodyBoxes, zs::Vector<int>& culledInds, zs::Vector<zs::AABBBox<3, typename VProps::value_t>> &ret) {
    using namespace zs;
    using T = typename VProps::value_t;
    using Ti = zs::conditional_t<zs::is_same_v<T, double>, zs::i64, zs::i32>;
    using bv_t = AABBBox<3, T>;
    constexpr int codim = EProps::inds.numChannels;
    static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
    constexpr auto space = execspace_e::cuda;
    zs::Vector<int> cnt{ret.get_allocator(), 1}; 
    cnt.setVal(0); 
    ret.resize(eles.size()); 
    culledInds.resize(eles.size()); 
    pol(zs::range(eles.size()), [eles = view<space>(eles), bvs = proxy<space>(ret),
                                 rbData = view<space>(rbData), 
                                 vData = view<space>(vData), bodyBoxes = proxy<space>(bodyBoxes), 
                                 culledInds = proxy<space>(culledInds), 
                                 cnt = proxy<space>(cnt), 
                                 ret = proxy<space>(ret), 
                                 dHat, 
                                 stepSize, voffset] ZS_LAMBDA(int ei) mutable {
        constexpr int dim = EProps::inds.numChannels;
        auto inds = eles.pack(EProps::inds, ei).template reinterpret_bits<Ti>() + voffset;
        auto x0 = vData.pack(VProps::xn, inds[0]);
        auto dir0 = vData.pack(VProps::dir, inds[0]);
        bv_t bv{get_bounding_box(x0, x0 + stepSize * dir0)};
        for (int d = 1; d != dim; ++d) {
            auto x = vData.pack(VProps::xn, inds[d]);
            auto dir = vData.pack(VProps::dir, inds[d]);
            merge(bv, x);
            merge(bv, x + stepSize * dir);
        }
        bv_t aug_bv{bv._min - dHat, bv._max + dHat}; 
        int bi = reinterpret_bits<Ti>(vData(VProps::body, inds[0]));
        auto excl_i = rbData(RBProps::exclGrpIdx, bi);
        for (int bj = 0; bj <  bodyBoxes.size(); bj++)
        {
            if (bj == bi)
                continue; 
            if (rbData(RBProps::isGhost, bj))
                continue; 
            if (excl_i >= 0)
            {
                auto excl_j = rbData(RBProps::exclGrpIdx, bj);
                if (excl_i == excl_j)
                    continue;  
            }
            auto a_bv = bodyBoxes[bj]; 
            if (overlaps(aug_bv, a_bv))
            {
                auto no = atomic_add(exec_cuda, &cnt[0], 1); 
                ret[no] = bv; 
                culledInds[no] = ei; 
                return; 
            }
        }
        
    });
    auto cntVal = cnt.getVal(); 
    ret.resize(cntVal);
    culledInds.resize(cntVal); 
}

namespace accd {

template <typename VecT>
constexpr bool 
ppccd(VecT p, VecT q, VecT dp, VecT dq, 
      typename VecT::value_type eta, typename VecT::value_type thickness,
      typename VecT::value_type &toc, typename VecT::value_type tStart = 0) {
  using T = typename VecT::value_type;
  auto mov = (dp + dq) / (T)2;
  dp -= mov;
  dq -= mov;
  T maxDispMag = dp.norm() + dq.norm(); 
  if (maxDispMag == 0)
    return false;

  T dist2_cur = (p - q).l2NormSqr(); 
  T dist_cur = zs::sqrt(dist2_cur); 
  T gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
  T toc_prev = toc;
  toc = tStart;
  int iter = 0;
  while (++iter < 20000) {
    // while (true) {
    T tocLowerBound = (1 - eta) * (dist2_cur - thickness * thickness) /
                      ((dist_cur + thickness) * maxDispMag);
    if (tocLowerBound < 0)
      printf("damn pp!\n");

    p += tocLowerBound * dp;
    q += tocLowerBound * dq;
    dist2_cur = (p - q).l2NormSqr(); 
    dist_cur = zs::sqrt(dist2_cur);
    if (toc &&
        ((dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap))
      break;

    toc += tocLowerBound;
    if (toc > toc_prev) {
      toc = toc_prev;
      return false;
    }
  }
  return true;
}

template <typename VecT>
constexpr bool
ptccd(VecT p, VecT t0, VecT t1, VecT t2, VecT dp, VecT dt0, VecT dt1, VecT dt2,
      typename VecT::value_type eta, typename VecT::value_type thickness,
      typename VecT::value_type &toc, typename VecT::value_type tStart = 0) {
  using T = typename VecT::value_type;
  auto mov = (dt0 + dt1 + dt2 + dp) / 4;
  dt0 -= mov;
  dt1 -= mov;
  dt2 -= mov;
  dp -= mov;
  T dispMag2Vec[3] = {dt0.l2NormSqr(), dt1.l2NormSqr(), dt2.l2NormSqr()};
  T tmp = zs::limits<T>::lowest();
  for (int i = 0; i != 3; ++i)
    if (dispMag2Vec[i] > tmp)
      tmp = dispMag2Vec[i];
  T maxDispMag = dp.norm() + zs::sqrt(tmp);
  if (maxDispMag == 0)
    return false;

  T dist2_cur = dist2_pt_unclassified(p, t0, t1, t2);
  T dist_cur = zs::sqrt(dist2_cur);
  T gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
  T toc_prev = toc;
  toc = tStart;
  int iter = 0;
  while (++iter < 20000) {
    // while (true) {
    T tocLowerBound = (1 - eta) * (dist2_cur - thickness * thickness) /
                      ((dist_cur + thickness) * maxDispMag);
    if (tocLowerBound < 0)
      printf("damn pt!\n");

    p += tocLowerBound * dp;
    t0 += tocLowerBound * dt0;
    t1 += tocLowerBound * dt1;
    t2 += tocLowerBound * dt2;
    dist2_cur = dist2_pt_unclassified(p, t0, t1, t2);
    dist_cur = zs::sqrt(dist2_cur);
    if (toc &&
        ((dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap))
      break;

    toc += tocLowerBound;
    if (toc > toc_prev) {
        // printf("PT: \np: %f, %f, %f\nt0: %f, %f, %f\nt1: %f, %f, %f\nt2: %f, %f, %f\n"
        //        "dp: %f, %f, %f\ndt0: %f, %f, %f\ndt1: %f, %f, %f\ndt2: %f, %f, %f\n"
        //        "dist2: %f, alpha: %f\n", 
        //        p(0), p(1), p(2), t0(0), t0(1), t0(2), t1(0), t1(1), t1(2), t2(0), t2(1), t2(2), 
        //        dp(0), dp(1), dp(2), dt0(0), dt0(1), dt0(2), dt1(0), dt1(1), dt1(2), dt2(0), dt2(1), dt2(2), 
        //        dist2_cur, toc);
        toc = toc_prev;
        return false;
    }
  }
  return true;
}
template <typename VecT>
constexpr bool
eeccd(VecT ea0, VecT ea1, VecT eb0, VecT eb1, VecT dea0, VecT dea1, VecT deb0,
      VecT deb1, typename VecT::value_type eta,
      typename VecT::value_type thickness, typename VecT::value_type &toc,
      typename VecT::value_type tStart = 0) {
  using T = typename VecT::value_type;
  auto mov = (dea0 + dea1 + deb0 + deb1) / 4;
  dea0 -= mov;
  dea1 -= mov;
  deb0 -= mov;
  deb1 -= mov;
  T maxDispMag = zs::sqrt(zs::max(dea0.l2NormSqr(), dea1.l2NormSqr())) +
                 zs::sqrt(zs::max(deb0.l2NormSqr(), deb1.l2NormSqr()));
  if (maxDispMag == 0)
    return false;

  T dist2_cur = dist2_ee_unclassified(ea0, ea1, eb0, eb1);
  T dFunc = dist2_cur - thickness * thickness;
  if (dFunc <= 0) {
    // since we ensured other place that all dist smaller than dHat are
    // positive, this must be some far away nearly parallel edges
    T dists[] = {(ea0 - eb0).l2NormSqr(), (ea0 - eb1).l2NormSqr(),
                 (ea1 - eb0).l2NormSqr(), (ea1 - eb1).l2NormSqr()};
    {
      dist2_cur = zs::limits<T>::max();
      for (const auto &dist : dists)
        if (dist < dist2_cur)
          dist2_cur = dist;
      // dist2_cur = *std::min_element(dists.begin(), dists.end());
    }
    dFunc = dist2_cur - thickness * thickness;
  }
  T dist_cur = zs::sqrt(dist2_cur);
  T gap = eta * dFunc / (dist_cur + thickness);
  T toc_prev = toc;
  toc = tStart;
  int iter = 0;
  while (++iter < 20000) {
    // while (true) {
    T tocLowerBound = (1 - eta) * dFunc / ((dist_cur + thickness) * maxDispMag);
    if (tocLowerBound < 0)
      printf("damn ee!\n");

    ea0 += tocLowerBound * dea0;
    ea1 += tocLowerBound * dea1;
    eb0 += tocLowerBound * deb0;
    eb1 += tocLowerBound * deb1;
    dist2_cur = dist2_ee_unclassified(ea0, ea1, eb0, eb1);
    dFunc = dist2_cur - thickness * thickness;
    if (dFunc <= 0) {
      // since we ensured other place that all dist smaller than dHat are
      // positive, this must be some far away nearly parallel edges
      T dists[] = {(ea0 - eb0).l2NormSqr(), (ea0 - eb1).l2NormSqr(),
                   (ea1 - eb0).l2NormSqr(), (ea1 - eb1).l2NormSqr()};
      {
        dist2_cur = zs::limits<T>::max();
        for (const auto &dist : dists)
          if (dist < dist2_cur)
            dist2_cur = dist;
      }
      dFunc = dist2_cur - thickness * thickness;
    }
    dist_cur = zs::sqrt(dist2_cur);
    if (toc && (dFunc / (dist_cur + thickness) < gap))
      break;

    toc += tocLowerBound;
    if (toc > toc_prev) {
      toc = toc_prev;
      return false;
    }
  }
  return true;
}
} // namespace accd

namespace ticcd {

template <typename VecT>
constexpr bool
ptccd(const VecT &p, const VecT &t0, const VecT &t1, const VecT &t2,
      const VecT &dp, const VecT &dt0, const VecT &dt1, const VecT &dt2,
      typename VecT::value_type eta, typename VecT::value_type thickness,
      typename VecT::value_type &toc) {
  using T = typename VecT::value_type;
  T t = toc;
  auto pend = p + t * dp;
  auto t0end = t0 + t * dt0;
  auto t1end = t1 + t * dt1;
  auto t2end = t2 + t * dt2;

  constexpr zs::vec<double, 3> err(-1, -1, -1);
  bool earlyTerminate = false;
  double ms = 1e-8;
  double toi{};
  const double tolerance = 1e-6;
  const double t_max = 1;
  const int max_itr = 3e5;
  double output_tolerance = 1e-6;
  while (vertexFaceCCD(p, t0, t1, t2, pend, t0end, t1end, t2end, err, ms, toi,
                       tolerance, t_max, max_itr, output_tolerance,
                       earlyTerminate, true) &&
         !earlyTerminate) {
    t = zs::min(t / 2, toi);
    pend = p + t * dp;
    t0end = t0 + t * dt0;
    t1end = t1 + t * dt1;
    t2end = t2 + t * dt2;
  }

  if (earlyTerminate) {
    toc = t;
    if (accd::ptccd(p, t0, t1, t2, dp, dt0, dt1, dt2, eta, thickness, toc)) {
      return true;
    }
  }
  if (t == toc) {
    return false;
  } else {
    toc = t * (1 - eta);
    return true;
  }
}

template <typename VecT>
constexpr bool
eeccd(const VecT &ea0, const VecT &ea1, const VecT &eb0, const VecT &eb1,
      const VecT &dea0, const VecT &dea1, const VecT &deb0, const VecT &deb1,
      typename VecT::value_type eta, typename VecT::value_type thickness,
      typename VecT::value_type &toc) {
  using T = typename VecT::value_type;
  T t = toc;
  auto ea0end = ea0 + t * dea0;
  auto ea1end = ea1 + t * dea1;
  auto eb0end = eb0 + t * deb0;
  auto eb1end = eb1 + t * deb1;

  constexpr zs::vec<double, 3> err(-1, -1, -1);
  bool earlyTerminate = false;
  double ms = 1e-8;
  double toi{};
  const double tolerance = 1e-6;
  const double t_max = 1;
  const int max_itr = 3e5;
  double output_tolerance = 1e-6;
  while (edgeEdgeCCD(ea0, ea1, eb0, eb1, ea0end, ea1end, eb0end, eb1end, err,
                     ms, toi, tolerance, t_max, max_itr, output_tolerance,
                     earlyTerminate, true) &&
         !earlyTerminate) {
    t = zs::min(t / 2, toi);
    ea0end = ea0 + t * dea0;
    ea1end = ea1 + t * dea1;
    eb0end = eb0 + t * deb0;
    eb1end = eb1 + t * deb1;
  }

  if (earlyTerminate) {
    toc = t;
    if (accd::eeccd(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, eta, thickness,
                    toc)) {
      return true;
    }
  }

  if (t == toc) {
    return false;
  } else {
    toc = t * (1 - eta);
    return true;
  }
}
} // namespace ticcd

}; // namespace tacipc