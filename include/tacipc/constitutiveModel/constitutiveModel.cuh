#pragma once
#include <variant>
#include <tacipc/types.hpp>
#include <tacipc/meta.hpp>
#include <tacipc/utils.cuh>
#include <tacipc/gradHessMATLAB.cuh>
#include <zensim/physics/ConstitutiveModel.hpp>
#include <zensim/physics/constitutive_models/StvkWithHencky.hpp>
#include <zensim/physics/constitutive_models/NeoHookean.hpp>
#include <zensim/math/matrix/Eigen.hpp>
#include <zensim/math/matrix/SVD.hpp>

namespace tacipc {
// matrix and vector type templates for codim 
template <class MatT>
using enable_if_mat_t_dim = zs::enable_if_all<MatT::dim == 2, MatT::template range_t<0>::value <= 3, MatT::template range_t<1>::value <= 3, std::is_floating_point_v<typename MatT::value_type>>;

template <class VecT>
using enable_if_vec_t_dim = zs::enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3, std::is_floating_point_v<typename VecT::value_type>>;

template <class VecT, enable_if_vec_t_dim<VecT> = 0>
using MatT = zs::vec<typename VecT::value_type, VecT::template range_t<0>::value, VecT::template range_t<0>::value>;

template <class MatT, enable_if_mat_t_dim<MatT> = 0>
using MatT4Mat = zs::vec<typename MatT::value_type, MatT::template range_t<0>::value * MatT::template range_t<1>::value, MatT::template range_t<0>::value * MatT::template range_t<1>::value>;

template <class Model> struct IsotropicConstitutiveModel {
    using model_t = Model;

    // energy, psi(F)
    template <class MatT, enable_if_mat_t_dim<MatT> = 0>
    constexpr decltype(auto) energy(const MatT &F) const noexcept 
    {
        auto [U, S, V] = zs::math::svd(F);
        return static_cast<const Model*>(this)->psi_sigma(S);
        // attention: dt^2 * vol remains to be multiplied
    }
    // gradient dpsi/dx(F, IB) and hessian d2psi/dx2(F, IB)
    template <class MatTA, class MatTB, enable_if_mat_t_dim<MatTA> = 0, enable_if_mat_t_dim<MatTB> = 0> // TODO: check if this is correct (shape(F) == shape(IB))
    constexpr decltype(auto) gradientAndHessian(const MatTA &F, const MatTB &IB) const noexcept
    {
        auto P = first_piola(F);
        constexpr int dim = MatTA::template range_t<0>::value;
        auto vecP = flatten(P); // view F as vec in a column-major fashion 
        auto dFdX = dFdXMatrix(IB);
        auto dFdXT = dFdX.transpose();
        auto vfdt2 = -dFdXT * vecP; // this is gradient

        auto Hq = first_piola_derivative(F, zs::true_c);
        auto H = dFdXT * Hq * dFdX; // this is hessian

        return zs::tuple{vfdt2, H};
        // attention: dt^2 * vol remains to be multiplied   
    }
    // first piola
    template <class MatT, enable_if_mat_t_dim<MatT> = 0>
    constexpr auto first_piola(const MatT &F) const noexcept
    {
        auto [U, S, V] = zs::math::svd(F);
        auto dE_dsigma = static_cast<const Model*>(this)->dpsi_dsigma(S);
        return diag_mul(U, dE_dsigma) * V.transpose();
    }
    // first piola derivative
    template <class MatT, enable_if_mat_t_dim<MatT> = 0, bool project_SPD = false>
    constexpr auto first_piola_derivative(const MatT& F, zs::wrapv<project_SPD> = {}) const noexcept
    {

        using namespace zs;
        using T = typename MatT::value_type;
        using Ti = typename MatT::index_type;
        constexpr int dim = MatT::template range_t<0>::value;

        auto [U, S, V] = math::svd(F);
        auto dE_dsigma = static_cast<const Model*>(this)->dpsi_dsigma(S);
        // A
        auto d2E_dsigma2 = static_cast<const Model*>(this)->d2psi_dsigma2(S);
        if constexpr (project_SPD) make_pd(d2E_dsigma2);
        // Bij
        using MatB = typename MatT::template variant_vec<T, integer_sequence<Ti, 2, 2>>;
        auto ComputeBij = [&dE_dsigma, &S = S,
                            Bij_left_coeffs = Bij_neg_coeff(S)](int i) -> MatB {  // i -> i, i + 1
            constexpr int dim = MatT::template range_t<0>::value;
            int j = (i + 1) % dim;
            T leftCoeff = Bij_left_coeffs[i];
            T rightDenom = math::max(S[i] + S[j], (T)1e-6);  // prevents division instability
            T rightCoeff = (dE_dsigma[i] + dE_dsigma[j]) / (rightDenom + rightDenom);
            return MatB{leftCoeff + rightCoeff, leftCoeff - rightCoeff, leftCoeff - rightCoeff,
                        leftCoeff + rightCoeff};
        };
        using MatH = typename MatT::template variant_vec<T, integer_sequence<Ti, dim * dim, dim * dim>>;
        MatH dPdF{};

        if constexpr (is_same_v<typename MatT::dims, index_sequence<3, 3>>) {
            auto B0 = ComputeBij(0) /*B12*/, B1 = ComputeBij(1) /*B23*/, B2 = ComputeBij(2) /*B13*/;
            if constexpr (project_SPD) {
            make_pd(B0);
            make_pd(B1);
            make_pd(B2);
            }
            for (int ji = 0; ji != dim * dim; ++ji) {
            int j = ji / dim;
            int i = ji - j * dim;
            for (int sr = 0; sr <= ji; ++sr) {
                int s = sr / dim;
                int r = sr - s * dim;
                dPdF(ji, sr) = dPdF(sr, ji)
                    = d2E_dsigma2(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0)
                    + d2E_dsigma2(0, 1) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1)
                    + d2E_dsigma2(0, 2) * U(i, 0) * V(j, 0) * U(r, 2) * V(s, 2)
                    + d2E_dsigma2(1, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0)
                    + d2E_dsigma2(1, 1) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1)
                    + d2E_dsigma2(1, 2) * U(i, 1) * V(j, 1) * U(r, 2) * V(s, 2)
                    + d2E_dsigma2(2, 0) * U(i, 2) * V(j, 2) * U(r, 0) * V(s, 0)
                    + d2E_dsigma2(2, 1) * U(i, 2) * V(j, 2) * U(r, 1) * V(s, 1)
                    + d2E_dsigma2(2, 2) * U(i, 2) * V(j, 2) * U(r, 2) * V(s, 2)
                    + B0(0, 0) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1)
                    + B0(0, 1) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0)
                    + B0(1, 0) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1)
                    + B0(1, 1) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0)
                    + B1(0, 0) * U(i, 1) * V(j, 2) * U(r, 1) * V(s, 2)
                    + B1(0, 1) * U(i, 1) * V(j, 2) * U(r, 2) * V(s, 1)
                    + B1(1, 0) * U(i, 2) * V(j, 1) * U(r, 1) * V(s, 2)
                    + B1(1, 1) * U(i, 2) * V(j, 1) * U(r, 2) * V(s, 1)
                    + B2(1, 1) * U(i, 0) * V(j, 2) * U(r, 0) * V(s, 2)
                    + B2(1, 0) * U(i, 0) * V(j, 2) * U(r, 2) * V(s, 0)
                    + B2(0, 1) * U(i, 2) * V(j, 0) * U(r, 0) * V(s, 2)
                    + B2(0, 0) * U(i, 2) * V(j, 0) * U(r, 2) * V(s, 0);
            }
            }
        } else if constexpr (is_same_v<typename MatT::dims, index_sequence<2, 2>>) {
            auto B = ComputeBij(0);
            if constexpr (project_SPD) make_pd(B);
            for (int ji = 0; ji != dim * dim; ++ji) {
            int j = ji / dim;
            int i = ji - j * dim;
            for (int sr = 0; sr <= ji; ++sr) {
                int s = sr / dim;
                int r = sr - s * dim;
                dPdF(ji, sr) = dPdF(sr, ji)
                    = d2E_dsigma2(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0)
                    + d2E_dsigma2(0, 1) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1)
                    + B(0, 0) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1)
                    + B(0, 1) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0)
                    + B(1, 0) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1)
                    + B(1, 1) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0)
                    + d2E_dsigma2(1, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0)
                    + d2E_dsigma2(1, 1) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1);
            }
            }
        } else if constexpr (is_same_v<typename MatT::dims, index_sequence<1, 1>>) {
            dPdF(0, 0) = d2E_dsigma2(0, 0);  // U = V = [1]
        }
        return dPdF;
    }
    
    // expressions with sigma 
    template <class VecT, enable_if_vec_t_dim<VecT> = 0>
    constexpr typename VecT::value_type psi_sigma(const VecT &S) const noexcept { return 0; }
    template <class VecT, enable_if_vec_t_dim<VecT> = 0>
    constexpr auto dpsi_dsigma(const VecT &S) const noexcept { return VecT::zeros();}
    template <class VecT, enable_if_vec_t_dim<VecT> = 0>
    constexpr auto d2psi_dsigma2(const VecT &S) const noexcept { return MatT<VecT>::zeros(); }
    template <class VecT, enable_if_vec_t_dim<VecT> = 0>
    constexpr auto Bij_neg_coeff(const VecT &S) const noexcept { return VecT::zeros(); }
};

template <class T> struct StVKWithHenckyModel : public IsotropicConstitutiveModel<StVKWithHenckyModel<T>> {
    using value_t = T;

    value_t lambda, mu; // Lame parameters
    zs::StvkWithHencky<T> _model;

    constexpr StVKWithHenckyModel() noexcept = default;
    constexpr StVKWithHenckyModel(value_t E, value_t nu) noexcept 
        : _model{E, nu}
    {
        zs::tie(mu, lambda) = zs::lame_parameters(E, nu);
    }

    template <class VecT, enable_if_vec_t_dim<VecT> = 0>
    constexpr typename VecT::value_type psi_sigma(const VecT &S) const noexcept
    {
        return _model.do_psi_sigma(S);
        const auto S_log = S.abs().log();
        const auto S_log_trace = S_log.sum();
        return mu * S_log.square().sum() + (value_t)0.5 * lambda * S_log_trace * S_log_trace;
    }
    template <class VecT, enable_if_vec_t_dim<VecT> = 0>
    constexpr auto dpsi_dsigma(const VecT &S) const noexcept
    {
        return _model.do_dpsi_dsigma(S);
        const auto S_log = S.abs().log();
        const auto S_log_trace = S_log.sum();
        return ((mu + mu) * S_log + lambda * S_log_trace) / S;
    }
    template <class VecT, enable_if_vec_t_dim<VecT> = 0>
    constexpr auto d2psi_dsigma2(const VecT &S) const noexcept
    {
        return _model.do_d2psi_dsigma2(S);
        constexpr auto dim = VecT::template range_t<0>::value;

        const auto S_log = S.abs().log();
        const auto S_log_trace = S_log.sum();
        const auto _2mu = mu + mu;
        const auto _1_m_S_log_trace = (value_t)1 - S_log_trace;
        MatT<VecT> d2E_dsigma2{};
        d2E_dsigma2(0, 0)
            = (_2mu * ((value_t)1 - S_log(0)) + lambda * _1_m_S_log_trace) / (S[0] * S[0]);
        if constexpr (dim > 1) {
            d2E_dsigma2(1, 1)
                = (_2mu * ((value_t)1 - S_log(1)) + lambda * _1_m_S_log_trace) / (S[1] * S[1]);
            d2E_dsigma2(0, 1) = d2E_dsigma2(1, 0) = lambda / (S[0] * S[1]);
        }
        if constexpr (dim > 2) {
            d2E_dsigma2(2, 2)
                = (_2mu * ((value_t)1 - S_log(2)) + lambda * _1_m_S_log_trace) / (S[2] * S[2]);
            d2E_dsigma2(0, 2) = d2E_dsigma2(2, 0) = lambda / (S[0] * S[2]);
            d2E_dsigma2(1, 2) = d2E_dsigma2(2, 1) = lambda / (S[1] * S[2]);
        }
        return d2E_dsigma2;
    }
    template <class VecT, enable_if_vec_t_dim<VecT> = 0>
    constexpr auto Bij_neg_coeff(const VecT &S) const noexcept
    {
        return _model.do_Bij_neg_coeff(S);
        using namespace zs;
        constexpr auto dim = VecT::template range_t<0>::value;
        using RetT = typename VecT::template variant_vec<
            value_t,
            integer_sequence<typename VecT::index_type, (dim == 3 ? 3 : 1)>>;
        RetT coeffs{};

        const auto S_log = S.abs().log();
        constexpr value_t eps = 1e-6;
        if constexpr (dim == 2) {
            auto q = zs::max(S[0] / S[1] - 1, -1 + eps);
            auto h = zs::abs(q) < eps ? (value_t)1 : (zs::log1p(q) / q);
            auto t = h / S[1];
            auto z = S_log[1] - t * S[1];
            coeffs[0] = -(lambda * (S_log[0] + S_log[1]) + (mu + mu) * z) / S.prod() * (value_t)0.5;
        } else if constexpr (dim == 3) {
            const auto S_log_trace = S_log.sum();
            const auto _2mu = mu + mu;
            coeffs[0]
                = -(lambda * S_log_trace
                    + _2mu * math::diff_interlock_log_over_diff(S(0), zs::abs(S(1)), S_log(1), eps))
                / (S[0] * S[1]) * (value_t)0.5;
            coeffs[1]
                = -(lambda * S_log_trace
                    + _2mu * math::diff_interlock_log_over_diff(S(1), zs::abs(S(2)), S_log(2), eps))
                / (S[1] * S[2]) * (value_t)0.5;
            coeffs[2]
                = -(lambda * S_log_trace
                    + _2mu * math::diff_interlock_log_over_diff(S(0), zs::abs(S(2)), S_log(2), eps))
                / (S[0] * S[2]) * (value_t)0.5;
        }
        return coeffs;
    }
};

template <class T> struct NeoHookeanModel : public IsotropicConstitutiveModel<NeoHookeanModel<T>> {
    using value_t = T;

    value_t lambda, mu; // Lame parameters
    zs::NeoHookean<T> _model;

    constexpr NeoHookeanModel() noexcept = default;
    constexpr NeoHookeanModel(value_t E, value_t nu) noexcept 
        : _model{E, nu}
    {
        zs::tie(mu, lambda) = zs::lame_parameters(E, nu);
    }

    template <class VecT, enable_if_vec_t_dim<VecT> = 0>
    constexpr typename VecT::value_type psi_sigma(const VecT &S) const noexcept
    {
        return _model.do_psi_sigma(FWD(S));
    }
    template <class VecT, enable_if_vec_t_dim<VecT> = 0>
    constexpr auto dpsi_dsigma(const VecT &S) const noexcept
    {
        return _model.do_dpsi_dsigma(FWD(S));
    }
    template <class VecT, enable_if_vec_t_dim<VecT> = 0>
    constexpr auto d2psi_dsigma2(const VecT &S) const noexcept
    {
        return _model.do_d2psi_dsigma2(FWD(S));
    }
    template <class VecT, enable_if_vec_t_dim<VecT> = 0>
    constexpr auto Bij_neg_coeff(const VecT &S) const noexcept
    {
        return _model.Bij_neg_coeff(FWD(S));
    }
};

template <class Model> struct AnisotropicConstitutiveModel {
    using model_t = Model;

    // energy, psi(F)
    template <class MatT, enable_if_mat_t_dim<MatT> = 0>
    constexpr decltype(auto) energy(const MatT &F) const noexcept 
    {
        return static_cast<const Model*>(this)->psi_F(F);
        // attention: dt^2 * vol remains to be multiplied
    }
    // gradient dpsi/dx(F, IB) and hessian d2psi/dx2(F, IB)
    template <class MatTA, class MatTB, enable_if_mat_t_dim<MatTA> = 0, enable_if_mat_t_dim<MatTB> = 0> // TODO: check if this is correct (shape(F) == shape(IB))
    constexpr decltype(auto) gradientAndHessian(const MatTA &F, const MatTB &IB) const noexcept
    {
        constexpr int dim = MatTA::template range_t<0>::value;
        auto P = first_piola(F);
        auto vecP = flatten(P); // view F as vec in a column-major fashion 
        auto dFdX = dFdXMatrix(IB, zs::wrapv<dim>{});
        auto dFdXT = dFdX.transpose();
        auto vfdt2 = -dFdXT * vecP; // this is gradient
        auto Hq = first_piola_derivative(F, zs::true_c);
        auto H = dFdXT * Hq * dFdX; // this is hessian

        return zs::tuple{vfdt2, H};
        // attention: dt^2 * vol remains to be multiplied   
    }
    // first piola
    template <class MatT, enable_if_mat_t_dim<MatT> = 0>
    constexpr auto first_piola(const MatT &F) const noexcept
    {
        return static_cast<const Model*>(this)->dpsi_dF(F);
    }
    // first piola derivative
    template <class MatT, enable_if_mat_t_dim<MatT> = 0, bool project_SPD = false>
    constexpr auto first_piola_derivative(const MatT& F, zs::wrapv<project_SPD> = {}) const noexcept
    {
        auto d2E_dF2 = static_cast<const Model*>(this)->d2psi_dF2(F);
        if constexpr (project_SPD) make_pd(d2E_dF2);
        return d2E_dF2;
    }
    
    template <class MatT, enable_if_mat_t_dim<MatT> = 0>
    constexpr typename MatT::value_type psi_F(const MatT &F) const noexcept { return 0; }
    template <class MatT, enable_if_mat_t_dim<MatT> = 0>
    constexpr auto dpsi_dF(const MatT &F) const noexcept { return MatT::zeros();}
    template <class MatT, enable_if_mat_t_dim<MatT> = 0>
    constexpr auto d2psi_dF2(const MatT &F) const noexcept { return MatT4Mat<MatT>::zeros(); }
};

/// ref: A Finite Element Formulation of Baraff-Witkin Cloth
// suggested by huang kemeng
template <class T> struct BaraffWitkinModel : public AnisotropicConstitutiveModel<BaraffWitkinModel<T>> {
    using value_t = T;

    value_t k; // stiffness
    value_t relShearStiffness; // relative shear stiffness

    constexpr BaraffWitkinModel() noexcept = default;
    constexpr BaraffWitkinModel(value_t k, value_t relShearStiffness) noexcept 
    : k{k}, relShearStiffness{relShearStiffness}
    {}

    template <class MatT, enable_if_mat_t_dim<MatT> = 0>
    constexpr typename MatT::value_type psi_F(const MatT &F) const noexcept
    {
        auto f0 = zs::col(F, 0);
        auto f1 = zs::col(F, 1);
        auto f0Norm = zs::sqrt(f0.l2NormSqr());
        auto f1Norm = zs::sqrt(f1.l2NormSqr());
        auto Estretch = k * (zs::sqr(f0Norm - 1) + zs::sqr(f1Norm - 1));
        auto Eshear = (k * relShearStiffness) * zs::sqr(f0.dot(f1));
        return Estretch + Eshear;
    }
    template <class MatT, enable_if_mat_t_dim<MatT> = 0>
    constexpr auto dpsi_dF(const MatT &F) const noexcept
    {
        constexpr int dim = MatT::template range_t<0>::value;
        auto f0 = zs::col(F, 0);
        auto f1 = zs::col(F, 1);
        auto f0Norm = zs::sqrt(f0.l2NormSqr());
        auto f1Norm = zs::sqrt(f1.l2NormSqr());
        auto f0Tf1 = f0.dot(f1);
        zs::vec<T, dim, 2> Pstretch{}, Pshear{};
        for (int d = 0; d != dim; ++d) {
            Pstretch(d, 0) = 2 * (1 - 1 / f0Norm) * F(d, 0);
            Pstretch(d, 1) = 2 * (1 - 1 / f1Norm) * F(d, 1);
            Pshear(d, 0) = 2 * f0Tf1 * f1(d);
            Pshear(d, 1) = 2 * f0Tf1 * f0(d);
        }
        return k * Pstretch + (k * relShearStiffness) * Pshear;
    }
    template <class MatT, enable_if_mat_t_dim<MatT> = 0>
    constexpr auto d2psi_dF2(const MatT &F) const noexcept
    {
        auto d2E_dF2 = d2psi_stretch_dF2(F) + d2psi_shear_dF2(F);
        constexpr int dim = MatT::template range_t<0>::value;
        for (int di = 0; di < 2 * dim; di++)
            for (int dj = 0; dj < 2 * dim; dj++)
                if (zs::isnan(d2E_dF2(di, dj)))
                    printf("[nan-hess-baraff-witkin] hess(%d, %d) = %f\n", 
                        di, dj, (float)d2E_dF2(di, dj)); 
        return d2E_dF2;
    }
    template <class MatT, enable_if_mat_t_dim<MatT> = 0>
    constexpr auto d2psi_stretch_dF2(const MatT &F) const noexcept
    {
        constexpr int dim = MatT::template range_t<0>::value;
        auto H = zs::vec<T, 2*dim, 2*dim>::zeros();
        const zs::vec<T, 2> u{1, 0};
        const zs::vec<T, 2> v{0, 1};
        const T I5u = (F * u).l2NormSqr();
        const T I5v = (F * v).l2NormSqr();
        const T invSqrtI5u = (T)1 / zs::sqrt(I5u);
        const T invSqrtI5v = (T)1 / zs::sqrt(I5v);

        for (int i = 0; i < dim; ++i) {
            H(i, i) = zs::max(1 - invSqrtI5u, (T)0);
            H(i + dim, i + dim) = zs::max(1 - invSqrtI5v, (T)0);
        }

        const auto fu = col(F, 0).normalized();
        const T uCoeff = (1 - invSqrtI5u >= 0) ? invSqrtI5u : (T)1;
        for (int i = 0; i != dim; ++i)
            for (int j = 0; j != dim; ++j)
                H(i, j) += uCoeff * fu(i) * fu(j);

        const auto fv = col(F, 1).normalized();
        const T vCoeff = (1 - invSqrtI5v >= 0) ? invSqrtI5v : (T)1;
        for (int i = 0; i != dim; ++i)
            for (int j = 0; j != dim; ++j)
                H(i + dim, j + dim) += vCoeff * fv(i) * fv(j);

        H *= k;
        return H;
    }
    template <class MatT, enable_if_mat_t_dim<MatT> = 0>
    constexpr auto d2psi_shear_dF2(const MatT &F) const noexcept
    {
        constexpr int dim = MatT::template range_t<0>::value;
        constexpr int _2dim = 2 * dim;
        using mat_t = zs::vec<T, _2dim, _2dim>;
        auto H = mat_t::zeros();
        const zs::vec<T, 2> u{1, 0};
        const zs::vec<T, 2> v{0, 1};
        const T I6 = (F * u).dot(F * v);
        const T signI6 = I6 >= 0 ? 1 : -1;

        for (int i = 0; i < dim; ++i) {
            H(i + dim, i) = H(i, i + dim) = (T)1;
        }

        const auto g_ = F * (dyadic_prod(u, v) + dyadic_prod(v, u));
        zs::vec<T, _2dim> g{};
        for (int j = 0, offset = 0; j != 2; ++j) {
            for (int i = 0; i != dim; ++i)
                g(offset++) = g_(i, j);
        }

        const T I2 = F.l2NormSqr();
        const T lambda0 = (T)0.5 * (I2 + zs::sqrt(I2 * I2 + (T)12 * I6 * I6));

        const zs::vec<T, _2dim> q0 = (I6 * H * g + lambda0 * g).normalized();

        auto t = mat_t::identity();
        t = 0.5 * (t + signI6 * H);

        const zs::vec<T, _2dim> Tq = t * q0;
        const auto normTq = Tq.l2NormSqr();

        mat_t d2PdF2 =
            zs::abs(I6) * (t - (dyadic_prod(Tq, Tq) / normTq)) + lambda0 * (dyadic_prod(q0, q0));
        d2PdF2 *= k * relShearStiffness;
        return d2PdF2;
    }
};

using IsotropicElasticModel_ = variant_wrap_floating_point<StVKWithHenckyModel, NeoHookeanModel>::type;
using AnisotropicElasticModel_ = variant_wrap_floating_point<BaraffWitkinModel>::type;
using Codim2ElasticModel_ = variant_wrap_floating_point<BaraffWitkinModel>::type;
using Codim3ElasticModel_ = variant_wrap_floating_point<StVKWithHenckyModel, NeoHookeanModel>::type;
template<class T>
using ElasticModel = std::variant<std::monostate, StVKWithHenckyModel<T>, NeoHookeanModel<T>, BaraffWitkinModel<T>>;
using ElasticModel_ = variant_wrap_floating_point<StVKWithHenckyModel, NeoHookeanModel, BaraffWitkinModel>::type;

template <class T>
constexpr bool is_isotropic_elastic_model_v = is_variant_member_v<T, IsotropicElasticModel_>;

template <class T>
constexpr bool is_anisotropic_elastic_model_v = is_variant_member_v<T, AnisotropicElasticModel_>;

template <class T>
constexpr int is_codim2_elastic_model_v = is_variant_member_v<T, Codim2ElasticModel_>;

template <class T>
constexpr int is_codim3_elastic_model_v = is_variant_member_v<T, Codim3ElasticModel_>;

template <class T>
constexpr int elastic_model_codim = is_codim2_elastic_model_v<T> ? 2 : (is_codim3_elastic_model_v<T> ? 3 : 0);

template <class T>
constexpr bool is_elastic_model_v = is_variant_member_v<T, ElasticModel_>;

template <class T>
constexpr bool is_soft_model_v = is_elastic_model_v<T>; // || is_plastic_model_v<T>;


// matrix and vector type templates for affine body dynamics
template <class MatT>
using enable_if_mat_t_aff = zs::enable_if_all<MatT::dim == 2, MatT::template range_t<0>::value <= 12, MatT::template range_t<1>::value <= 12, std::is_floating_point_v<typename MatT::value_type>>;

template <class VecT>
using enable_if_vec_t_aff = zs::enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 12, std::is_floating_point_v<typename VecT::value_type>>;

template <class MatT, enable_if_mat_t_aff<MatT> = 0>
using VecFlatT = zs::vec<typename MatT::value_type, MatT::template range_t<0>::value * MatT::template range_t<1>::value>; 
template <class MatT, enable_if_mat_t_aff<MatT> = 0>
using MatFlatT = zs::vec<typename MatT::value_type, VecFlatT<MatT>::template range_t<0>::value, VecFlatT<MatT>::template range_t<0>::value>;

// affine body constitutive model w.r.t. linear component A
template <class Model> struct AffineLinearModel {
    using model_t = Model;

    // energy psi(A)
    template <class MatT, enable_if_mat_t_aff<MatT> = 0>
    constexpr typename MatT::value_type energy(const MatT &A) const noexcept
    {
        return static_cast<const Model*>(this)->energy(A);
    }
    // gradient dpsi/dA(A) and hessian d2psi/dA2(A)
    template <class MatT, enable_if_mat_t_aff<MatT> = 0>
    constexpr auto gradientAndHessian(const MatT &A) const noexcept
    {
        return static_cast<const Model*>(this)->gradientAndHessian(A);
    }
    constexpr auto getParams() const noexcept { return static_cast<const Model*>(this)->getParams(); }
};

// orthogonal constitutive model in Affine Body Dynamics
template <class T> struct OrthogonalModel : public AffineLinearModel<OrthogonalModel<T>> {
    using value_t = T;

    value_t E; // stiffness parameter

    OrthogonalModel() noexcept = default;
    constexpr OrthogonalModel(value_t E) noexcept : E{E} {}

    // energy psi(A)
    template <class MatT, enable_if_mat_t_aff<MatT> = 0>
    constexpr typename MatT::value_type energy(const MatT &A) const noexcept
    {
        return (A * A.transpose() - MatT::identity()).l2NormSqr() * E; 
        // attention: dt^2 * vol remains to be multiplied
    }
    // gradient dpsi/dA(A) and hessian d2psi/dA2(A)
    template <class MatT, enable_if_mat_t_aff<MatT> = 0>
    constexpr auto gradientAndHessian(const MatT &A) const noexcept
    {
        VecFlatT<MatT> grad; 
        MatFlatT<MatT> hess; 
        ortho_grad_hess_func(A.data(), grad.data(), hess.data()); 
        zs::make_pd(hess);
        auto coef = E; 
        hess *= coef, grad *= -coef; 
        return std::tuple{grad, hess};
        // attention: dt^2 * vol remains to be multiplied
    }
    constexpr auto getParams() const noexcept { return std::tuple{E}; }
};

template<class T>
using AffineModel = std::variant<std::monostate, OrthogonalModel<T>>;
using AffineModel_ = variant_wrap_floating_point<OrthogonalModel>::type;

template <class T>
constexpr bool is_affine_model_v = is_variant_member_v<T, AffineModel_>;

template <class T>
constexpr bool is_rigid_model_v = is_affine_model_v<T>; // || ...

template <class T, BodyType BodyType_v> struct ConstitutiveModel {};
template <class T> struct ConstitutiveModel<T, BodyType::Soft>
{
    enum struct ElasticModelType 
    { 
        None,
        StVKWithHencky,
        NeoHookean,
        BaraffWitkin
    };
    enum struct PlasticModelType 
    { 
        None,
        VonMises
    };
    using value_t = T;

    ElasticModel<T> elasticModel;
    // TODO: add plasticity model

    constexpr bool hasElasticModel() const noexcept { return !std::holds_alternative<std::monostate>(elasticModel); }
    constexpr ElasticModelType getElasticModelType() const noexcept { return static_cast<ElasticModelType>(elasticModel.index()); }
    template <class ModelT, std::enable_if_t<is_soft_model_v<ModelT>, int> = 0>
    constexpr void setModel(const ModelT& model) noexcept
    {
        if constexpr (is_elastic_model_v<ModelT>)
            elasticModel = model;
    }
    template <class ModelT, std::enable_if_t<is_soft_model_v<ModelT>, int> = 0>
    constexpr void setModel(ModelT&& model) noexcept
    {
        if constexpr (is_elastic_model_v<ModelT>)
            elasticModel = model;
    }
};
template <class T> struct ConstitutiveModel<T, BodyType::Rigid>
{
    enum struct AffineModelType 
    { 
        None,
        Orthogonal 
    };
    using value_t = T;

    AffineModel<T> affineModel;

    constexpr bool hasAffineModel() const noexcept { return !std::holds_alternative<std::monostate>(affineModel); }
    constexpr AffineModelType getAffineModelType() const noexcept { return static_cast<AffineModelType>(affineModel.index()); }
    template <class ModelT, std::enable_if_t<is_rigid_model_v<ModelT>, int> = 0>
    constexpr void setModel(ModelT const& model) noexcept
    {
        if constexpr (is_affine_model_v<ModelT>)
            affineModel = model;
    }
    template <class ModelT, std::enable_if_t<is_rigid_model_v<ModelT>, int> = 0>
    constexpr void setModel(ModelT&& model) noexcept
    {
        if constexpr (is_affine_model_v<ModelT>)
            affineModel = model;
    }
};
} // namespace tacipc