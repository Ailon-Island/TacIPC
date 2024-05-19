#include <tacipc/solver/Solver.cuh>
#if !(s_enableInModelAffineEnergy) || (s_enableMatlabGradHessConversion)
#include <tacipc/gradHessMATLAB.cuh>
#include <tacipc/gradHessConversion.hpp>
#endif // !(s_enableInModelAffineEnergy) || (s_enableMatlabGradHessConversion)
#include <zensim/physics/ConstitutiveModel.hpp>
#if !(s_enableInModelAffineEnergy)
#include <zensim/math/matrix/Eigen.hpp>
#include <zensim/math/matrix/SVD.hpp>
#endif // !(s_enableInModelAffineEnergy)
#if s_enableMollification
#include <zensim/geometry/SpatialQuery.hpp>
#endif // s_enableMollification
#if s_enableFriction
#include <zensim/geometry/Friction.hpp>
#endif // s_enableFriction
// InertialEnergy<T>, AffineEnergy<T>, SoftEnergy<T>, BarrierEnergy<T>

#define LOG_ENERGY 1

namespace tacipc
{
/// Inertial energy
template <class T>
T ABDSolver::InertialEnergy<T>::energy(ABDSolver &solver, pol_t &pol)
{
    static_assert(
        std::is_same_v<T, ABDSolver::T>,
        "The value type of InertialEnergy does not match the solver's.");

    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using value_t = T;
    using vec12 = typename ABDSolver::vec12;
    using mat12 = typename ABDSolver::mat12;

    auto &es = solver.temp;
    T E = 0;

    // for rigid bodies
    auto &rbData = solver.rbData;
    es.resize(count_warps(rbData.size()));
    es.reset(0);

    pol(range(rbData.size()),
        [rbData = view<space>(rbData),
         dt = solver.dt,
         bodyNum = rbData.size(),
         es = view<space>(es, false_c, "es_rigid_inertial")] __device__(int bi) mutable
        {
            int isBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, bi));
            T e = 0;
            if (!isBC)
            {
                mat12 M = rbData.pack(RBProps::M, bi);
                vec12 q = rbData.pack(RBProps::qn, bi);
                vec12 q_tilde = rbData.pack(RBProps::qTilde, bi);
                auto extf = rbData.pack(RBProps::extf, bi) +
                            rbData.pack(RBProps::grav, bi);
                e = (T)0.5 * ((q - q_tilde) * M).dot(q - q_tilde) -
                    (T)(dt * dt) * q.dot(extf);
            }
            reduce_to(bi, bodyNum, e, es[bi / 32]);
        });
    E += reduce(pol, es);

    // for soft bodies
    auto &vData = solver.vData;
    es.resize(count_warps(SoftBodyHandle::vNum));
    es.reset(0);

    pol(range(SoftBodyHandle::vNum),
        [vData = view<space>(vData),
         dt = solver.dt,
         sb_vOffset = RigidBodyHandle::vNum,
         n = SoftBodyHandle::vNum,
         es = view<space>(es, false_c, "es_soft_inertial")] __device__(int id)
        {
            int vi = sb_vOffset + id;
            int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi));
            T e = 0;
            if (!isBC)
            {
                // auto m = zs::sqr(vData(VProps::ws, vi));
                auto m = vData(VProps::m, vi);
                auto x = vData.pack(VProps::xn, vi);
                auto extf = vData.pack(VProps::extf, vi);
                e = (T)0.5 * m * (x - vData.pack(VProps::xTilde, vi)).l2NormSqr() -
                    extf.dot(x) * dt * dt;
            }
            reduce_to(id, n, e, es[id / 32]);
        });
    E += reduce(pol, es);

#if LOG_ENERGY
    fmt::print("Inertial energy = {}\n", E);
#endif

    return E;
}

template <class T>
void ABDSolver::InertialEnergy<T>::addGradientAndHessian(ABDSolver &solver,
                                                         pol_t &pol)
{
    static_assert(
        std::is_same_v<T, ABDSolver::T>,
        "The value type of InertialEnergy does not match the solver's.");

    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using value_t = T;
    using mat3 = typename ABDSolver::mat3;
    using vec12 = typename ABDSolver::vec12;
    using mat12 = typename ABDSolver::mat12;

    auto &rbData = solver.rbData;
    auto &vData = solver.vData;
    auto &dofData = solver.dofData;
    auto &hess = solver.sysHess;

    // for rigid bodies
    pol(range(rbData.size()),
        [rbData = view<space>(rbData), dofData = view<space>(dofData),
         hess = port<space>(hess), 
         dt = solver.dt] __device__(int rbi) mutable
        {
            int isBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, rbi));
            // auto M = mat12::identity() * limits<T>::epsilon();
            auto M = mat12::zeros();
            if (!isBC)
            {
                auto qn = rbData.pack(RBProps::qn, rbi);
                auto q_tilde = rbData.pack(RBProps::qTilde, rbi);
                M = rbData.pack(RBProps::M, rbi);

                auto extf = rbData.pack(RBProps::extf, rbi) +
                            rbData.pack(RBProps::grav, rbi);

                auto grad = extf * dt * dt - M * (qn - q_tilde);
                for (int d = 0; d < 12; d++)
                    dofData(DOFProps::grad, rbi * 12 + d) += grad(d);
            }

            // if (computeHessian)
            {
                if (isnan(M.norm()))
                {
                    printf("M is nan at rbi = %d\n", rbi);
                }
                hess.addRigidHessNoTile(rbi, M, true, rbData);
#if !s_enableAutoPrecondition
                printf("add inertial precondition for rigid bodies\n");
                for (int k = 0; k < 4; k++)
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++)
                            rbData(RBProps::Pre, k * 9 + 3 * i + j, rbi) += M(k * 3
                            + i, k * 3 + j);
#endif
            }
        });

    // for soft bodies
    pol(range(SoftBodyHandle::vNum),
        [vData = view<space>(vData, false_c, "vData"), dofData = view<space>(dofData, false_c, "dofData"),
         hess = port<space>(hess), 
         sb_vOffset = RigidBodyHandle::vNum, dt = solver.dt,
         rbDofs = RigidBodyHandle::bodyNum * 12] __device__(int id) mutable
        {
            int vi = sb_vOffset + id;
            // auto H = mat3::identity() * limits<T>::epsilon();
            auto H = mat3::zeros();
            int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi));
            if (!isBC)
            {
                // auto m = zs::sqr(vData(VProps::ws, vi));
                auto m = vData(VProps::m, vi);
                auto grad =
                    -m * (vData.pack(VProps::xn, vi) - vData.pack(VProps::xTilde, vi)) +
                    vData.pack(VProps::extf, vi) * dt * dt;
                // printf("Soft vert inertial grad[%d]: %f %f %f\n", id, (float)grad(0), (float)grad(1), (float)grad(2));
                for (int d = 0; d < 3; d++)
                    dofData(DOFProps::grad, id * 3 + rbDofs + d) += grad(d);
                {
                    for (int i = 0; i < 3; i++)
                    {
#if !s_enableAutoPrecondition
                        vData(VProps::Pre, 3 * i + i, vi) += m;
#endif

                        H(i, i) = m;
                        if (zs::isnan(m))
                        {
                            printf("m = %f is nan at vi = %d!!!!!!\n", (float)m,
                                vi);
                        }
                    }
                }
            }
            // printf("Vertex inertial hess[%d] norm: %f\n", id, (float)H.norm());
            hess.addSoftHessNoTile(id, H, true, vData, sb_vOffset);
        });
}

template <class T>
T ABDSolver::AffineEnergy<T>::energy(ABDSolver &solver, pol_t &pol)
{
    static_assert(
        std::is_same_v<T, ABDSolver::T>,
        "The value type of AffineEnergy does not match the solver's.");

    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using value_t = T;
    using vec12 = typename ABDSolver::vec12;

    auto &es = solver.temp;
    T E = 0;

    // for rigid bodies (only)
    auto &rbData = solver.rbData;
    es.resize(count_warps(rbData.size()));
    es.reset(0);

#if s_enableInModelAffineEnergy
    pol(range(rbData.size()),
        [rbData = view<space>(rbData),
         bodyNum = rbData.size(),
         es = view<space>(es), dt = solver.dt,
         rigidBodies = solver.rigidBodies] __device__(int bi) mutable
        {
            int isBC = reinterpret_bits<Ti>(rbData(isBCTag, bi));
            T e = 0;
            if (!isBC)
            {
                // T E = rbData(ETag, bi);
                vec12 q = rbData.pack(RBProps::qn, bi);
                auto A = q2A(q);
                auto vol = rbData(RBProps::vol, bi);
                auto &model = rigidBodies[bi].model();
                std::visit(
                    [&e, &A](auto &model)
                    {
                        if constexpr (is_affine_model_v<decltype(model)>)
                            e = model.energy(A);
                    },
                    model.affineModel);
                e *= (T)(dt * dt * vol);
            }
            reduce_to(bi, bodyNum, e, es[bi / 32]);
        });
#else
    pol(range(rbData.size()),
        [rbData = view<space>(rbData),
         bodyNum = rbData.size(),
         es = view<space>(es, false_c, "es_orthogonal"), dt = solver.dt] __device__(int bi) mutable
        {
            int isBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, bi));
            T e = 0;
            if (!isBC)
            {
                // T E = rbData(ETag, bi);
                vec12 q = rbData.pack(RBProps::qn, bi);
                auto A = q2A(q);
                auto E = rbData(RBProps::E, bi);
                auto vol = rbData(RBProps::vol, bi);

                e = (A * A.transpose() - mat3::identity()).l2NormSqr() *
                    (T)(dt * dt * vol * E);
            }
            reduce_to(bi, bodyNum, e, es[bi / 32]);
        });
#endif

    E += reduce(pol, es);

#if LOG_ENERGY
    fmt::print("Affine energy = {}\n", E);
#endif 

    return E;
}

template <class T>
void ABDSolver::AffineEnergy<T>::addGradientAndHessian(ABDSolver &solver,
                                                       pol_t &pol)
{
    static_assert(
        std::is_same_v<T, ABDSolver::T>,
        "The value type of AffineEnergy does not match the solver's.");

    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using value_t = T;
    using vec9 = vec<T, 9>;
    using mat9 = vec<T, 9, 9>;
    using vec12 = typename ABDSolver::vec12;
    using mat12 = typename ABDSolver::mat12;

    auto &rbData = solver.rbData;
    auto &dofData = solver.dofData;
    auto &hess = solver.sysHess;

#if s_enableInModelAffineEnergy
    pol(range(rbData.size()),
        [rbData = view<space>(rbData), dofData = view<space>(dofData),
         sysHess = port<space>(hess), 
         dt = solver.dt,
         rigidBodies = solver.rigidBodies] __device__(int rbi) mutable
        {
            int isBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, rbi));
            if (isBC)
                return; // skip kinematic objects

            auto q = rbData.pack(RBProps::qn, rbi);
            auto A = q2A(q);

            auto &model = rigidBodies[rbi].model();
            vec9 AGrad;
            mat9 AHess;
            std::visit(
                [&A, &AGrad, &AHess](auto &&model)
                {
                    if constexpr (is_affine_model_v<decltype(model)>)
                        std::tie(AGrad, AHess) = model.gradientAndHessian(A);
                },
                model.affineModel);

            // auto E = rbData("E", rbi);
            auto vol = rbData(RBProps::vol, rbi);
            auto coef = vol * dt * dt;
            AHess *= coef, AGrad *= coef;
            for (int d = 0; d < 9; d++)
                dofData(DOFProps::grad, rbi * 12 + 3 + d) += AGrad(d);
            // if (includeHessian)
            {
                auto qhess = Ahess2qhess(AHess);
                hess.addRigidHessNoTilerbi, qsysHess);
                for (int k = 0; k < 3; k++)
                    for (int di = 0; di < 3; di++)
                        for (int dj = 0; dj < 3; dj++)
                            rbData(RBProps::Pre, (k + 1) * 9 + 3 * di + dj, rbi) +=
                                AHess(k * 3 + di, k * 3 + dj);
            }
        });
#else
    pol(range(rbData.size()),
        [rbData = view<space>(rbData), dofData = view<space>(dofData),
         hess = port<space>(hess), 
         dt = solver.dt] __device__(int rbi) mutable
        {
            int isBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, rbi));
            if (isBC)
                return; // skip kinematic objects

            auto q = rbData.pack(RBProps::qn, rbi);
            // auto A = q2A(q);
            auto E = rbData(RBProps::E, rbi);
            auto vol = rbData(RBProps::vol, rbi);
            vec9 AGrad;
            mat9 AHess;

            ortho_grad_hess_func(q.data() + 3, AGrad.data(), AHess.data());
            zs::make_pd(AHess);
            auto coef = E * vol * dt * dt;
            AHess *= coef, AGrad *= -coef;
            for (int d = 0; d < 9; d++)
                dofData(DOFProps::grad, rbi * 12 + 3 + d) += AGrad(d);
            // if (includeHessian)
            {
                auto qhess = Ahess2qhess(AHess);
                if (isnan(AHess.norm()))
                {
                    printf("Affine hess is nan at rbi = %d\n", rbi);
                }
                hess.addRigidHessNoTile(rbi, qhess, true, rbData);
#if !s_enableAutoPrecondition
                printf("add affine precondition for rigid bodies\n");
                for (int k = 0; k < 3; k++)
                    for (int di = 0; di < 3; di++)
                        for (int dj = 0; dj < 3; dj++)
                            rbData(RBProps::Pre, (k + 1) * 9 + 3 * di + dj, rbi) +=
                            AHess(k * 3 + di, k * 3 + dj);
#endif
            }
        });
#endif
}

template <class T>
T ABDSolver::SoftEnergy<T>::energy(ABDSolver &solver, pol_t &pol)
{
    static_assert(
        std::is_same_v<T, ABDSolver::T>,
        "The value type of AffineEnergy does not match the solver's.");

    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using value_t = T;

    T E = 0;

    for (auto &softBody : solver.softBodies)
    {
        auto &model = softBody.model();
        if (model.hasElasticModel())
        {
            E += std::visit([&](auto &model) { return elasticEnergy(solver, pol, softBody, model); },
                model.elasticModel);
        }
    }
#if LOG_ENERGY
    fmt::print("Soft energy = {}\n", E);
#endif 
    return E;
}

template <class T>
void ABDSolver::SoftEnergy<T>::addGradientAndHessian(ABDSolver &solver,
                                                         pol_t &pol)
{
    static_assert(
        std::is_same_v<T, ABDSolver::T>,
        "The value type of AffineEnergy does not match the solver's.");

    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using value_t = T;

    for (auto &softBody : solver.softBodies)
    {
        auto &model = softBody.model();
        if (model.hasElasticModel())
        {
            std::visit(
                [&](auto &model) { 
                    addElasticGradientAndHessian(solver, pol, softBody, model); 
                },
                model.elasticModel);
        }
    }
}

template <class T> template <class Model>
T ABDSolver::SoftEnergy<T>::elasticEnergy(ABDSolver &solver, pol_t &pol, SoftBodyHandle &softBody, const Model &model)
{
    if constexpr (!std::is_same_v<Model, std::monostate>)
    {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        auto &es = solver.temp;

        auto &vData = solver.vData;

        auto &dt = solver.dt;

        auto &eles = softBody.elems();
        es.resize(count_warps(eles.size()));
        es.reset(0);

        if constexpr (elastic_model_codim<Model> == 2)
        {
            pol(range(eles.size()),
                [eles = proxy<space>(eles), vData = view<space>(vData), es = view<space>(es), 
                IBTag = eles.getPropertyOffset("IB"),
                indsTag = eles.getPropertyOffset("inds"),
                volTag = eles.getPropertyOffset("vol"),
                model = model,
                vOffset = softBody.voffset(), n = eles.size()] __device__(int ei) mutable {
                    auto IB = eles.template pack<2, 2>(IBTag, ei);
                    auto inds = eles.template pack<3>(indsTag, ei).template reinterpret_bits<Ti>() + vOffset;
                    auto vole = eles(volTag, ei);

                    auto F = deformation_gradient(vData, IB, inds);
                    T e = model.energy(F) * vole;

                    reduce_to(ei, n, e, es[ei / 32]);
                });
            return (reduce(pol, es) * dt * dt);
        }
        else if constexpr (elastic_model_codim<Model> == 3)
        {
            pol(zs::range(eles.size()),
                [vData = view<space>(vData), eles = view<space>(eles), 
                es = view<space>(es), 
                IBTag = eles.getPropertyOffset("IB"),
                indsTag = eles.getPropertyOffset("inds"),
                volTag = eles.getPropertyOffset("vol"),
                model, 
                vOffset = softBody.voffset(), n = eles.size()] __device__(int ei) mutable {
                    auto IB = eles.template pack<3, 3>(IBTag, ei);
                    auto inds = eles.template pack<4>(indsTag, ei).template reinterpret_bits<Ti>() + vOffset;
                    auto vole = eles(volTag, ei);
                    // mat3 F = deformation_gradient(vData, IB, inds);

                    vec3 xs[4] = {vData.pack(VProps::xn, inds[0]), vData.pack(VProps::xn, inds[1]),
                                  vData.pack(VProps::xn, inds[2]), vData.pack(VProps::xn, inds[3])};

                    mat3 F{};
                    {
                        auto x1x0 = xs[1] - xs[0];
                        auto x2x0 = xs[2] - xs[0];
                        auto x3x0 = xs[3] - xs[0];
                        auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1], x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
                        F = Ds * IB;
                    }

                    T e = model.energy(F) * vole;
                    // printf("Elastic energy[%d]: %f\n", ei, (float)e);
                    reduce_to(ei, n, e, es[ei / 32]);
                });
            return (reduce(pol, es) * dt * dt);
        }
    }
    else 
    {
        return 0;
    }
}

template <class T> template <class Model>
void ABDSolver::SoftEnergy<T>::addElasticGradientAndHessian(ABDSolver &solver, pol_t &pol, SoftBodyHandle &softBody, const Model &model)
{
    if constexpr (!std::is_same_v<Model, std::monostate>)
    {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        auto &rbData = solver.rbData;
        auto &vData = solver.vData;
        auto &dofData = solver.dofData;
        auto &hess = solver.sysHess;

        auto &dt = solver.dt;

        auto &eles = softBody.elems();

        if constexpr (elastic_model_codim<Model> == 2) 
        {
            if (softBody.isBC())
                return;
            pol(zs::range(eles.size()),
                [vData = view<space>(vData),
                 rbData = view<space>(rbData), 
                 eles = view<space>(eles), 
                 IBTag = eles.getPropertyOffset("IB"),
                 indsTag = eles.getPropertyOffset("inds"),
                 volTag = eles.getPropertyOffset("vol"),
                 model, dt = dt,
                 sysHess = port<space>(hess),
                 dofData = view<space>(dofData),
                 sb_vOffset = RigidBodyHandle::vNum,
                 rbDofs = RigidBodyHandle::bodyNum * 12,
                 vOffset = softBody.voffset(),
                 kinematicSatisfied = solver.kinematicSatisfied
                 ] __device__(int ei) mutable {
                    auto IB = eles.template pack<2, 2>(IBTag, ei);
                    auto inds = eles.template pack<3>(indsTag, ei).template reinterpret_bits<Ti>() + vOffset;
                    auto vole = eles(volTag, ei);
                    auto F = deformation_gradient(vData, IB, inds);    

                    for (int di = 0; di < 2; di++)
                        for (int dj = 0; dj < 2; dj++)
                        {
                            if (zs::isnan(IB(di, dj)))
                                printf("nan!!!! IB(%d, %d) = %f\n", 
                                    di, dj, (float)IB(di, dj)); 
                        }

                    // printf("IB = %f\t%f\n%f\t%f\n", (float)IB(0, 0), (float)IB(0, 1), (float)IB(1, 0), (float)IB(1, 1));
                    // printf("inds = %d\t%d\t%d\n", inds[0], inds[1], inds[2]);
                    // printf("F(%dx%d) = %f\t%f\n%f\t%f\n%f\t%f\n", (int)decltype(F)::template range_t<0>::value, (int)decltype(F)::template range_t<1>::value, (float)F(0, 0), (float)F(0, 1), (float)F(1, 0), (float)F(1, 1), (float)F(2, 0), (float)F(2, 1));
                    auto [grad, hess] = model.gradientAndHessian(F, IB);
                    // for (int d = 0; d < 9; d++)
                    // {
                    //     printf("grad[%d] = %f\n", d, (float)grad(d));
                    // }
                    grad *= vole * dt * dt;
                    hess *= vole * dt * dt; 

                    zs::vec<int, 3> segStart {(inds[0] - sb_vOffset) * 3 + rbDofs,
                                         (inds[1] - sb_vOffset) * 3 + rbDofs,
                                         (inds[2] - sb_vOffset) * 3 + rbDofs};
                    zs::vec<int, 3> segLen {3, 3, 3};
                    zs::vec<int, 3> segIsKin;
                    for (int d = 0; d < 3; d++)
                        segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, inds[d])); 
                    scatterGradientAndHessian(segStart, segLen, segIsKin, grad, hess, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied, true, false);
                });
        }
        else if constexpr (elastic_model_codim<Model> == 3)
        {
            pol(zs::range(eles.size()),
                [rbData = view<space>(rbData),
                vData = view<space>(vData), 
                eles = view<space>(eles), 
                dofData = view<space>(dofData),
                sysHess = port<space>(hess),
                IBTag = eles.getPropertyOffset("IB"),
                indsTag = eles.getPropertyOffset("inds"),
                volTag = eles.getPropertyOffset("vol"),
                model, 
                dt = dt,
                vOffset = softBody.voffset(),
                sb_vOffset = RigidBodyHandle::vNum,
                rbDofs = RigidBodyHandle::bodyNum * 12,
                kinematicSatisfied = solver.kinematicSatisfied
                ] __device__(int ei) mutable {
                    auto IB = eles.template pack<3, 3>(IBTag, ei);
                    auto inds = eles.template pack<4>(indsTag, ei).template reinterpret_bits<Ti>() + vOffset;
                    auto vole = eles(volTag, ei);
                    // mat3 F = deformation_gradient(vData, IB, inds);

                    vec3 xs[4] = {vData.pack(VProps::xn, inds[0]), vData.pack(VProps::xn, inds[1]),
                                  vData.pack(VProps::xn, inds[2]), vData.pack(VProps::xn, inds[3])};

                    mat3 F{};
                    {
                        auto x1x0 = xs[1] - xs[0];
                        auto x2x0 = xs[2] - xs[0];
                        auto x3x0 = xs[3] - xs[0];
                        auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1], x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
                        F = Ds * IB;
                    }

                    // auto [grad, hess] = model.gradientAndHessian(F, IB);
                    // grad = vole * grad * dt * dt;
                    // hess = vole * dt * dt;

                    auto P = model.first_piola(F);
                    auto vecP = flatten(P); // view F as vec in a column-major fashion 
                    auto dFdX = dFdXMatrix(IB);
                    auto dFdXT = dFdX.transpose();
                    auto grad = -vole * (dFdXT * vecP) * dt * dt;

                    auto Hq = model.first_piola_derivative(F, true_c);
                    auto hess = dFdXT * Hq * dFdX * vole * dt * dt;

                    // printf("Elastic grad[%d]: %f %f %f\n", ei, (float)grad(0), (float)grad(1), (float)grad(2));

                    zs::vec<int, 4> segStart {(inds[0] - sb_vOffset) * 3 + rbDofs,
                                              (inds[1] - sb_vOffset) * 3 + rbDofs,
                                              (inds[2] - sb_vOffset) * 3 + rbDofs,
                                              (inds[3] - sb_vOffset) * 3 + rbDofs};
                    zs::vec<int, 4> segLen {3, 3, 3, 3};
                    zs::vec<int, 4> segIsKin;
                    for (int d = 0; d < 4; d++)
                        segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, inds[d])); 
                    // printf("Elastic hess[%d-%d-%d-%d] norm: %f\n", (int)inds[0], (int)inds[1], (int)inds[2], (int)inds[3], (float)hess.norm());
                    scatterGradientAndHessian(segStart, segLen, segIsKin, grad, hess, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied, true, false);
                });
        }
    }
}

template <class T>
T ABDSolver::GroundBarrierEnergy<T>::energy(ABDSolver &solver, pol_t &pol)
{
    static_assert(
        std::is_same_v<T, ABDSolver::T>,
        "The value type of GroundBarrierEnergy does not match the solver's.");

    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using value_t = T;

    auto &es = solver.temp;
    T E = 0;

    auto &vData = solver.vData;
    auto &svData = solver.svData;
    es.resize(count_warps(svData.size()));
    es.reset(0);

    auto activeGap2 = solver.dHat * solver.dHat + 2 * solver.dHat * solver.xi;
    auto xi2 = solver.xi * solver.xi;
    auto thickness2 = activeGap2 + xi2;

    // boundary energy
    pol(range(svData.size()),
        [vData = view<space>(vData), svData = view<space>(svData),
         gn = solver.groundNormal,
         xi2 = xi2,
         activeGap2 = activeGap2,
         thickness2 = thickness2,
         n = svData.size(),
         es = view<space>(es, false_c, "es_boundary")] __device__(int svi) mutable
        {
            int vi = reinterpret_bits<Ti>(svData(SVProps::inds, svi));
            int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi));
            auto x = vData.pack(VProps::xn, vi);
            auto dist = gn.dot(x);
            auto dist2 = dist * dist;
            // printf("dist2[%d] = %.20f, dHat2 = %.20f\n", vi, (float)dist2,
            // (float)dHat2);
            T e = 0;
            if (dist2 < thickness2 && !isBC)
                e = -zs::sqr(dist2 - thickness2) * 
                    zs::log((dist2 - xi2) / activeGap2);
            reduce_to(svi, n, e, es[svi / 32]);
        });
    E += reduce(pol, es) * solver.kappa;
#if LOG_ENERGY
    fmt::print("Ground barrier energy = {:.20f}\n", E);
#endif 

#if s_enableBoundaryFriction
    if (solver.enableBoundaryFriction)
        if (solver.fricMu != 0) {
            es.resize(count_warps(svData.size()));
            es.reset(0);
            pol(range(svData.size()),
                [vData = view<space>(vData),
                svData = view<space>(svData), 
                es = view<space>(es), 
                gn = solver.groundNormal, 
                epsvh = solver.epsv * solver.dt, 
                fricMu = solver.fricMu, n = svData.size()] ZS_LAMBDA(int svi) mutable { 
                    const int vi = svData(SVProps::inds, svi, ti_c); 
                    int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi)); 
                    auto fn = svData(SVProps::fn, svi); 
                    T e = 0; 
                    if (fn != 0 && !isBC) {
                        auto x = vData.pack(VProps::xn, vi);
                        auto dx = x - vData.pack(VProps::xHat, vi);
                        auto relDX = dx - gn.dot(dx) * gn;
                        auto relDXNorm2 = relDX.l2NormSqr();
                        auto relDXNorm = zs::sqrt(relDXNorm2);
                        if (relDXNorm > epsvh) {
                            e = fn * (relDXNorm - epsvh / 2);
                        } else {
                            e = fn * relDXNorm2 / epsvh / 2;
                        }
                    }
                    reduce_to(svi, n, e, es[svi / 32]);
                });
            auto e = reduce(pol, es) * solver.fricMu;
            E += e;
#if LOG_ENERGY
            fmt::print("Ground friction energy = {:.20f}\n", e);
#endif 
        }
#endif

    return E;
}

template <class T>
void ABDSolver::GroundBarrierEnergy<T>::addGradientAndHessian(ABDSolver &solver,
                                                              pol_t &pol)
{
    static_assert(
        std::is_same_v<T, ABDSolver::T>,
        "The value type of GroundBarrierEnergy does not match the solver's.");

    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using value_t = T;
    using vec3 = vec<T, 3>;
    using mat3 = vec<T, 3, 3>;

    auto &vData = solver.vData;
    auto &rbData = solver.rbData;
    auto &svData = solver.svData;
    auto &dofData = solver.dofData;
    auto &hess = solver.sysHess;

    auto xi2 = solver.xi * solver.xi;
    auto activeGap2 = solver.dHat * solver.dHat + 2 * solver.dHat * solver.xi;
    auto thickness2 = activeGap2 + xi2;

    pol(range(svData.size()),
        [vData = view<space>(vData), rbData = view<space>(rbData),
         svData = view<space>(svData), hess = port<space>(hess),
         gn = solver.groundNormal,
         xi2 = xi2,
         activeGap2 = activeGap2,
         thickness2 = thickness2,
         n = svData.size(),
         rbDofs = RigidBodyHandle::bodyNum * 12, rb_vNum = RigidBodyHandle::vNum,
         dt = solver.dt,
         kappa = solver.kappa,
         dofData = view<space>(dofData)] __device__(int svi) mutable
        {
            int vi = reinterpret_bits<Ti>(svData(SVProps::inds, svi));
            int bi = reinterpret_bits<Ti>(vData(VProps::body, vi));
            bool hasRBHess = bi >= 0;
            bool hasSBHess = !hasRBHess;
            auto rbHess = mat12::zeros();
            auto sbHess = mat3::zeros();
            if (hasRBHess) // rigid body
            {
                int rbIsBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, bi));
                if (rbIsBC)
                    hasRBHess = false; // skip kinematic objects
            }
            int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi));
            if (isBC)
            {
                hasRBHess = false;
                hasSBHess = false;
            }
            if (hasRBHess || hasSBHess) // need to compute gradient and hessian
            {
                auto x = vData.pack(VProps::xn, vi);
                auto dist = gn.dot(x);
                auto dist2 = dist * dist;
                auto d2 = dist2 - xi2;
                auto t = dist2 - thickness2;
                if (t < 0)
                {
                    auto g_b =
                        t * zs::log(d2 / activeGap2) * -2 - (t * t) / d2;
                    auto H_b =
                        (zs::log(d2 / activeGap2) * -2.0 - t * 4.0 / d2) +
                        1.0 / (d2 * d2) * (t * t);
                    auto param = 4 * H_b * dist2 + 2 * g_b;
                    if (hasRBHess) // rigid body
                    {
                        // grad
                        auto J = vData.pack(VProps::J, vi);
                        auto grad =
                            -J.transpose() * gn * (kappa * g_b * 2 * dist);
                        auto bi = reinterpret_bits<Ti>(vData(VProps::body, vi));
                        for (int d = 0; d < 12; ++d)
                        {
                            atomic_add(exec_cuda, &dofData(DOFProps::grad, 12 * bi + d),
                                       grad(d));
                            atomic_add(exec_cuda, &rbData(RBProps::contact, d, bi),
                                       grad(d));
                        }

                        // hess
                        if (param > 0)
                        {
                            auto nn = dyadic_prod(gn, gn);
                            rbHess = J.transpose() * (kappa * param) * nn * J;
                            /// hess will be added in the end
#if !s_enableAutoPrecondition
                            // precondtion
                            printf("add ground barrier precondition for rigid bodies\n");
                            for (int k = 0; k < 4; k++)
                                for (int di = 0; di < 3; di++)
                                    for (int dj = 0; dj < 3; dj++)
                                        atomic_add(exec_cuda,
                                        &rbData(RBProps::Pre, k * 9 + di * 3 + dj,
                                        bi), rbHess(k * 3 + di, k * 3 + dj));
#endif
                        }
                        else
                        {
                            hasRBHess = false;
                        }
                    }
                    else // if (hasSBHess), soft body
                    {
                        auto grad = -gn * (kappa * g_b * 2 * dist);
                        for (int d = 0; d < 3; ++d)
                        {
                            atomic_add(
                                exec_cuda,
                                &dofData(DOFProps::grad, rbDofs + (vi - rb_vNum) * 3 + d),
                                grad(d));
                            atomic_add(exec_cuda, &vData(VProps::contact, d, vi), grad(d));
                        }

                        if (param > 0)
                        {
                            auto nn = dyadic_prod(gn, gn);
                            sbHess = (kappa * param) * nn;
                            /// hess will be added in the end
#if !s_enableAutoPrecondition
                            printf("add ground barrier precondition for soft bodies\n");
                            for (int di = 0; di < 3; di++)
                                for (int dj = 0; dj < 3; dj++)
                                    vData(VProps::Pre, di * 3 + dj, vi) += sbHess(di, dj);
#endif
                        }
                        else
                        {
                            hasSBHess = false;
                        }
                    }
                }
            }
            if (hasRBHess && isnan(rbHess.norm()))
            {
                printf("Ground barrier hess is nan at rb %d\n", bi);
            }
            if (hasSBHess && isnan(sbHess.norm()))
            {
                printf("Ground barrier hess is nan at sb v %d\n", vi - rb_vNum);
            }
            hess.addRigidHess(bi, rbHess, hasRBHess, true, rbData);
            hess.addSoftHess(vi - rb_vNum, sbHess, hasSBHess, true, vData, rb_vNum);
        });

#if s_enableBoundaryFriction
    if (solver.enableBoundaryFriction)
        if (solver.fricMu != 0) {
            pol(range(svData.size()), [vData = view<space>(vData), svData = view<space>(svData),
                                    rbData = view<space>(rbData), 
                                    dofData = view<space>(dofData),
                                    hess = port<space>(hess),
                                    epsvh = solver.epsv * solver.dt, gn = solver.groundNormal, fricMu = solver.fricMu,
                                    sb_vOffset = RigidBodyHandle::vNum, rbDofs = RigidBodyHandle::bodyNum * 12] ZS_LAMBDA(int svi) mutable {
                int vi = reinterpret_bits<Ti>(svData(SVProps::inds, svi));
                int bi = reinterpret_bits<Ti>(vData(VProps::body, vi));
                bool hasRBHess = bi >= 0;
                bool hasSBHess = !hasRBHess;
                auto rbHess = mat12::zeros();
                auto sbHess = mat3::zeros();

                if (hasRBHess) // rigid body
                {
                    int rbIsBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, bi));
                    if (rbIsBC)
                        hasRBHess = false; // skip kinematic objects
                }
                int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi));
                if (isBC)
                {
                    hasRBHess = false;
                    hasSBHess = false;
                }
                auto fn = svData(SVProps::fn, svi);
                if (fn == 0) {
                    hasRBHess = false;
                    hasSBHess = false;
                }
                // printf("sv [%d] hasRBHess = %d, hasSBHess = %d\n", svi, (int)hasRBHess, (int)hasSBHess);
                if (hasRBHess || hasSBHess)
                {
                    auto dx = vData.pack(VProps::xn, vi) - vData.pack(VProps::xHat, vi);
                    auto coeff = fn * fricMu;
                    auto relDX = dx - gn.dot(dx) * gn;
                    auto relDXNorm2 = relDX.l2NormSqr();
                    auto relDXNorm = zs::sqrt(relDXNorm2);

                    vec3 sbGrad{};
                    if (relDXNorm2 > epsvh * epsvh)
                        sbGrad = -relDX * (coeff / relDXNorm);
                    else
                        sbGrad = -relDX * (coeff / epsvh);

                    if (relDXNorm2 > epsvh * epsvh) {
                        zs::vec<T, 2, 2> mat{relDX[0] * relDX[0] * -coeff / relDXNorm2 / relDXNorm + coeff / relDXNorm,
                                                relDX[0] * relDX[2] * -coeff / relDXNorm2 / relDXNorm,
                                                relDX[0] * relDX[2] * -coeff / relDXNorm2 / relDXNorm,
                                                relDX[2] * relDX[2] * -coeff / relDXNorm2 / relDXNorm + coeff / relDXNorm};
                        make_pd(mat);
                        sbHess(0, 0) = mat(0, 0);
                        sbHess(0, 2) = mat(0, 1);
                        sbHess(2, 0) = mat(1, 0);
                        sbHess(2, 2) = mat(1, 1);
                    } else {
                        sbHess(0, 0) = coeff / epsvh;
                        sbHess(2, 2) = coeff / epsvh;
                    }

                    if (hasRBHess) // rigid body 
                    {
                        // grad
                        auto J = vData.pack(VProps::J, vi);
                        auto rbGrad = J.transpose() * sbGrad;
                        for (int d = 0; d < 12; d++)
                        {
                            atomic_add(exec_cuda, &dofData(DOFProps::grad, bi * 12 + d), rbGrad(d));
                            // printf("sv [%d] rbGrad[%d] = %f\n", svi, d, (float)rbGrad(d));
                            atomic_add(exec_cuda, &rbData(RBProps::contact, d, bi), rbGrad(d));
                        }

                        // hess
                        rbHess = J.transpose() * sbHess * J;
#if !s_enableAutoPrecondition
                        // Pre
                        printf("add ground friction precondition for rigid bodies\n");
                        for (int k = 0; k < 4; k++)
                            for (int di = 0; di < 3; di++)
                                for (int dj = 0; dj < 3; dj++)
                                atomic_add(exec_cuda,
                                    &rbData(RBProps::Pre, k * 9 + 3 * di + dj, bi),
                                    rbHess(k * 3 + di, k * 3 + dj)); 
#endif
                    }
                    else // soft body
                    {
                        // grad
                        for (int d = 0; d < 3; d++)
                        {
                            atomic_add(exec_cuda, &dofData(DOFProps::grad, (vi - sb_vOffset) * 3 + rbDofs + d), sbGrad(d));
                            atomic_add(exec_cuda, &vData(VProps::contact, d, vi), sbGrad(d));
                        }    

                        // hess
#if !s_enableAutoPrecondition
                        // Pre
                        printf("add ground friction precondition for soft bodies\n");
                        for (int di = 0; di < 3; di++)
                            for (int dj = 0; dj < 3; dj++)
                                vData(VProps::Pre, di * 3 + dj, vi) += sbHess(di, dj); 
#endif
                    }
                }

                hess.addRigidHess(bi, rbHess, hasRBHess, true, rbData);
                hess.addSoftHess(vi - sb_vOffset, sbHess, hasSBHess, true, vData, sb_vOffset);
            });
        }
#endif 
}

#if s_enableContact
template <class T>
void ABDSolver::BarrierEnergy<T>::update(ABDSolver &solver,
                                         ABDSolver::pol_t &pol, bool forGrad)
{
    static_assert(
        std::is_same_v<T, ABDSolver::T>,
        "The value type of BarrierEnergy does not match the solver's.");

    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    solver.findBarrierCollisions(pol, solver.xi);

    if (!forGrad)
        return;
    { // TODO
        if (!solver.useAbsKappaDhat)
            solver.suggestKappa(pol);
        if ((solver.enableFriction || solver.enableBoundaryFriction) && solver.updateBasis && solver.fricMu != 0)
            solver.precomputeFrictions(pol, solver.xi);
    }
}

template <class T>
T ABDSolver::BarrierEnergy<T>::energy(ABDSolver &solver, pol_t &pol)
{
    static_assert(
        std::is_same_v<T, ABDSolver::T>,
        "The value type of BarrierEnergy does not match the solver's.");

    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using value_t = T;

    auto &es = solver.temp;
    T E = 0;

    auto &vData = solver.vData;
    auto &svData = solver.svData;
    auto &PP = solver.PP;
    auto &PE = solver.PE;
    auto &PT = solver.PT;
    auto &EE = solver.EE;
#if s_enableMollification
    auto &PPM = solver.PPM;
    auto &PEM = solver.PEM;
    auto &EEM = solver.EEM;
#endif // s_enableMollification
#if s_enableFriction
    auto &FPP = solver.FPP;
    auto &fricPP = solver.fricPP;
    auto &FPE = solver.FPE;
    auto &fricPE = solver.fricPE;
    auto &FPT = solver.FPT;
    auto &fricPT = solver.fricPT;
    auto &FEE = solver.FEE;
    auto &fricEE = solver.fricEE;
#endif // s_enableFriction

    auto &dHat = solver.dHat;
    auto &xi = solver.xi;
    auto &kappa = solver.kappa;
#if s_enableFriction
    auto &fricMu = solver.fricMu;
    auto &epsv = solver.epsv;
    auto &dt = solver.dt;
#endif // s_enableFriction

    {
        auto activeGap2 = dHat * dHat + 2 * xi * dHat;
        auto numPP = PP.getCount();
        es.resize(count_warps(numPP));
        es.reset(0);
        pol(range(numPP),
            [vData = view<space>(vData), PP = PP.port(),
             es = view<space>(es, false_c, "es_PP"),
             xi2 = xi * xi, activeGap2, n = numPP] __device__(int ppi) mutable
            {
                auto pp = PP[ppi];
                auto x0 = vData.pack(VProps::xn, pp[0]);
                auto x1 = vData.pack(VProps::xn, pp[1]);
                auto dist2 = dist2_pp(x0, x1);
                // printf("%d-th pp dist2 * 1000: %f, pp[0]: %d, pp[1]: %d\n",
                // ppi, (float)(dist2 * 1000), (int)pp[0], (int)pp[1]);
                if (dist2 < activeGap2 * 0.01)
                    printf("dist very small!\n");
                if (dist2 < xi2)
                    printf("dist already smaller than xi!\n");

                auto I5 = dist2 / activeGap2;
                auto lenE = (activeGap2 * I5 - activeGap2);
                auto e = -lenE * lenE * zs::log(I5);
                reduce_to(ppi, n, e, es[ppi / 32]);
            });
        E += reduce(pol, es) * kappa;

        auto numPE = PE.getCount();
        // printf("numPE: %d\n", numPE);
        es.resize(count_warps(numPE));
        es.reset(0);
        pol(range(numPE),
            [vData = view<space>(vData), PE = PE.port(),
             es = view<space>(es, false_c, "es_PE"), 
             xi2 = xi * xi, dHat = dHat, activeGap2,
             n = numPE] __device__(int pei) mutable
            {
                auto pe = PE[pei];
                auto p = vData.pack(VProps::xn, pe[0]);
                auto e0 = vData.pack(VProps::xn, pe[1]);
                auto e1 = vData.pack(VProps::xn, pe[2]);

                auto dist2 = dist2_pe(p, e0, e1);
                // printf("%d-th pe dist2 * 1000: %f, p: %d, e0: %d, e1: %d\n",
                // pei, (float)(dist2 * 1000), (int)pe[0], (int)pe[1],
                // (int)pe[2]);
                if (dist2 < activeGap2 * 0.01)
                    printf("dist very small!\n");
                if (dist2 < xi2)
                    printf("dist already smaller than xi!\n");
                // atomic_add(exec_cuda, &res[0],
                //           zs::barrier(dist2 - xi2, activeGap2, kappa));
                // es[pei] = zs::barrier(dist2 - xi2, activeGap2, (T)1);

                auto I5 = dist2 / activeGap2;
                auto lenE = (activeGap2 * I5 - activeGap2);
                auto e = -lenE * lenE * zs::log(I5);
                reduce_to(pei, n, e, es[pei / 32]);
            });
        E += reduce(pol, es) * kappa;

        auto numPT = PT.getCount();
        // printf("numPT: %d\n", numPT);
        es.resize(count_warps(numPT));
        es.reset(0);
        pol(range(numPT),
            [vData = view<space>(vData), PT = PT.port(),
             es = view<space>(es, false_c, "PT"),
             xi2 = xi * xi, dHat = dHat, activeGap2,
             n = numPT] __device__(int pti) mutable
            {
                auto pt = PT[pti];
                auto p = vData.pack(VProps::xn, pt[0]);
                auto t0 = vData.pack(VProps::xn, pt[1]);
                auto t1 = vData.pack(VProps::xn, pt[2]);
                auto t2 = vData.pack(VProps::xn, pt[3]);

                auto dist2 = dist2_pt(p, t0, t1, t2);
                // printf("%d-th pt dist2 * 1000: %f, body: [%d, %d], p: %d, t0: "
                //        "%d, t1: %d, t2: %d\n",
                //        pti, (float)(dist2 * 1000),
                //        (int)reinterpret_bits<Ti>(vData(VProps::body, pt[0])),
                //        (int)reinterpret_bits<Ti>(vData(VProps::body, pt[1])),
                //        (int)pt[0], (int)pt[1], (int)pt[2], (int)pt[3]);
                if (dist2 < activeGap2 * 0.01)
                    printf("dist very small!\n");
                if (dist2 < xi2)
                    printf("dist already smaller than xi!\n");
                // atomic_add(exec_cuda, &res[0],
                //           zs::barrier(dist2 - xi2, activeGap2, kappa));
                // es[pti] = zs::barrier(dist2 - xi2, activeGap2, (T)1);

                auto I5 = dist2 / activeGap2;
                auto lenE = (activeGap2 * I5 - activeGap2);
                auto e = -lenE * lenE * zs::log(I5);
                reduce_to(pti, n, e, es[pti / 32]);
            });
        E += reduce(pol, es) * kappa;

        auto numEE = EE.getCount();
        es.resize(count_warps(numEE));
        es.reset(0);
        pol(range(numEE),
            [vData = view<space>(vData), EE = EE.port(),
             es = view<space>(es, false_c, "ee_EE"), 
             xi2 = xi * xi, dHat = dHat, activeGap2,
             n = numEE] __device__(int eei) mutable
            {
                auto ee = EE[eei];
                auto ea0 = vData.pack(VProps::xn, ee[0]);
                auto ea1 = vData.pack(VProps::xn, ee[1]);
                auto eb0 = vData.pack(VProps::xn, ee[2]);
                auto eb1 = vData.pack(VProps::xn, ee[3]);

                auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
                // printf("%d-th ee dist2 * 1000: %f, ee: %d, %d, %d, %d\n",
                // eei, (float)(dist2 * 1000), (int)ee[0], (int)ee[1],
                // (int)ee[2], (int)ee[3]);
                if (dist2 < activeGap2 * 0.01)
                    printf("dist very small!\n");
                if (dist2 < xi2)
                    printf("dist already smaller than xi!\n");
                // atomic_add(exec_cuda, &res[0],
                //           zs::barrier(dist2 - xi2, activeGap2, kappa));
                // es[eei] = zs::barrier(dist2 - xi2, activeGap2, (T)1);

                auto I5 = dist2 / activeGap2;
                auto lenE = (activeGap2 * I5 - activeGap2);
                auto e = -lenE * lenE * zs::log(I5);
                reduce_to(eei, n, e, es[eei / 32]);
            });
        // printf("EE energy: %.20f\n", (float)reduce(pol, es) * kappa);
        E += reduce(pol, es) * kappa;

#if s_enableMollification
        auto numEEM = EEM.getCount();
        es.resize(count_warps(numEEM));
        es.reset(0);
        pol(range(numEEM),
            [vData = view<space>(vData), EEM = EEM.port(),
             es = view<space>(es), 
             xi2 = xi * xi, dHat = dHat, activeGap2,
             n = numEEM] __device__(int eemi) mutable
            {
                auto eem = EEM[eemi];
                auto ea0 = vData.pack(VProps::xn, eem[0]);
                auto ea1 = vData.pack(VProps::xn, eem[1]);
                auto eb0 = vData.pack(VProps::xn, eem[2]);
                auto eb1 = vData.pack(VProps::xn, eem[3]);

                auto v0 = ea1 - ea0;
                auto v1 = eb1 - eb0;
                auto c = v0.cross(v1).norm();
                auto I1 = c * c;
                T e = 0;
                if (I1 != 0)
                {
                    auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
                    if (dist2 < xi2)
                        printf("dist already smaller than xi!\n");
                    auto I2 = dist2 / activeGap2;

                    auto rv0 = vData.pack(VProps::x0, eem[0]);
                    auto rv1 = vData.pack(VProps::x0, eem[1]);
                    auto rv2 = vData.pack(VProps::x0, eem[2]);
                    auto rv3 = vData.pack(VProps::x0, eem[3]);
                    T epsX = mollifier_threshold_ee(rv0, rv1, rv2, rv3);
                    e = (2 - I1 / epsX) * (I1 / epsX) *
                        -zs::sqr(activeGap2 - activeGap2 * I2) * zs::log(I2);
                }
                reduce_to(eemi, n, e, es[eemi / 32]);
            });
        E += reduce(pol, es) * kappa;

        auto numPPM = PPM.getCount();
        es.resize(count_warps(numPPM));
        es.reset(0);
        pol(range(numPPM),
            [vData = view<space>(vData), PPM = PPM.port(),
             es = view<space>(es), 
             xi2 = xi * xi, dHat = dHat,
             activeGap2, n = numPPM] __device__(int ppmi) mutable
            {
                auto ppm = PPM[ppmi];

                auto v0 =
                    vData.pack(VProps::xn, ppm[1]) - vData.pack(VProps::xn, ppm[0]);
                auto v1 =
                    vData.pack(VProps::xn, ppm[3]) - vData.pack(VProps::xn, ppm[2]);
                auto c = v0.cross(v1).norm();
                auto I1 = c * c;
                T e = 0;
                if (I1 != 0)
                {
                    auto dist2 = dist2_pp(vData.pack(VProps::xn, ppm[0]),
                                          vData.pack(VProps::xn, ppm[2]));
                    if (dist2 < xi2)
                        printf("dist already smaller than xi!\n");
                    auto I2 = dist2 / activeGap2;

                    auto rv0 = vData.pack(VProps::x0, ppm[0]);
                    auto rv1 = vData.pack(VProps::x0, ppm[1]);
                    auto rv2 = vData.pack(VProps::x0, ppm[2]);
                    auto rv3 = vData.pack(VProps::x0, ppm[3]);
                    T epsX = mollifier_threshold_ee(rv0, rv1, rv2, rv3);
                    e = (2 - I1 / epsX) * (I1 / epsX) *
                        -zs::sqr(activeGap2 - activeGap2 * I2) * zs::log(I2);
                }
                reduce_to(ppmi, n, e, es[ppmi / 32]);
            });
        E += reduce(pol, es) * kappa;

        auto numPEM = PEM.getCount();
        es.resize(count_warps(numPEM));
        es.reset(0);
        pol(range(numPEM),
            [vData = view<space>(vData), PEM = PEM.port(),
             es = view<space>(es),
             xi2 = xi * xi, dHat = dHat,
             activeGap2, n = numPEM] __device__(int pemi) mutable
            {
                auto pem = PEM[pemi];

                auto p = vData.pack(VProps::xn, pem[0]);
                auto e0 = vData.pack(VProps::xn, pem[2]);
                auto e1 = vData.pack(VProps::xn, pem[3]);
                auto v0 = vData.pack(VProps::xn, pem[1]) - p;
                auto v1 = e1 - e0;
                auto c = v0.cross(v1).norm();
                auto I1 = c * c;
                T e = 0;
                if (I1 != 0)
                {
                    auto dist2 = dist2_pe(p, e0, e1);
                    if (dist2 < xi2)
                        printf("dist already smaller than xi!\n");
                    auto I2 = dist2 / activeGap2;

                    auto rv0 = vData.pack(VProps::x0, pem[0]);
                    auto rv1 = vData.pack(VProps::x0, pem[1]);
                    auto rv2 = vData.pack(VProps::x0, pem[2]);
                    auto rv3 = vData.pack(VProps::x0, pem[3]);
                    T epsX = mollifier_threshold_ee(rv0, rv1, rv2, rv3);
                    e = (2 - I1 / epsX) * (I1 / epsX) *
                        -zs::sqr(activeGap2 - activeGap2 * I2) * zs::log(I2);
                }
                reduce_to(pemi, n, e, es[pemi / 32]);
            });
        E += reduce(pol, es) * kappa;
#endif // mollification

#if s_enableFriction
        if (solver.enableFriction)
            if (fricMu != 0) {
                auto numFPP = FPP.getCount();
                es.resize(count_warps(numFPP));
                es.reset(0);
                pol(range(numFPP), [vData = view<space>(vData),
                                    fricPP = view<space>(fricPP),
                                    FPP = FPP.port(), es = view<space>(es),
                                    epsvh = epsv * dt,
                                    n = numFPP] __device__(int fppi) mutable {
                    auto fpp = FPP[fppi];
                    auto p0 =
                        vData.pack(VProps::xn, fpp[0]) - vData.pack(VProps::xHat, fpp[0]);
                    auto p1 =
                        vData.pack(VProps::xn, fpp[1]) - vData.pack(VProps::xHat, fpp[1]);
                    auto basis = fricPP.pack(FPPProps::basis, fppi);
                    auto fn = fricPP(FPPProps::fn, fppi);
                    auto relDX3D = point_point_rel_dx(p0, p1);
                    auto relDX = basis.transpose() * relDX3D;
                    auto relDXNorm2 = relDX.l2NormSqr();
                    auto e = f0_SF(relDXNorm2, epsvh) * fn;
                    reduce_to(fppi, n, e, es[fppi / 32]);
                });
                E += reduce(pol, es) * fricMu;

                auto numFPE = FPE.getCount();
                es.resize(count_warps(numFPE));
                es.reset(0);
                pol(range(numFPE), [vData = view<space>(vData),
                                    fricPE = view<space>(fricPE),
                                    FPE = FPE.port(), es = view<space>(es),
                                    epsvh = epsv * dt,
                                    n = numFPE] __device__(int fpei) mutable {
                    auto fpe = FPE[fpei];
                    auto p =
                        vData.pack(VProps::xn, fpe[0]) - vData.pack(VProps::xHat, fpe[0]);
                    auto e0 =
                        vData.pack(VProps::xn, fpe[1]) - vData.pack(VProps::xHat, fpe[1]);
                    auto e1 =
                        vData.pack(VProps::xn, fpe[2]) - vData.pack(VProps::xHat, fpe[2]);
                    auto basis = fricPE.pack(FPEProps::basis, fpei);
                    auto fn = fricPE(FPEProps::fn, fpei);
                    auto yita = fricPE(FPEProps::yita, fpei);
                    auto relDX3D = point_edge_rel_dx(p, e0, e1, yita);
                    auto relDX = basis.transpose() * relDX3D;
                    auto relDXNorm2 = relDX.l2NormSqr();
                    auto e = f0_SF(relDXNorm2, epsvh) * fn;
                    reduce_to(fpei, n, e, es[fpei / 32]);
                });
                E += reduce(pol, es) * fricMu;

                auto numFPT = FPT.getCount();
                es.resize(count_warps(numFPT));
                es.reset(0);
                pol(range(numFPT), [vData = view<space>(vData),
                                    fricPT = view<space>(fricPT),
                                    FPT = FPT.port(), es = view<space>(es),
                                    epsvh = epsv * dt,
                                    n = numFPT] __device__(int fpti) mutable {
                    auto fpt = FPT[fpti];
                    auto p =
                        vData.pack(VProps::xn, fpt[0]) - vData.pack(VProps::xHat, fpt[0]);
                    auto v0 =
                        vData.pack(VProps::xn, fpt[1]) - vData.pack(VProps::xHat, fpt[1]);
                    auto v1 =
                        vData.pack(VProps::xn, fpt[2]) - vData.pack(VProps::xHat, fpt[2]);
                    auto v2 =
                        vData.pack(VProps::xn, fpt[3]) - vData.pack(VProps::xHat, fpt[3]);
                    auto basis = fricPT.pack(FPTProps::basis, fpti);
                    auto fn = fricPT(FPTProps::fn, fpti);
                    auto betas = fricPT.pack(FPTProps::beta, fpti);
                    auto relDX3D =
                        point_triangle_rel_dx(p, v0, v1, v2, betas[0], betas[1]);
                    auto relDX = basis.transpose() * relDX3D;
                    auto relDXNorm2 = relDX.l2NormSqr();
                    auto e = f0_SF(relDXNorm2, epsvh) * fn;
                    reduce_to(fpti, n, e, es[fpti / 32]);
                });
                E += reduce(pol, es) * fricMu;

                auto numFEE = FEE.getCount();
                es.resize(count_warps(numFEE));
                es.reset(0);
                pol(range(numFEE), [vData = view<space>(vData),
                                    fricEE = view<space>(fricEE),
                                    FEE = FEE.port(), es = view<space>(es),
                                    epsvh = epsv * dt,
                                    n = numFEE] __device__(int feei) mutable {
                    auto fee = FEE[feei];
                    auto e0 =
                        vData.pack(VProps::xn, fee[0]) - vData.pack(VProps::xHat, fee[0]);
                    auto e1 =
                        vData.pack(VProps::xn, fee[1]) - vData.pack(VProps::xHat, fee[1]);
                    auto e2 =
                        vData.pack(VProps::xn, fee[2]) - vData.pack(VProps::xHat, fee[2]);
                    auto e3 =
                        vData.pack(VProps::xn, fee[3]) - vData.pack(VProps::xHat, fee[3]);
                    auto basis = fricEE.pack(FEEProps::basis, feei);
                    auto fn = fricEE(FEEProps::fn, feei);
                    auto gammas = fricEE.pack(FEEProps::gamma, feei);
                    auto relDX3D =
                        edge_edge_rel_dx(e0, e1, e2, e3, gammas[0], gammas[1]);
                    auto relDX = basis.transpose() * relDX3D;
                    auto relDXNorm2 = relDX.l2NormSqr();
                    auto e = f0_SF(relDXNorm2, epsvh) * fn;
                    reduce_to(feei, n, e, es[feei / 32]);
                });
                E += reduce(pol, es) * fricMu;
            }
        
#endif
    }
#if LOG_ENERGY
    fmt::print("Barrier energy: {}\n", E);
#endif 
    return E;
}

template <class T>
void ABDSolver::BarrierEnergy<T>::addGradientAndHessian(ABDSolver &solver,
                                                        pol_t &pol)
{
    static_assert(
        std::is_same_v<T, ABDSolver::T>,
        "The value type of BarrierEnergy does not match the solver's.");

    addBarrierGradientAndHessian(solver, pol);
#if s_enableFriction
    if (solver.enableFriction)
        addFrictionGradientAndHessian(solver, pol);
#endif
}

template <class T>
void ABDSolver::BarrierEnergy<T>::addBarrierGradientAndHessian(ABDSolver &solver, pol_t &pol)
{
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using value_t = T;
    using Vec12View = zs::vec_view<T, zs::integer_sequence<int, 12>>;
    using Vec9View = zs::vec_view<T, zs::integer_sequence<int, 9>>;
    using Vec6View = zs::vec_view<T, zs::integer_sequence<int, 6>>;


    auto &vData = solver.vData;
    auto &rbData = solver.rbData;
    auto &svData = solver.svData;
    auto &dofData = solver.dofData;
    auto &hess = solver.sysHess;

    // auto &gtemp = solver.temp;
    // auto dof = solver.dof();
    // gtemp.resize(dof);
    // pol(range(dof),
    //     [grad = view<space>(gtemp, false_c, "barrier_grad_hess_gtemp"), dofData = view<space>(dofData)] __device__(int d) mutable
    //     { grad(d) = dofData(DOFProps::grad, d); });

    auto &PP = solver.PP;
    auto &PE = solver.PE;
    auto &PT = solver.PT;
    auto &EE = solver.EE;
#if s_enableMollification
    auto &PPM = solver.PPM;
    auto &PEM = solver.PEM;
    auto &EEM = solver.EEM;
#endif // s_enableMollification

    auto &dHat = solver.dHat;
    auto &xi = solver.xi;
    auto &kappa = solver.kappa;

    zs::CppTimer timer;
    T activeGap2 = dHat * dHat + (T)2.0 * xi * dHat;

    auto numPP = PP.getCount();
    fmt::print("nPP: {}, nPE: {}, nPT: {}, nEE: {}\n", PP.getCount(),
               PE.getCount(), PT.getCount(), EE.getCount());
    if (solver.enableMollification)
        fmt::print("nPPM: {}, nPEM: {}, nEEM: {}\n", PPM.getCount(),
                   PEM.getCount(), EEM.getCount());
    pol(range(numPP),
        [vData = view<space>(vData), rbData = view<space>(rbData),
         dofData = view<space>(dofData), rbDofs = RigidBodyHandle::bodyNum * 12,
         rb_vNum = RigidBodyHandle::vNum, sysHess = port<space>(hess),
         PP = PP.port(), 
         xi2 = xi * xi, dHat = dHat,
         activeGap2, kappa = kappa,
         kinematicSatisfied = solver.kinematicSatisfied
         ] __device__(int ppi) mutable
        {
            auto pp = PP[ppi];
            auto x0 = vData.pack(VProps::xn, pp[0]);
            auto x1 = vData.pack(VProps::xn, pp[1]);
            auto v0 = x1 - x0;
            auto Ds = v0;
            auto dis = v0.norm();

            auto vec_normal = -v0.normalized();
            auto target = vec3{0, 1, 0};

            auto vec = vec_normal.cross(target);
            T cos = vec_normal.dot(target);
            auto rotation = mat3::identity();

            auto d_hat_sqrt = dHat;
            if (cos + 1 == 0)
            {
                rotation(0, 0) = -1;
                rotation(1, 1) = -1;
            }
            else
            {
                mat3 cross_vec{0,       -vec[2], vec[1], vec[2], 0,
                               -vec[0], -vec[1], vec[0], 0};
                rotation += cross_vec + cross_vec * cross_vec / (1 + cos);
            }

            auto pos0 = x0 + (d_hat_sqrt - dis) * vec_normal;

            auto rotate_uv0 = rotation * pos0;
            auto rotate_uv1 = rotation * x1;

            auto uv0 = rotate_uv0[1];
            auto uv1 = rotate_uv1[1];

            auto u0 = uv1 - uv0;
            auto Dm = u0;
            auto DmInv = 1 / u0;
            auto F = Ds * DmInv;
            T I5 = F.dot(F);

            auto tmp = F * 2;
            vec3 flatten_pk1 = kappa *
                               -(activeGap2 * activeGap2 * (I5 - 1) *
                                 (1 + 2 * zs::log(I5) - 1 / I5)) *
                               tmp;

            auto PFPx = zs::vec<T, 3, 6>::zeros();
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j)
                {
                    PFPx(i, j) = i == j ? -DmInv : 0;
                    PFPx(i, 3 + j) = i == j ? DmInv : 0;
                }

            auto grad = -PFPx.transpose() * flatten_pk1;
            scatterContactForce(pp, grad, vData);

            // ABD: update q grad instead of x grad
            int bodyA = reinterpret_bits<Ti>(vData(VProps::body, pp[0]));
            int bodyB = reinterpret_bits<Ti>(vData(VProps::body, pp[1]));
            auto J0 = vData.pack(VProps::J, pp[0]);
            auto J1 = vData.pack(VProps::J, pp[1]);

            T lambda0 = kappa * (2 * activeGap2 * activeGap2 *
                                 (6 + 2 * zs::log(I5) - 7 * I5 -
                                  6 * I5 * zs::log(I5) + 1 / I5));
            auto Q0 = F / zs::sqrt(I5);
            auto H = lambda0 * dyadic_prod(Q0, Q0);
            auto hess = PFPx.transpose() * H * PFPx;

            if (bodyA >= 0 && bodyB >= 0)
            {
                int isKinA = reinterpret_bits<Ti>(rbData(VProps::isBC, bodyA));
                int isKinB = reinterpret_bits<Ti>(rbData(VProps::isBC, bodyB));
                zs::vec<int, 2> segStart{bodyA * 12, bodyB * 12};
                zs::vec<int, 2> segLen{12, 12};
                zs::vec<int, 2> segIsKin{isKinA, isKinB};
#if s_enableMatlabGradHessConversion
                zs::vec<T, 24> grad_q;
                zs::vec<T, 24, 24> hess_q;
                auto Jv0 = vData.pack(VProps::JVec, pp[0]);
                auto Jv1 = vData.pack(VProps::JVec, pp[1]);
                grad_hess_pp_AA_conversion(grad.data(), hess.data(),
                                           Jv0.data(), Jv1.data(),
                                           grad_q.data(), hess_q.data());
#else 
                mat6x24 J = mat6x24::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; j++) {
                        J(i, j) = J0(i, j);
                        J(i + 3, j + 12) = J1(i, j);
                    }
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
#endif
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA < 0 && bodyB < 0)
            {
                printf("PP Soft-Soft\n");
                zs::vec<int, 2> segStart{(pp[0] - rb_vNum) * 3 + rbDofs,
                                         (pp[1] - rb_vNum) * 3 + rbDofs};
                zs::vec<int, 2> segLen{3, 3};
                zs::vec<int, 2> segIsKin;
                for (int d = 0; d < 2; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, pp[d]));
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad,
                                          hess, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA >= 0)
            {
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA));
                // 12 + 3
                zs::vec<int, 2> segStart{bodyA * 12,
                                         (pp[1] - rb_vNum) * 3 + rbDofs};
                zs::vec<int, 2> segLen{12, 3};
                zs::vec<int, 2> segIsKin{
                    isKinA, reinterpret_bits<Ti>(vData(VProps::isBC, pp[1]))};
                auto J = zs::vec<T, 6, 15>::zeros();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 12; j++)
                        J(i, j) = J0(i, j);
                for (int i = 0; i < 3; i++)
                    J(3 + i, 12 + i) = 1;
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyB >= 0)
            {
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB));
                // 3 + 12
                zs::vec<int, 2> segStart{(pp[0] - rb_vNum) * 3 + rbDofs,
                                         bodyB * 12};
                zs::vec<int, 2> segLen{3, 12};
                zs::vec<int, 2> segIsKin{
                    reinterpret_bits<Ti>(vData(VProps::isBC, pp[0])), isKinB};
                auto J = zs::vec<T, 6, 15>::zeros();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 12; j++)
                        J(3 + i, 3 + j) = J1(i, j);
                for (int i = 0; i < 3; i++)
                    J(i, i) = 1;
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }
        });

    auto numPE = PE.getCount();
    pol(range(numPE),
        [vData = view<space>(vData), rbData = view<space>(rbData),
         dofData = view<space>(dofData), rbDofs = RigidBodyHandle::bodyNum * 12,
         rb_vNum = RigidBodyHandle::vNum, sysHess = port<space>(hess),
         PE = PE.port(), 
         xi2 = xi * xi, dHat = dHat,
         activeGap2, kappa = kappa,
         kinematicSatisfied = solver.kinematicSatisfied
         ] __device__(int pei) mutable
        {
            auto pe = PE[pei];
            auto p = vData.pack(VProps::xn, pe[0]);
            auto e0 = vData.pack(VProps::xn, pe[1]);
            auto e1 = vData.pack(VProps::xn, pe[2]);

            auto v0 = e0 - p;
            auto v1 = e1 - p;

            zs::vec<T, 3, 2> Ds{v0[0], v1[0], v0[1], v1[1], v0[2], v1[2]};
            auto triangle_normal = v0.cross(v1).normalized();
            auto target = vec3{0, 1, 0};

            auto vec = triangle_normal.cross(target);
            auto cos = triangle_normal.dot(target);

            auto edge_normal = (e0 - e1).cross(triangle_normal).normalized();
            auto dis = (p - e0).dot(edge_normal);

            auto rotation = mat3::identity();
            T d_hat_sqrt = dHat;
            if (cos + 1 == 0.0)
            {
                rotation(0, 0) = -1;
                rotation(1, 1) = -1;
            }
            else
            {
                mat3 cross_vec{0,       -vec[2], vec[1], vec[2], 0,
                               -vec[0], -vec[1], vec[0], 0};
                rotation += cross_vec + cross_vec * cross_vec / (1 + cos);
            }

            auto pos0 = p + (d_hat_sqrt - dis) * edge_normal;

            auto rotate_uv0 = rotation * pos0;
            auto rotate_uv1 = rotation * e0;
            auto rotate_uv2 = rotation * e1;
            auto rotate_normal = rotation * edge_normal;

            using vec2 = zs::vec<T, 2>;
            auto uv0 = vec2(rotate_uv0[0], rotate_uv0[2]);
            auto uv1 = vec2(rotate_uv1[0], rotate_uv1[2]);
            auto uv2 = vec2(rotate_uv2[0], rotate_uv2[2]);
            auto normal = vec2(rotate_normal[0], rotate_normal[2]);

            auto u0 = uv1 - uv0;
            auto u1 = uv2 - uv0;

            using mat2 = zs::vec<T, 2, 2>;
            mat2 Dm{u0(0), u1(0), u0(1), u1(1)};
            auto DmInv = inverse(Dm);

            zs::vec<T, 3, 2> F = Ds * DmInv;
            // T I5 = normal.dot(F.transpose() * F * normal);
            T I5 = (F * normal).l2NormSqr();
            auto nn = dyadic_prod(normal, normal);
            auto fnn = F * nn;
            auto tmp = flatten(fnn) * 2;

            zs::vec<T, 6> flatten_pk1{};
            flatten_pk1 = kappa *
                          -(activeGap2 * activeGap2 * (I5 - 1) *
                            (1 + 2 * zs::log(I5) - 1 / I5)) *
                          tmp;

            zs::vec<T, 6, 9> PFPx = dFdXMatrix(DmInv, wrapv<3>{});

            auto grad = -PFPx.transpose() * flatten_pk1;
            scatterContactForce(pe, grad, vData);

            // ABD: update q grad instead of x grad
            int bodyA = reinterpret_bits<Ti>(vData(VProps::body, pe[0]));
            int bodyB = reinterpret_bits<Ti>(vData(VProps::body, pe[1]));
            auto Jp = vData.pack(VProps::J, pe[0]);
            auto Je0 = vData.pack(VProps::J, pe[1]);
            auto Je1 = vData.pack(VProps::J, pe[2]);
            T lambda0 = kappa * (2 * activeGap2 * activeGap2 *
                                 (6 + 2 * zs::log(I5) - 7 * I5 -
                                  6 * I5 * zs::log(I5) + 1 / I5));
            auto q0 = flatten(fnn) / zs::sqrt(I5);
            auto H = lambda0 * dyadic_prod(q0, q0);
            auto hess = PFPx.transpose() * H * PFPx;

            if (bodyA >= 0 && bodyB >= 0)
            {
                // printf("PE Rigid-Rigid\n");
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA));
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB));
                zs::vec<int, 2> segStart{bodyA * 12, bodyB * 12};
                zs::vec<int, 2> segLen{12, 12};
                zs::vec<int, 2> segIsKin{isKinA, isKinB};
#if s_enableMatlabGradHessConversion
                zs::vec<T, 24> grad_q;
                zs::vec<T, 24, 24> hess_q;
                auto Jv0 = vData.pack(VProps::JVec, pe[0]);
                auto Jv1 = vData.pack(VProps::JVec, pe[1]);
                auto Jv2 = vData.pack(VProps::JVec, pe[2]);
                grad_hess_pe_AA_conversion(grad.data(), hess.data(),
                                           Jv0.data(), Jv1.data(), Jv2.data(),
                                           grad_q.data(), hess_q.data());
#else 
                mat9x24 J = mat9x24::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; ++j) {
                        J(i, j) = Jp(i, j);
                        J(i + 3, j + 12) = Je0(i, j);
                        J(i + 6, j + 12) = Je1(i, j);
                    }
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
#endif
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA < 0 && bodyB < 0)
            {
                printf("PE Soft-Soft\n");
                zs::vec<int, 3> segStart{(pe[0] - rb_vNum) * 3 + rbDofs,
                                         (pe[1] - rb_vNum) * 3 + rbDofs,
                                         (pe[2] - rb_vNum) * 3 + rbDofs};
                zs::vec<int, 3> segLen{3, 3, 3};
                zs::vec<int, 3> segIsKin;
                for (int d = 0; d < 3; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, pe[d]));
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad,
                                          hess, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA >= 0)
            {
                // printf("PE Rigid-Soft\n");
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA));
                // 12 + 3
                zs::vec<int, 3> segStart{bodyA * 12,
                                         (pe[1] - rb_vNum) * 3 + rbDofs,
                                         (pe[2] - rb_vNum) * 3 + rbDofs};
                zs::vec<int, 3> segLen{12, 3, 3};
                zs::vec<int, 3> segIsKin{isKinA, 0, 0};
                for (int d = 1; d < 3; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, pe[d]));
                auto J = zs::vec<T, 9, 18>::zeros();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 12; j++)
                        J(i, j) = Jp(i, j);
                for (int i = 0; i < 6; i++)
                    J(3 + i, 12 + i) = 1;
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyB >= 0)
            {
                // printf("PE Soft-Rigid\n");
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB));
                // 3 + 12
                zs::vec<int, 2> segStart{(pe[0] - rb_vNum) * 3 + rbDofs,
                                         bodyB * 12};
                zs::vec<int, 2> segLen{3, 12};
                zs::vec<int, 2> segIsKin{
                    reinterpret_bits<Ti>(vData(VProps::isBC, pe[0])), isKinB};
                auto J = zs::vec<T, 9, 15>::zeros();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 12; j++)
                        J(3 + i, 3 + j) = Je0(i, j);
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 12; j++)
                        J(6 + i, 3 + j) = Je1(i, j);
                for (int i = 0; i < 3; i++)
                    J(i, i) = 1;
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }
        });

    auto numPT = PT.getCount();
    pol(range(numPT),
        [vData = view<space>(vData), rbData = view<space>(rbData),
         dofData = view<space>(dofData), rbDofs = RigidBodyHandle::bodyNum * 12,
         rb_vNum = RigidBodyHandle::vNum, sysHess = port<space>(hess),
         PT = PT.port(), 
         xi2 = xi * xi, dHat = dHat,
         activeGap2, kappa = kappa,
         kinematicSatisfied = solver.kinematicSatisfied
         ] __device__(int pti) mutable
        {
            auto pt = PT[pti];
            // printf("pt[%d]: %d %d %d %d\n", pti, (int)pt[0], (int)pt[1], (int)pt[2], (int)pt[3]);
            // printf("pt[%d] xi2: %f, dHat: %f, activeGap2: %f, kappa: %f\n", pti, (float)xi2, (float)dHat, (float)activeGap2, (float)kappa);
            auto p = vData.pack(VProps::xn, pt[0]);
            auto t0 = vData.pack(VProps::xn, pt[1]);
            auto t1 = vData.pack(VProps::xn, pt[2]);
            auto t2 = vData.pack(VProps::xn, pt[3]);
            // printf("pt[%d]: (%f, %f, %f) and (%f, %f, %f)-(%f, %f, %f)-(%f, %f, %f)\n", pti, (float)p[0], (float)p[1], (float)p[2], (float)t0[0], (float)t0[1], (float)t0[2], (float)t1[0], (float)t1[1], (float)t1[2], (float)t2[0], (float)t2[1], (float)t2[2]);

            auto v0 = t0 - p;
            auto v1 = t1 - p;
            auto v2 = t2 - p;
            mat3 Ds{v0[0], v1[0], v2[0], v0[1], v1[1], v2[1], v0[2], v1[2], v2[2]};
            auto normal = (t1 - t0).cross(t2 - t0).normalized();
            auto dis = v0.dot(normal);
            auto d_hat_sqrt = dHat;
            zs::vec<T, 9, 12> PDmPx{};
            if (dis > 0) {
                normal = -normal;
            } else {
                dis = -dis;
            }
            auto pos0 = p + normal * (d_hat_sqrt - dis);

            auto u0 = t0 - pos0;
            auto u1 = t1 - pos0;
            auto u2 = t2 - pos0;
            mat3 Dm{u0[0], u1[0], u2[0], u0[1], u1[1], u2[1], u0[2], u1[2], u2[2]};
            auto DmInv = inverse(Dm);
            auto F = Ds * DmInv;
            auto [uu, ss, vv] = math::qr_svd(F);
            auto values = zs::sqr(ss.sum() - 2);
            T I5 = (F * normal).l2NormSqr();

            zs::vec<T, 9> flatten_pk1{};
            {
                auto tmp = flatten(F * dyadic_prod(normal, normal)) * 2;
                flatten_pk1 = kappa * -(activeGap2 * activeGap2 * (I5 - 1) * (1 + 2 * zs::log(I5) - 1 / I5)) * tmp;
            }

            auto PFPx = dFdXMatrix(DmInv, wrapv<3>{});

            auto grad = -PFPx.transpose() * flatten_pk1;
            scatterContactForce(pt, grad, vData);

            // printf("pt grad[%d]: %f %f %f %f %f %f %f %f %f %f %f %f\n", pti, (float)grad[0], (float)grad[1], (float)grad[2], (float)grad[3], (float)grad[4], (float)grad[5], (float)grad[6], (float)grad[7], (float)grad[8], (float)grad[9], (float)grad[10], (float)grad[11]);
            // ABD: update q grad instead of x grad
            int bodyA = reinterpret_bits<Ti>(vData(VProps::body, pt[0]));
            int bodyB = reinterpret_bits<Ti>(vData(VProps::body, pt[1]));
            auto Jp = vData.pack(VProps::J, pt[0]);
            auto Jt0 = vData.pack(VProps::J, pt[1]);
            auto Jt1 = vData.pack(VProps::J, pt[2]);
            auto Jt2 = vData.pack(VProps::J, pt[3]);

            T lambda0 = kappa * (2 * activeGap2 * activeGap2 * (6 + 2 * zs::log(I5) - 7 * I5 - 6 * I5 * zs::log(I5) + 1 / I5));
            auto q0 = flatten(F * dyadic_prod(normal, normal)) / zs::sqrt(I5);
            auto hess = PFPx.transpose() * (lambda0 * dyadic_prod(q0, q0)) * PFPx;

            // printf("pt lambda0[%d]: %f\n", pti, (float)lambda0);
            // printf("pt hess[%d] norm: %f\n", pti, (float)hess.norm());

            if (bodyA >= 0 && bodyB >= 0)
            {
                // printf("PT Rigid[%d]-Rigid[%d]\n", bodyA, bodyB);
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA));
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB));

                zs::vec<int, 2> segStart{bodyA * 12, bodyB * 12};
                zs::vec<int, 2> segLen{12, 12};
                zs::vec<int, 2> segIsKin{isKinA, isKinB};
#if s_enableMatlabGradHessConversion
                zs::vec<T, 24> grad_q;
                zs::vec<T, 24, 24> hess_q;
                auto Jv0 = vData.pack(VProps::JVec, pt[0]);
                auto Jv1 = vData.pack(VProps::JVec, pt[1]);
                auto Jv2 = vData.pack(VProps::JVec, pt[2]);
                auto Jv3 = vData.pack(VProps::JVec, pt[3]);
                grad_hess_pt_AA_conversion(
                    grad.data(), hess.data(), Jv0.data(), Jv1.data(),
                    Jv2.data(), Jv3.data(), grad_q.data(), hess_q.data());
#else 
                mat12x24 J = mat12x24::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; ++j) {
                        J(i, j) = Jp(i, j);
                        J(i + 3, j + 12) = Jt0(i, j);
                        J(i + 6, j + 12) = Jt1(i, j);
                        J(i + 9, j + 12) = Jt2(i, j);
                    }
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
#endif
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA < 0 && bodyB < 0)
            {
                printf("PT Soft-Soft\n");
                zs::vec<int, 4> segStart{(pt[0] - rb_vNum) * 3 + rbDofs,
                                         (pt[1] - rb_vNum) * 3 + rbDofs,
                                         (pt[2] - rb_vNum) * 3 + rbDofs,
                                         (pt[3] - rb_vNum) * 3 + rbDofs};
                zs::vec<int, 4> segLen{3, 3, 3, 3};
                zs::vec<int, 4> segIsKin;
                for (int d = 0; d < 4; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, pt[d]));
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad,
                                          hess, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA >= 0)
            {
                // printf("PT Rigid[%d]-Soft\n", bodyA);
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA));
                zs::vec<int, 4> segStart{bodyA * 12,
                                         (pt[1] - rb_vNum) * 3 + rbDofs,
                                         (pt[2] - rb_vNum) * 3 + rbDofs,
                                         (pt[3] - rb_vNum) * 3 + rbDofs};
                zs::vec<int, 4> segLen{12, 3, 3, 3};
                zs::vec<int, 4> segIsKin{isKinA, 0, 0, 0};
                for (int d = 1; d < 4; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, pt[d]));
                auto J = zs::vec<T, 12, 21>::zeros();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 12; j++)
                        J(i, j) = Jp(i, j);
                for (int i = 0; i < 9; i++)
                    J(3 + i, 12 + i) = 1;
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyB >= 0)
            {
                // printf("PT Soft-Rigid[%d]\n", bodyB);
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB));
                zs::vec<int, 2> segStart{(pt[0] - rb_vNum) * 3 + rbDofs,
                                         bodyB * 12};
                zs::vec<int, 2> segLen{3, 12};
                zs::vec<int, 2> segIsKin{
                    reinterpret_bits<Ti>(vData(VProps::isBC, pt[0])), isKinB};
                auto J = zs::vec<T, 12, 15>::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; ++j)
                    {
                        J(i + 3, j + 3) = Jt0(i, j);
                        J(i + 6, j + 3) = Jt1(i, j);
                        J(i + 9, j + 3) = Jt2(i, j);
                    }
                for (int i = 0; i < 3; i++)
                    J(i, i) = 1;
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }
        });

    auto numEE = EE.getCount();
    pol(range(numEE),
        [vData = view<space>(vData), rbData = view<space>(rbData),
         dofData = view<space>(dofData), rb_vNum = RigidBodyHandle::vNum,
         rbDofs = RigidBodyHandle::bodyNum * 12, sysHess = port<space>(hess),
         EE = EE.port(), 
         xi2 = xi * xi, dHat = dHat,
         activeGap2, kappa = kappa,
         kinematicSatisfied = solver.kinematicSatisfied
         ] __device__(int eei) mutable
        {
            auto ee = EE[eei];
            auto ea0 = vData.pack(VProps::xn, ee[0]);
            auto ea1 = vData.pack(VProps::xn, ee[1]);
            auto eb0 = vData.pack(VProps::xn, ee[2]);
            auto eb1 = vData.pack(VProps::xn, ee[3]);

            auto v0 = ea1 - ea0;
            auto v1 = eb0 - ea0;
            auto v2 = eb1 - ea0;
            mat3 Ds{v0[0], v1[0], v2[0], v0[1], v1[1],
                    v2[1], v0[2], v1[2], v2[2]};
            auto normal = v0.cross(eb1 - eb0).normalized();
            auto dis = v1.dot(normal);
            auto d_hat_sqrt = dHat;
            if (dis < 0)
            {
                normal = -normal;
                dis = -dis;
            }
            auto pos2 = eb0 + normal * (d_hat_sqrt - dis);
            auto pos3 = eb1 + normal * (d_hat_sqrt - dis);
            if (d_hat_sqrt - dis < 0)
                printf("FUCKING WRONG EEHESS! dhat - dis = %f (which < 0)\n",
                       d_hat_sqrt - dis);

            auto u0 = v0;
            auto u1 = pos2 - ea0;
            auto u2 = pos3 - ea0;
            mat3 Dm{u0[0], u1[0], u2[0], u0[1], u1[1],
                    u2[1], u0[2], u1[2], u2[2]};
            auto DmInv = inverse(Dm);
            auto F = Ds * DmInv;
            auto I5 = (F * normal).l2NormSqr();
            // T I5 = normal.dot(F.transpose() * F * normal);

            zs::vec<T, 9> flatten_pk1{};
            {
                auto tmp = flatten(F * dyadic_prod(normal, normal));
                flatten_pk1 = -2 * kappa *
                              (activeGap2 * activeGap2 * (I5 - 1) *
                               (1 + 2 * zs::log(I5) - 1 / I5)) *
                              tmp;
            }

            zs::vec<T, 9, 12> PFPx = dFdXMatrix(DmInv, wrapv<3>{});

            auto grad = -PFPx.transpose() * flatten_pk1;
            scatterContactForce(ee, grad, vData);

            // ABD: update q grad instead of x grad
            int bodyA = reinterpret_bits<Ti>(vData(VProps::body, ee[0]));
            int bodyB = reinterpret_bits<Ti>(vData(VProps::body, ee[2]));
            auto Jea0 = vData.pack(VProps::J, ee[0]);
            auto Jea1 = vData.pack(VProps::J, ee[1]);
            auto Jeb0 = vData.pack(VProps::J, ee[2]);
            auto Jeb1 = vData.pack(VProps::J, ee[3]);

            T lambda0 = kappa * (2 * activeGap2 * activeGap2 *
                                 (6 + 2 * zs::log(I5) - 7 * I5 -
                                  6 * I5 * zs::log(I5) + 1 / I5));

            if (lambda0 < 0)
                printf("FUCKING WRONG EEHESS! lambda0 = %e, I5 = %e\n", lambda0,
                       I5);

            auto nn = dyadic_prod(normal, normal);
            auto fnn = F * nn;
            auto q0 = flatten(fnn) / zs::sqrt(I5);
            auto hess =
                PFPx.transpose() * (lambda0 * dyadic_prod(q0, q0)) * PFPx;

            if (bodyA >= 0 && bodyB >= 0)
            {
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA));
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB));
                zs::vec<int, 2> segStart{bodyA * 12, bodyB * 12};
                zs::vec<int, 2> segLen{12, 12};
                zs::vec<int, 2> segIsKin{isKinA, isKinB};
#if s_enableMatlabGradHessConversion
                zs::vec<T, 24> grad_q;
                zs::vec<T, 24, 24> hess_q;
                auto Jv0 = vData.pack(VProps::JVec, ee[0]);
                auto Jv1 = vData.pack(VProps::JVec, ee[1]);
                auto Jv2 = vData.pack(VProps::JVec, ee[2]);
                auto Jv3 = vData.pack(VProps::JVec, ee[3]);
                grad_hess_ee_AA_conversion(
                    grad.data(), hess.data(), Jv0.data(), Jv1.data(),
                    Jv2.data(), Jv3.data(), grad_q.data(), hess_q.data());
#else 
                mat12x24 J = mat12x24::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; ++j) {
                        J(i, j) = Jea0(i, j);
                        J(i + 3, j) = Jea1(i, j);
                        J(i + 6, j + 12) = Jeb0(i, j);
                        J(i + 9, j + 12) = Jeb1(i, j);
                    }
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
#endif
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA < 0 && bodyB < 0)
            {
                printf("EE Soft-Soft\n");
                zs::vec<int, 4> segStart{(ee[0] - rb_vNum) * 3 + rbDofs,
                                         (ee[1] - rb_vNum) * 3 + rbDofs,
                                         (ee[2] - rb_vNum) * 3 + rbDofs,
                                         (ee[3] - rb_vNum) * 3 + rbDofs};
                zs::vec<int, 4> segLen{3, 3, 3, 3};
                zs::vec<int, 4> segIsKin;
                for (int d = 0; d < 4; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, ee[d]));
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad,
                                          hess, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA >= 0)
            {
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA));
                zs::vec<int, 3> segStart{bodyA * 12,
                                         (ee[2] - rb_vNum) * 3 + rbDofs,
                                         (ee[3] - rb_vNum) * 3 + rbDofs};
                zs::vec<int, 3> segLen{12, 3, 3};
                zs::vec<int, 3> segIsKin{isKinA, 0, 0};
                for (int d = 1; d < 3; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, ee[d]));
                auto J = zs::vec<T, 12, 18>::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; ++j)
                    {
                        J(i, j) = Jea0(i, j);
                        J(i + 3, j) = Jea1(i, j);
                    }
                for (int i = 0; i < 6; i++)
                    J(i + 6, i + 12) = 1;

                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyB >= 0)
            {
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB));
                zs::vec<int, 3> segStart{(ee[0] - rb_vNum) * 3 + rbDofs,
                                         (ee[1] - rb_vNum) * 3 + rbDofs,
                                         bodyB * 12};
                zs::vec<int, 3> segLen{3, 3, 12};
                zs::vec<int, 3> segIsKin{0, 0, isKinB};
                for (int d = 0; d < 2; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, ee[d]));
                auto J = zs::vec<T, 12, 18>::zeros();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 12; j++)
                    {
                        J(6 + i, 6 + j) = Jeb0(i, j);
                        J(9 + i, 6 + j) = Jeb1(i, j);
                    }
                for (int i = 0; i < 6; i++)
                    J(i, i) = 1;
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }
        });

#if s_enableMollification
    if (!solver.enableMollification)
        return;
    using Vec12View = zs::vec_view<T, zs::integer_sequence<int, 12>>;
    using Vec9View = zs::vec_view<T, zs::integer_sequence<int, 9>>;
    using Vec6View = zs::vec_view<T, zs::integer_sequence<int, 6>>;
    auto get_mollifier = [] __device__(const auto &ea0Rest, const auto &ea1Rest,
                                       const auto &eb0Rest, const auto &eb1Rest,
                                       const auto &ea0, const auto &ea1,
                                       const auto &eb0, const auto &eb1)
    {
        T epsX = mollifier_threshold_ee(ea0Rest, ea1Rest, eb0Rest, eb1Rest);
        return zs::make_tuple(mollifier_ee(ea0, ea1, eb0, eb1, epsX),
                              mollifier_grad_ee(ea0, ea1, eb0, eb1, epsX),
                              mollifier_hess_ee(ea0, ea1, eb0, eb1, epsX));
    };
    // mollifier
    auto numEEM = EEM.getCount();
    pol(range(numEEM),
        [vData = view<space>(vData), dofData = view<space>(dofData),
         rbData = view<space>(rbData), sysHess = port<space>(hess),
         rbDofs = RigidBodyHandle::bodyNum * 12, rb_vNum = RigidBodyHandle::vNum,
         EEM = EEM.port(), xi2 = xi * xi, dHat = dHat,
         activeGap2, kappa = kappa, get_mollifier,
         kinematicSatisfied = solver.kinematicSatisfied
         ] __device__(int eemi) mutable
        {
            auto eem = EEM[eemi]; // <x, y, z, w>
            auto ea0Rest = vData.pack(VProps::x0, eem[0]);
            auto ea1Rest = vData.pack(VProps::x0, eem[1]);
            auto eb0Rest = vData.pack(VProps::x0, eem[2]);
            auto eb1Rest = vData.pack(VProps::x0, eem[3]);
            auto ea0 = vData.pack(VProps::xn, eem[0]);
            auto ea1 = vData.pack(VProps::xn, eem[1]);
            auto eb0 = vData.pack(VProps::xn, eem[2]);
            auto eb1 = vData.pack(VProps::xn, eem[3]);

            // ABD: update q grad instead of x grad
            int bodyA = reinterpret_bits<Ti>(vData(VProps::body, eem[0]));
            int bodyB = reinterpret_bits<Ti>(vData(VProps::body, eem[2]));
            auto Jea0 = vData.pack(VProps::J, eem[0]);
            auto Jea1 = vData.pack(VProps::J, eem[1]);
            auto Jeb0 = vData.pack(VProps::J, eem[2]);
            auto Jeb1 = vData.pack(VProps::J, eem[3]);

            auto eeGrad = dist_grad_ee(ea0, ea1, eb0, eb1);
            auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
            if (dist2 < xi2)
                printf("dist already smaller than xi!\n");
            auto barrierDist2 = barrier(dist2 - xi2, activeGap2, kappa);
            auto barrierDistGrad =
                barrier_gradient(dist2 - xi2, activeGap2, kappa);
            auto barrierDistHess =
                barrier_hessian(dist2 - xi2, activeGap2, kappa);
            auto [mollifierEE, mollifierGradEE, mollifierHessEE] =
                get_mollifier(ea0Rest, ea1Rest, eb0Rest, eb1Rest, ea0, ea1, eb0,
                              eb1);

            auto scaledMollifierGrad = barrierDist2 * mollifierGradEE;
            auto scaledEEGrad = mollifierEE * barrierDistGrad * eeGrad;
            auto grad_x = -(scaledMollifierGrad + scaledEEGrad);
            vec12 grad_x_vec;
            for (int i = 0; i < 4; i++)
                for (int d = 0; d < 3; d++)
                    grad_x_vec(i * 3 + d) = grad_x(i, d); // reshape?
            scatterContactForce(eem, grad_x_vec, vData);

            // hessian
            auto eeGrad_ = Vec12View{eeGrad.data()};
            auto eemHess =
                barrierDist2 * mollifierHessEE +
                barrierDistGrad *
                    (dyadic_prod(Vec12View{mollifierGradEE.data()}, eeGrad_) +
                     dyadic_prod(eeGrad_, Vec12View{mollifierGradEE.data()}));

            auto hess = dist_hess_ee(ea0, ea1, eb0, eb1);
            hess = (barrierDistHess * dyadic_prod(eeGrad_, eeGrad_) +
                      barrierDistGrad * hess);
            eemHess += mollifierEE * hess;
            // make pd
            if (mollifierEE != 0)
                make_pd(eemHess);

            if (bodyA >= 0 && bodyB >= 0)
            {
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA));
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB));
                zs::vec<int, 2> segStart{bodyA * 12, bodyB * 12};
                zs::vec<int, 2> segLen{12, 12};
                zs::vec<int, 2> segIsKin{isKinA, isKinB};
#if s_enableMatlabGradHessConversion
                zs::vec<T, 24> grad_q;
                zs::vec<T, 24, 24> hess_q;
                auto Jv0 = vData.pack(VProps::JVec, eem[0]);
                auto Jv1 = vData.pack(VProps::JVec, eem[1]);
                auto Jv2 = vData.pack(VProps::JVec, eem[2]);
                auto Jv3 = vData.pack(VProps::JVec, eem[3]);
                grad_hess_ee_AA_conversion(
                    grad_x_vec.data(), eemHess.data(), Jv0.data(), Jv1.data(),
                    Jv2.data(), Jv3.data(), grad_q.data(), hess_q.data());
#else  
                mat12x24 J = mat12x24::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; ++j) {
                        J(i, j) = Jea0(i, j);
                        J(i + 3, j) = Jea1(i, j);
                        J(i + 6, j + 12) = Jeb0(i, j);
                        J(i + 9, j + 12) = Jeb1(i, j);
                    }
                auto grad_q = J.transpose() * grad_x_vec;
                auto hess_q = J.transpose() * eemHess * J;
#endif
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA < 0 && bodyB < 0)
            {
                printf("EEM Soft-Soft\n");
                zs::vec<int, 4> segStart{(eem[0] - rb_vNum) * 3 + rbDofs,
                                         (eem[1] - rb_vNum) * 3 + rbDofs,
                                         (eem[2] - rb_vNum) * 3 + rbDofs,
                                         (eem[3] - rb_vNum) * 3 + rbDofs};
                zs::vec<int, 4> segLen{3, 3, 3, 3};
                zs::vec<int, 4> segIsKin;
                for (int d = 0; d < 4; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, eem[d]));
                scatterGradientAndHessian(segStart, segLen, segIsKin,
                                          grad_x_vec, eemHess, dofData, rbData,
                                          vData, rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA >= 0)
            {
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA));
                zs::vec<int, 3> segStart{bodyA * 12,
                                         (eem[2] - rb_vNum) * 3 + rbDofs,
                                         (eem[3] - rb_vNum) * 3 + rbDofs};
                zs::vec<int, 3> segLen{12, 3, 3};
                zs::vec<int, 3> segIsKin{isKinA, 0, 0};
                for (int d = 1; d < 3; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, eem[d]));
                auto J = zs::vec<T, 12, 18>::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; ++j)
                    {
                        J(i, j) = Jea0(i, j);
                        J(i + 3, j) = Jea1(i, j);
                    }
                for (int i = 0; i < 6; i++)
                    J(i + 6, i + 12) = 1;

                auto grad_q = J.transpose() * grad_x_vec;
                auto hess_q = J.transpose() * eemHess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyB >= 0)
            {
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB));
                zs::vec<int, 3> segStart{(eem[0] - rb_vNum) * 3 + rbDofs,
                                         (eem[1] - rb_vNum) * 3 + rbDofs,
                                         bodyB * 12};
                zs::vec<int, 3> segLen{3, 3, 12};
                zs::vec<int, 3> segIsKin{0, 0, isKinB};
                for (int d = 0; d < 2; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, eem[d]));
                auto J = zs::vec<T, 12, 18>::zeros();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 12; j++)
                    {
                        J(6 + i, 6 + j) = Jeb0(i, j);
                        J(9 + i, 6 + j) = Jeb1(i, j);
                    }
                for (int i = 0; i < 6; i++)
                    J(i, i) = 1;
                auto grad_q = J.transpose() * grad_x_vec;
                auto hess_q = J.transpose() * eemHess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }
        });

    auto numPPM = PPM.getCount();
    pol(range(numPPM),
        [vData = view<space>(vData), rbData = view<space>(rbData),
         sysHess = port<space>(hess), dofData = view<space>(dofData),
         rbDofs = RigidBodyHandle::bodyNum * 12, rb_vNum = RigidBodyHandle::vNum,
         PPM = PPM.port(), xi2 = xi * xi, dHat = dHat,
         activeGap2, kappa = kappa, get_mollifier,
         kinematicSatisfied = solver.kinematicSatisfied 
         ] __device__(int ppmi) mutable
        {
            auto ppm = PPM[ppmi]; // <x, z, y, w>, <0, 2, 1, 3>
            auto ea0 = vData.pack(VProps::xn, ppm[0]);
            auto ea1 = vData.pack(VProps::xn, ppm[1]);
            auto eb0 = vData.pack(VProps::xn, ppm[2]);
            auto eb1 = vData.pack(VProps::xn, ppm[3]);
            // ABD: update q grad instead of x grad
            int bodyA = reinterpret_bits<Ti>(vData(VProps::body, ppm[0]));
            int bodyB = reinterpret_bits<Ti>(vData(VProps::body, ppm[2]));
            auto Jea0 = vData.pack(VProps::J, ppm[0]);
            auto Jea1 = vData.pack(VProps::J, ppm[1]);
            auto Jeb0 = vData.pack(VProps::J, ppm[2]);
            auto Jeb1 = vData.pack(VProps::J, ppm[3]);
            auto ppGrad = dist_grad_pp(ea0, eb0);
            auto dist2 = dist2_pp(ea0, eb0);
            if (dist2 < xi2)
            {
                printf("dist already smaller than xi!\n");
            }
            auto barrierDist2 = barrier(dist2 - xi2, activeGap2, kappa);
            auto barrierDistGrad =
                barrier_gradient(dist2 - xi2, activeGap2, kappa);
            auto barrierDistHess =
                barrier_hessian(dist2 - xi2, activeGap2, kappa);
            auto ea0Rest = vData.pack(VProps::x0, ppm[0]);
            auto ea1Rest = vData.pack(VProps::x0, ppm[1]);
            auto eb0Rest = vData.pack(VProps::x0, ppm[2]);
            auto eb1Rest = vData.pack(VProps::x0, ppm[3]);
            auto [mollifierEE, mollifierGradEE, mollifierHessEE] =
                get_mollifier(ea0Rest, ea1Rest, eb0Rest, eb1Rest, ea0, ea1, eb0,
                              eb1);

            auto scaledMollifierGrad = barrierDist2 * mollifierGradEE;
            auto scaledPPGrad = mollifierEE * barrierDistGrad * ppGrad;
            auto grad_x_0 =
                -(row(scaledMollifierGrad, 0) + row(scaledPPGrad, 0));
            auto grad_x_1 = -row(scaledMollifierGrad, 1);
            auto grad_x_2 =
                -(row(scaledMollifierGrad, 2) + row(scaledPPGrad, 1));
            auto grad_x_3 = -row(scaledMollifierGrad, 3);
            vec12 grad_x_vec;
            for (int i = 0; i < 3; i++)
                grad_x_vec(i) = grad_x_0(i);
            for (int i = 0; i < 3; i++)
                grad_x_vec(3 + i) = grad_x_1(i);
            for (int i = 0; i < 3; i++)
                grad_x_vec(6 + i) = grad_x_2(i);
            for (int i = 0; i < 3; i++)
                grad_x_vec(9 + i) = grad_x_3(i);
            scatterContactForce(ppm, grad_x_vec, vData);

            // hessian
            using GradT = zs::vec<T, 12>;
            auto extendedPPGrad = GradT::zeros();
            for (int d = 0; d != 3; ++d)
            {
                extendedPPGrad(d) = barrierDistGrad * ppGrad(0, d);
                extendedPPGrad(6 + d) = barrierDistGrad * ppGrad(1, d);
            }
            auto ppmHess =
                barrierDist2 * mollifierHessEE +
                dyadic_prod(Vec12View{mollifierGradEE.data()}, extendedPPGrad) +
                dyadic_prod(extendedPPGrad, Vec12View{mollifierGradEE.data()});

            auto hess = dist_hess_pp(ea0, eb0);
            auto ppGrad_ = Vec6View{ppGrad.data()};

            hess = (barrierDistHess * dyadic_prod(ppGrad_, ppGrad_) +
                      barrierDistGrad * hess);
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j)
                {
                    ppmHess(0 + i, 0 + j) += mollifierEE * hess(0 + i, 0 + j);
                    ppmHess(0 + i, 6 + j) += mollifierEE * hess(0 + i, 3 + j);
                    ppmHess(6 + i, 0 + j) += mollifierEE * hess(3 + i, 0 + j);
                    ppmHess(6 + i, 6 + j) += mollifierEE * hess(3 + i, 3 + j);
                }
            // make pd
            if (mollifierEE != 0)
                make_pd(ppmHess);

            if (bodyA >= 0 && bodyB >= 0)
            {
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA));
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB));
                zs::vec<int, 2> segStart{bodyA * 12, bodyB * 12};
                zs::vec<int, 2> segLen{12, 12};
                zs::vec<int, 2> segIsKin{isKinA, isKinB};
#if s_enableMatlabGradHessConversion
                zs::vec<T, 24> grad_q;
                zs::vec<T, 24, 24> hess_q;
                auto Jv0 = vData.pack(VProps::JVec, ppm[0]);
                auto Jv1 = vData.pack(VProps::JVec, ppm[1]);
                auto Jv2 = vData.pack(VProps::JVec, ppm[2]);
                auto Jv3 = vData.pack(VProps::JVec, ppm[3]);
                grad_hess_ee_AA_conversion(
                    grad_x_vec.data(), ppmHess.data(), Jv0.data(), Jv1.data(),
                    Jv2.data(), Jv3.data(), grad_q.data(), hess_q.data());
#else 
                mat12x24 J = mat12x24::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; ++j) {
                        J(i, j) = Jea0(i, j);
                        J(i + 3, j) = Jea1(i, j);
                        J(i + 6, j + 12) = Jeb0(i, j);
                        J(i + 9, j + 12) = Jeb1(i, j);
                    }
                auto grad_q = J.transpose() * grad_x_vec;
                auto hess_q = J.transpose() * ppmHess * J;
#endif
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA < 0 && bodyB < 0)
            {
                printf("PPM Soft-Soft\n");
                zs::vec<int, 4> segStart{(ppm[0] - rb_vNum) * 3 + rbDofs,
                                         (ppm[1] - rb_vNum) * 3 + rbDofs,
                                         (ppm[2] - rb_vNum) * 3 + rbDofs,
                                         (ppm[3] - rb_vNum) * 3 + rbDofs};
                zs::vec<int, 4> segLen{3, 3, 3, 3};
                zs::vec<int, 4> segIsKin;
                for (int d = 0; d < 4; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, ppm[d]));
                scatterGradientAndHessian(segStart, segLen, segIsKin,
                                          grad_x_vec, ppmHess, dofData, rbData,
                                          vData, rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA >= 0)
            {
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA));
                zs::vec<int, 3> segStart{bodyA * 12,
                                         (ppm[2] - rb_vNum) * 3 + rbDofs,
                                         (ppm[3] - rb_vNum) * 3 + rbDofs};
                zs::vec<int, 3> segLen{12, 3, 3};
                zs::vec<int, 3> segIsKin{isKinA, 0, 0};
                for (int d = 1; d < 3; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, ppm[d]));
                auto J = zs::vec<T, 12, 18>::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; ++j)
                    {
                        J(i, j) = Jea0(i, j);
                        J(i + 3, j) = Jea1(i, j);
                    }
                for (int i = 0; i < 6; i++)
                    J(i + 6, i + 12) = 1;

                auto grad_q = J.transpose() * grad_x_vec;
                auto hess_q = J.transpose() * ppmHess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyB >= 0)
            {
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB));
                zs::vec<int, 3> segStart{(ppm[0] - rb_vNum) * 3 + rbDofs,
                                         (ppm[1] - rb_vNum) * 3 + rbDofs,
                                         bodyB * 12};
                zs::vec<int, 3> segLen{3, 3, 12};
                zs::vec<int, 3> segIsKin{0, 0, isKinB};
                for (int d = 0; d < 2; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, ppm[d]));
                auto J = zs::vec<T, 12, 18>::zeros();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 12; j++)
                    {
                        J(6 + i, 6 + j) = Jeb0(i, j);
                        J(9 + i, 6 + j) = Jeb1(i, j);
                    }
                for (int i = 0; i < 6; i++)
                    J(i, i) = 1;
                auto grad_q = J.transpose() * grad_x_vec;
                auto hess_q = J.transpose() * ppmHess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }
        });

    auto numPEM = PEM.getCount();
    pol(range(numPEM),
        [vData = view<space>(vData), rbData = view<space>(rbData),
         sysHess = port<space>(hess), rbDofs = RigidBodyHandle::bodyNum * 12,
         rb_vNum = RigidBodyHandle::vNum, dofData = view<space>(dofData),
         PEM = PEM.port(), xi2 = xi * xi, dHat = dHat,
         activeGap2, kappa = kappa, get_mollifier,
         kinematicSatisfied = solver.kinematicSatisfied 
         ] __device__(int pemi) mutable
        {
            auto pem = PEM[pemi]; // <x, w, y, z>, <0, 2, 3, 1>
            auto ea0Rest = vData.pack(VProps::x0, pem[0]);
            auto ea1Rest = vData.pack(VProps::x0, pem[1]);
            auto eb0Rest = vData.pack(VProps::x0, pem[2]);
            auto eb1Rest = vData.pack(VProps::x0, pem[3]);
            auto ea0 = vData.pack(VProps::xn, pem[0]);
            auto ea1 = vData.pack(VProps::xn, pem[1]);
            auto eb0 = vData.pack(VProps::xn, pem[2]);
            auto eb1 = vData.pack(VProps::xn, pem[3]);
            // ABD: update q grad instead of x grad
            int bodyA = reinterpret_bits<Ti>(vData(VProps::body, pem[0]));
            int bodyB = reinterpret_bits<Ti>(vData(VProps::body, pem[2]));
            auto Jea0 = vData.pack(VProps::J, pem[0]);
            auto Jea1 = vData.pack(VProps::J, pem[1]);
            auto Jeb0 = vData.pack(VProps::J, pem[2]);
            auto Jeb1 = vData.pack(VProps::J, pem[3]);

            auto peGrad = dist_grad_pe(ea0, eb0, eb1);
            auto dist2 = dist2_pe(ea0, eb0, eb1);
            if (dist2 < xi2)
            {
                printf("dist already smaller than xi!\n");
            }
            auto barrierDist2 = barrier(dist2 - xi2, activeGap2, kappa);
            auto barrierDistGrad =
                barrier_gradient(dist2 - xi2, activeGap2, kappa);
            auto barrierDistHess =
                barrier_hessian(dist2 - xi2, activeGap2, kappa);
            auto [mollifierEE, mollifierGradEE, mollifierHessEE] =
                get_mollifier(ea0Rest, ea1Rest, eb0Rest, eb1Rest, ea0, ea1, eb0,
                              eb1);

            auto scaledMollifierGrad = barrierDist2 * mollifierGradEE;
            auto scaledPEGrad = mollifierEE * barrierDistGrad * peGrad;
            auto grad_x_0 =
                -(row(scaledMollifierGrad, 0) + row(scaledPEGrad, 0));
            auto grad_x_1 = -row(scaledMollifierGrad, 1);
            auto grad_x_2 =
                -(row(scaledMollifierGrad, 2) + row(scaledPEGrad, 1));
            auto grad_x_3 =
                -(row(scaledMollifierGrad, 3) + row(scaledPEGrad, 2));
            vec12 grad_x_vec;
            for (int i = 0; i < 3; i++)
                grad_x_vec(i) = grad_x_0(i);
            for (int i = 0; i < 3; i++)
                grad_x_vec(3 + i) = grad_x_1(i);
            for (int i = 0; i < 3; i++)
                grad_x_vec(6 + i) = grad_x_2(i);
            for (int i = 0; i < 3; i++)
                grad_x_vec(9 + i) = grad_x_3(i);
            scatterContactForce(pem, grad_x_vec, vData);

            // hessian
            using GradT = zs::vec<T, 12>;
            auto extendedPEGrad = GradT::zeros();
            for (int d = 0; d != 3; ++d)
            {
                extendedPEGrad(d) = barrierDistGrad * peGrad(0, d);
                extendedPEGrad(6 + d) = barrierDistGrad * peGrad(1, d);
                extendedPEGrad(9 + d) = barrierDistGrad * peGrad(2, d);
            }
            auto pemHess =
                barrierDist2 * mollifierHessEE +
                dyadic_prod(Vec12View{mollifierGradEE.data()}, extendedPEGrad) +
                dyadic_prod(extendedPEGrad, Vec12View{mollifierGradEE.data()});

            auto hess = dist_hess_pe(ea0, eb0, eb1);
            auto peGrad_ = Vec9View{peGrad.data()};

            hess = (barrierDistHess * dyadic_prod(peGrad_, peGrad_) +
                      barrierDistGrad * hess);
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j)
                {
                    pemHess(0 + i, 0 + j) += mollifierEE * hess(0 + i, 0 + j);
                    //
                    pemHess(0 + i, 6 + j) += mollifierEE * hess(0 + i, 3 + j);
                    pemHess(0 + i, 9 + j) += mollifierEE * hess(0 + i, 6 + j);
                    //
                    pemHess(6 + i, 0 + j) += mollifierEE * hess(3 + i, 0 + j);
                    pemHess(9 + i, 0 + j) += mollifierEE * hess(6 + i, 0 + j);
                    //
                    pemHess(6 + i, 6 + j) += mollifierEE * hess(3 + i, 3 + j);
                    pemHess(6 + i, 9 + j) += mollifierEE * hess(3 + i, 6 + j);
                    pemHess(9 + i, 6 + j) += mollifierEE * hess(6 + i, 3 + j);
                    pemHess(9 + i, 9 + j) += mollifierEE * hess(6 + i, 6 + j);
                }

            // make pd
            if (mollifierEE != 0)
                make_pd(pemHess);

            if (bodyA >= 0 && bodyB >= 0)
            {
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA));
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB));
                zs::vec<int, 2> segStart{bodyA * 12, bodyB * 12};
                zs::vec<int, 2> segLen{12, 12};
                zs::vec<int, 2> segIsKin{isKinA, isKinB};
#if s_enableMatlabGradHessConversion
                zs::vec<T, 24> grad_q;
                zs::vec<T, 24, 24> hess_q;
                auto Jv0 = vData.pack(VProps::JVec, pem[0]);
                auto Jv1 = vData.pack(VProps::JVec, pem[1]);
                auto Jv2 = vData.pack(VProps::JVec, pem[2]);
                auto Jv3 = vData.pack(VProps::JVec, pem[3]);
                grad_hess_ee_AA_conversion(
                    grad_x_vec.data(), pemHess.data(), Jv0.data(), Jv1.data(),
                    Jv2.data(), Jv3.data(), grad_q.data(), hess_q.data());
#else 
                mat12x24 J = mat12x24::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; ++j) {
                        J(i, j) = Jea0(i, j);
                        J(i + 3, j) = Jea1(i, j);
                        J(i + 6, j + 12) = Jeb0(i, j);
                        J(i + 9, j + 12) = Jeb1(i, j);
                    }
                auto grad_q = J.transpose() * grad_x_vec;
                auto hess_q = J.transpose() * pemHess * J;
#endif
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA < 0 && bodyB < 0)
            {
                printf("PEM Soft-Soft\n");
                zs::vec<int, 4> segStart{(pem[0] - rb_vNum) * 3 + rbDofs,
                                         (pem[1] - rb_vNum) * 3 + rbDofs,
                                         (pem[2] - rb_vNum) * 3 + rbDofs,
                                         (pem[3] - rb_vNum) * 3 + rbDofs};
                zs::vec<int, 4> segLen{3, 3, 3, 3};
                zs::vec<int, 4> segIsKin;
                for (int d = 0; d < 4; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, pem[d]));
                scatterGradientAndHessian(segStart, segLen, segIsKin,
                                          grad_x_vec, pemHess, dofData, rbData,
                                          vData, rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA >= 0)
            {
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA));
                zs::vec<int, 3> segStart{bodyA * 12,
                                         (pem[2] - rb_vNum) * 3 + rbDofs,
                                         (pem[3] - rb_vNum) * 3 + rbDofs};
                zs::vec<int, 3> segLen{12, 3, 3};
                zs::vec<int, 3> segIsKin{isKinA, 0, 0};
                for (int d = 1; d < 3; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, pem[d]));
                auto J = zs::vec<T, 12, 18>::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; ++j)
                    {
                        J(i, j) = Jea0(i, j);
                        J(i + 3, j) = Jea1(i, j);
                    }
                for (int i = 0; i < 6; i++)
                    J(i + 6, i + 12) = 1;

                auto grad_q = J.transpose() * grad_x_vec;
                auto hess_q = J.transpose() * pemHess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyB >= 0)
            {
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB));
                zs::vec<int, 3> segStart{(pem[0] - rb_vNum) * 3 + rbDofs,
                                         (pem[1] - rb_vNum) * 3 + rbDofs,
                                         bodyB * 12};
                zs::vec<int, 3> segLen{3, 3, 12};
                zs::vec<int, 3> segIsKin{0, 0, isKinB};
                for (int d = 0; d < 2; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, pem[d]));
                auto J = zs::vec<T, 12, 18>::zeros();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 12; j++)
                    {
                        J(6 + i, 6 + j) = Jeb0(i, j);
                        J(9 + i, 6 + j) = Jeb1(i, j);
                    }
                for (int i = 0; i < 6; i++)
                    J(i, i) = 1;
                auto grad_q = J.transpose() * grad_x_vec;
                auto hess_q = J.transpose() * pemHess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q,
                                          hess_q, dofData, rbData, vData,
                                          rbDofs, rb_vNum, sysHess, kinematicSatisfied);
                return;
            }
        });
#endif // s_enableMollification
    // Vector<T> gNorm{vData.get_allocator(), 1};
    // gNorm.setVal(0, 0);
    // pol(range(dof),
    //     [grad = view<space>(gtemp), gNorm = view<space>(gNorm),
    //      dofData = view<space>(dofData)] __device__(int d)
    //     {
    //         grad(d) = dofData(DOFProps::grad, d) - grad(d) ;
    //         atomic_add(exec_cuda, &gNorm[0], grad(d) * grad(d));
    //     });
    // fmt::print("Barrier grad with norm {}:\n", (float)zs::sqrt(gNorm.getVal(0)));
    // for (int d = 0; d < dof; d++)
    //     fmt::print("{} ", (float)gtemp.getVal(d));
    // fmt::print("\n");
    return;
}

template <class T>
void ABDSolver::BarrierEnergy<T>::addFrictionGradientAndHessian(ABDSolver &solver, pol_t &pol)
{
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    auto &vData = solver.vData;
    auto &rbData = solver.rbData;
    auto &svData = solver.svData;
    auto &dofData = solver.dofData;
    auto &hess = solver.sysHess;

    auto &FPP = solver.FPP;
    auto &fricPP = solver.fricPP;
    auto &FPE = solver.FPE;
    auto &fricPE = solver.fricPE;
    auto &FPT = solver.FPT;
    auto &fricPT = solver.fricPT;
    auto &FEE = solver.FEE;
    auto &fricEE = solver.fricEE;

    auto &dHat = solver.dHat;
    auto &xi = solver.xi;
    auto &epsv = solver.epsv;
    auto &dt = solver.dt;
    auto &fricMu = solver.fricMu;

    T activeGap2 = dHat * dHat + (T)2.0 * xi * dHat;

    fmt::print("nFPP: {}, nFPE: {}, nFPT: {}, nFEE: {}\n", FPP.getCount(), FPE.getCount(), FPT.getCount(), FEE.getCount());

    auto numFPP = FPP.getCount();
    pol(range(numFPP),
        [vData = view<space>(vData), fricPP = view<space>(fricPP),
         rbData = view<space>(rbData), sysHess = port<space>(hess),
         dofData = view<space>(dofData), sb_vOffset = RigidBodyHandle::vNum,
         rbDofs = RigidBodyHandle::bodyNum * 12,
         FPP = FPP.port(), epsvh = epsv * dt, fricMu = fricMu,
         kinematicSatisfied = solver.kinematicSatisfied 
         ] __device__(int fppi) mutable {
            auto fpp = FPP[fppi];
            auto p0 = vData.pack(VProps::xn, fpp[0]) - vData.pack(VProps::xHat, fpp[0]);
            auto p1 = vData.pack(VProps::xn, fpp[1]) - vData.pack(VProps::xHat, fpp[1]);

            auto basis = fricPP.pack(FPPProps::basis, fppi);
            auto fn = fricPP(FPPProps::fn, fppi);
            auto relDX3D = point_point_rel_dx(p0, p1);
            auto relDX = basis.transpose() * relDX3D;
            auto relDXNorm2 = relDX.l2NormSqr();
            auto relDXNorm = zs::sqrt(relDXNorm2);
            auto f1_div_relDXNorm = zs::f1_SF_div_rel_dx_norm(relDXNorm2, epsvh);
            relDX *= f1_div_relDXNorm * fricMu * fn;
            auto TTTDX = -point_point_rel_dx_tan_to_mesh(relDX, basis); // 2 * 3 == transpose ==> 3 * 2 == multiply with J.t => 12 * 2
            auto grad_x0 = row(TTTDX, 0);
            auto grad_x1 = row(TTTDX, 1);
            int bodyA = reinterpret_bits<Ti>(vData(VProps::body, fpp[0]));
            int bodyB = reinterpret_bits<Ti>(vData(VProps::body, fpp[1]));
            auto J0 = vData.pack(VProps::J, fpp[0]);
            auto J1 = vData.pack(VProps::J, fpp[1]);
            zs::vec<T, 6> grad;
            // gradient
            for (int d = 0; d != 3; ++d) {
                grad(d) = grad_x0(d);
                grad(d + 3) = grad_x1(d);
            }
            scatterContactForce(fpp, grad, vData);

            relDX = basis.transpose() * relDX3D;
            auto TT = point_point_TT(basis); // 2x6
            auto f2_term = f2_SF_term(relDXNorm2, epsvh);
            using HessT = zs::vec<T, 6, 6>;
            auto hess = HessT::zeros();
            if (relDXNorm2 >= epsvh * epsvh) {
                zs::vec<T, 2> ubar{-relDX[1], relDX[0]};
                hess = dyadic_prod(
                    TT.transpose() *
                        ((fricMu * fn * f1_div_relDXNorm / relDXNorm2) * ubar),
                    ubar * TT);
            } else {
                if (relDXNorm == 0) {
                    if (fn > 0)
                        hess = fricMu * fn * f1_div_relDXNorm * TT.transpose() * TT;
                    // or ignored
                } else {
                    auto innerMtr = dyadic_prod((f2_term / relDXNorm) * relDX, relDX);
                    innerMtr(0, 0) += f1_div_relDXNorm;
                    innerMtr(1, 1) += f1_div_relDXNorm;
                    innerMtr *= fricMu * fn;
                    //
                    make_pd(innerMtr);
                    hess = TT.transpose() * innerMtr * TT;
                }
            }

            if (bodyA >= 0 && bodyB >= 0) {
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA)); 
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB)); 
                zs::vec<int, 2> segStart {bodyA * 12, bodyB * 12};
                zs::vec<int, 2> segLen {12, 12};
                zs::vec<int, 2> segIsKin{isKinA, isKinB}; 
                mat6x24 J = mat6x24::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; j++) {
                        J(i, j) = J0(i, j);
                        J(i + 3, j + 12) = J1(i, j);
                    }
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q, hess_q, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA < 0 && bodyB < 0)
            {
                printf("FPP Soft-Soft\n");
                zs::vec<int, 2> segStart {(fpp[0] - sb_vOffset) * 3 + rbDofs, (fpp[1] - sb_vOffset) * 3 + rbDofs};
                zs::vec<int, 2> segLen {3, 3};
                zs::vec<int, 2> segIsKin {0, 0}; 
                for (int d = 0; d < 2; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, fpp[d]));
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad, hess, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA >= 0)
            {
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA)); 
                // 12 + 3
                zs::vec<int, 2> segStart {bodyA * 12, (fpp[1] - sb_vOffset) * 3 + rbDofs};
                zs::vec<int, 2> segLen {12, 3};
                zs::vec<int, 2> segIsKin {isKinA, vData(VProps::isBC, fpp[1])};
                auto J = zs::vec<T, 6, 15>::zeros();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 12; j++)
                        J(i, j) = J0(i, j);
                for (int i = 0; i < 3; i++)
                    J(3 + i, 12 + i) = 1;
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q, hess_q, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyB >= 0)
            {
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB));
                // 3 + 12
                zs::vec<int, 2> segStart {(fpp[0] - sb_vOffset) * 3 + rbDofs, bodyB * 12};
                zs::vec<int, 2> segLen {3, 12};
                zs::vec<int, 2> segIsKin {vData(VProps::isBC, fpp[0]), isKinB};
                auto J = zs::vec<T, 6, 15>::zeros();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 12; j++)
                        J(3 + i, 3 + j) = J1(i, j);
                for (int i = 0; i < 3; i++)
                    J(i, i) = 1;
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q, hess_q, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied);
                return;
            }
        });

    auto numFPE = FPE.getCount();
    pol(range(numFPE),
        [vData = view<space>(vData), fricPE = view<space>(fricPE),
         rbData = view<space>(rbData), sysHess = port<space>(hess),
         dofData = view<space>(dofData), sb_vOffset = RigidBodyHandle::vNum,
         rbDofs = RigidBodyHandle::bodyNum * 12,
         FPE = FPE.port(), epsvh = epsv * dt, fricMu = fricMu,
         kinematicSatisfied = solver.kinematicSatisfied
         ] __device__(int fpei) mutable {
            auto fpe = FPE[fpei];
            auto p = vData.pack(VProps::xn, fpe[0]) - vData.pack(VProps::xHat, fpe[0]);
            auto e0 = vData.pack(VProps::xn, fpe[1]) - vData.pack(VProps::xHat, fpe[1]);
            auto e1 = vData.pack(VProps::xn, fpe[2]) - vData.pack(VProps::xHat, fpe[2]);

            auto basis = fricPE.pack(FPEProps::basis, fpei);
            auto fn = fricPE(FPEProps::fn, fpei);
            auto yita = fricPE(FPEProps::yita, fpei);
            auto relDX3D = point_edge_rel_dx(p, e0, e1, yita);
            auto relDX = basis.transpose() * relDX3D;
            auto relDXNorm2 = relDX.l2NormSqr();
            auto relDXNorm = zs::sqrt(relDXNorm2);
            auto f1_div_relDXNorm = zs::f1_SF_div_rel_dx_norm(relDXNorm2, epsvh);
            relDX *= f1_div_relDXNorm * fricMu * fn;
            auto TTTDX = -point_edge_rel_dx_tan_to_mesh(relDX, basis, yita);
            // gradient
            int bodyA = reinterpret_bits<Ti>(vData(VProps::body, fpe[0]));
            int bodyB = reinterpret_bits<Ti>(vData(VProps::body, fpe[1]));
            auto Jp = vData.pack(VProps::J, fpe[0]);
            auto Je0 = vData.pack(VProps::J, fpe[1]);
            auto Je1 = vData.pack(VProps::J, fpe[2]);
            zs::vec<T, 9> grad;
            for (int vi = 0; vi < 3; vi++)
                for (int di = 0; di < 3; di++)
                    grad(vi * 3 + di) = TTTDX(vi, di);
            scatterContactForce(fpe, grad, vData);

            relDX = basis.transpose() * relDX3D;
            auto TT = point_edge_TT(basis, yita); // 2x9
            auto f2_term = f2_SF_term(relDXNorm2, epsvh);
            using HessT = zs::vec<T, 9, 9>;
            auto hess = HessT::zeros();
            if (relDXNorm2 >= epsvh * epsvh) {
                zs::vec<T, 2> ubar{-relDX[1], relDX[0]};
                hess = dyadic_prod(
                    TT.transpose() *
                        ((fricMu * fn * f1_div_relDXNorm / relDXNorm2) * ubar),
                    ubar * TT);
            } else {
                if (relDXNorm == 0) {
                    if (fn > 0)
                        hess = fricMu * fn * f1_div_relDXNorm * TT.transpose() * TT;
                    // or ignored
                } else {
                    auto innerMtr = dyadic_prod((f2_term / relDXNorm) * relDX, relDX);
                    innerMtr(0, 0) += f1_div_relDXNorm;
                    innerMtr(1, 1) += f1_div_relDXNorm;
                    innerMtr *= fricMu * fn;
                    //
                    make_pd(innerMtr);
                    hess = TT.transpose() * innerMtr * TT;
                }
            }

            if (bodyA >= 0 && bodyB >= 0) {
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA)); 
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB)); 
                zs::vec<int, 2> segStart {bodyA * 12, bodyB * 12};
                zs::vec<int, 2> segLen {12, 12};
                zs::vec<int, 2> segIsKin {isKinA, isKinB}; 
                mat9x24 J = mat9x24::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; ++j) {
                        J(i, j) = Jp(i, j);
                        J(i + 3, j + 12) = Je0(i, j);
                        J(i + 6, j + 12) = Je1(i, j);
                    }
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q, hess_q, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA < 0 && bodyB < 0)
            {
                printf("FPE Soft-Soft\n");
                zs::vec<int, 3> segStart {(fpe[0] - sb_vOffset) * 3 + rbDofs,
                                         (fpe[1] - sb_vOffset) * 3 + rbDofs,
                                         (fpe[2] - sb_vOffset) * 3 + rbDofs};
                zs::vec<int, 3> segLen {3, 3, 3};
                zs::vec<int, 3> segIsKin {0, 0, 0}; 
                for (int d = 0; d < 3; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, fpe[d]));
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad, hess, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA >= 0)
            {
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA)); 
                // 12 + 3
                zs::vec<int, 3> segStart {bodyA * 12,
                                         (fpe[1] - sb_vOffset) * 3 + rbDofs,
                                         (fpe[2] - sb_vOffset) * 3 + rbDofs};
                zs::vec<int, 3> segLen {12, 3, 3};
                zs::vec<int, 3> segIsKin {isKinA, 0, 0}; 
                for (int d = 1; d < 3; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, fpe[d]));
                auto J = zs::vec<T, 9, 18>::zeros();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 12; j++)
                        J(i, j) = Jp(i, j);
                for (int i = 0; i < 6; i++)
                    J(3 + i, 12 + i) = 1;
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q, hess_q, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyB >= 0)
            {
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB)); 
                // 3 + 12
                zs::vec<int, 3> segStart {(fpe[0] - sb_vOffset) * 3 + rbDofs, bodyB * 12};
                zs::vec<int, 3> segLen {3, 12};
                zs::vec<int, 3> segIsKin {vData(VProps::isBC, fpe[0]), isKinB};
                auto J = zs::vec<T, 9, 15>::zeros();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 12; j++)
                        J(3 + i, 3 + j) = Je0(i, j);
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 12; j++)
                        J(6 + i, 3 + j) = Je1(i, j);
                for (int i = 0; i < 3; i++)
                    J(i, i) = 1;
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q, hess_q, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied);
                return;
            }
        });

    auto numFPT = FPT.getCount();
    pol(range(numFPT),
        [vData = view<space>(vData), fricPT = view<space>(fricPT),
         rbData = view<space>(rbData), sysHess = port<space>(hess),
         dofData = view<space>(dofData), sb_vOffset = RigidBodyHandle::vNum,
         rbDofs = RigidBodyHandle::bodyNum * 12,
         FPT = FPT.port(), epsvh = epsv * dt, fricMu = fricMu,
         kinematicSatisfied = solver.kinematicSatisfied
         ] __device__(int fpti) mutable {
            auto fpt = FPT[fpti];
            auto p = vData.pack(VProps::xn, fpt[0]) - vData.pack(VProps::xHat, fpt[0]);
            auto v0 = vData.pack(VProps::xn, fpt[1]) - vData.pack(VProps::xHat, fpt[1]);
            auto v1 = vData.pack(VProps::xn, fpt[2]) - vData.pack(VProps::xHat, fpt[2]);
            auto v2 = vData.pack(VProps::xn, fpt[3]) - vData.pack(VProps::xHat, fpt[3]);

            auto basis = fricPT.pack(FPTProps::basis, fpti);
            auto fn = fricPT(FPTProps::fn, fpti);
            auto betas = fricPT.pack(FPTProps::beta, fpti);
            auto relDX3D = point_triangle_rel_dx(p, v0, v1, v2, betas[0], betas[1]);
            auto relDX = basis.transpose() * relDX3D;
            auto relDXNorm2 = relDX.l2NormSqr();
            auto relDXNorm = zs::sqrt(relDXNorm2);
            auto f1_div_relDXNorm = zs::f1_SF_div_rel_dx_norm(relDXNorm2, epsvh);
            relDX *= f1_div_relDXNorm * fricMu * fn;
            auto TTTDX = -point_triangle_rel_dx_tan_to_mesh(relDX, basis, betas[0],
                                                            betas[1]);
            // gradient
            int bodyA = reinterpret_bits<Ti>(vData(VProps::body, fpt[0]));
            int bodyB = reinterpret_bits<Ti>(vData(VProps::body, fpt[1]));
            auto Jp = vData.pack(VProps::J, fpt[0]);
            auto Jt0 = vData.pack(VProps::J, fpt[1]);
            auto Jt1 = vData.pack(VProps::J, fpt[2]);
            auto Jt2 = vData.pack(VProps::J, fpt[3]);

            vec12 grad;
            for (int vi = 0; vi < 4; vi++)
                for (int di = 0; di < 3; di++)
                    grad(vi * 3 + di) = TTTDX(vi, di);
            scatterContactForce(fpt, grad, vData);

            relDX = basis.transpose() * relDX3D;
            auto TT = point_triangle_TT(basis, betas[0], betas[1]); // 2x12
            auto f2_term = f2_SF_term(relDXNorm2, epsvh);
            using HessT = zs::vec<T, 12, 12>;
            auto hess = HessT::zeros();
            if (relDXNorm2 >= epsvh * epsvh) {
                zs::vec<T, 2> ubar{-relDX[1], relDX[0]};
                hess = dyadic_prod(
                    TT.transpose() *
                        ((fricMu * fn * f1_div_relDXNorm / relDXNorm2) * ubar),
                    ubar * TT);
            } else {
                if (relDXNorm == 0) {
                    if (fn > 0)
                        hess = fricMu * fn * f1_div_relDXNorm * TT.transpose() * TT;
                    // or ignored
                } else {
                    auto innerMtr = dyadic_prod((f2_term / relDXNorm) * relDX, relDX);
                    innerMtr(0, 0) += f1_div_relDXNorm;
                    innerMtr(1, 1) += f1_div_relDXNorm;
                    innerMtr *= fricMu * fn;
                    //
                    make_pd(innerMtr);
                    hess = TT.transpose() * innerMtr * TT;
                }
            }
            if (bodyA >= 0 && bodyB >= 0) {
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA)); 
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB)); 
                zs::vec<int, 2> segStart {bodyA * 12, bodyB * 12};
                zs::vec<int, 2> segLen {12, 12};
                zs::vec<int, 2> segIsKin {isKinA, isKinB}; 
                mat12x24 J = mat12x24::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; ++j) {
                        J(i, j) = Jp(i, j);
                        J(i + 3, j + 12) = Jt0(i, j);
                        J(i + 6, j + 12) = Jt1(i, j);
                        J(i + 9, j + 12) = Jt2(i, j);
                    }
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q, hess_q, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA < 0 && bodyB < 0)
            {
                printf("FPT Soft-Soft\n");
                zs::vec<int, 4> segStart {(fpt[0] - sb_vOffset) * 3 + rbDofs,
                                         (fpt[1] - sb_vOffset) * 3 + rbDofs,
                                         (fpt[2] - sb_vOffset) * 3 + rbDofs,
                                         (fpt[3] - sb_vOffset) * 3 + rbDofs};
                zs::vec<int, 4> segLen {3, 3, 3, 3};
                zs::vec<int, 4> segIsKin {0, 0, 0, 0}; 
                for (int d = 0; d < 4; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, fpt[d]));
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad, hess, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA >= 0)
            {
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA)); 
                zs::vec<int, 4> segStart {bodyA * 12,
                                         (fpt[1] - sb_vOffset) * 3 + rbDofs,
                                         (fpt[2] - sb_vOffset) * 3 + rbDofs,
                                         (fpt[3] - sb_vOffset) * 3 + rbDofs};
                zs::vec<int, 4> segLen {12, 3, 3, 3};
                zs::vec<int, 4> segIsKin {isKinA, 0, 0, 0}; 
                for (int d = 1; d < 4; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, fpt[d]));
                auto J = zs::vec<T, 12, 21>::zeros();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 12; j++)
                        J(i, j) = Jp(i, j);
                for (int i = 0; i < 9; i++)
                    J(3 + i, 12 + i) = 1;
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q, hess_q, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyB >= 0)
            {
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB)); 
                zs::vec<int, 2> segStart {(fpt[0] - sb_vOffset) * 3 + rbDofs, bodyB * 12};
                zs::vec<int, 2> segLen {3, 12};
                zs::vec<int, 2> segIsKin {vData(VProps::isBC, fpt[0]), isKinB};
                auto J = zs::vec<T, 12, 15>::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; ++j) {
                        J(i + 3, j + 3) = Jt0(i, j);
                        J(i + 6, j + 3) = Jt1(i, j);
                        J(i + 9, j + 3) = Jt2(i, j);
                    }
                for (int i = 0; i < 3; i++)
                    J(i, i) = 1;
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q, hess_q, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied);
                return;
            }
        });

    auto numFEE = FEE.getCount();
    pol(range(numFEE),
        [vData = view<space>(vData), fricEE = view<space>(fricEE),
         rbData = view<space>(rbData), sysHess = port<space>(hess),
         dofData = view<space>(dofData), sb_vOffset = RigidBodyHandle::vNum,
         rbDofs = RigidBodyHandle::bodyNum * 12,
         FEE = FEE.port(), epsvh = epsv * dt, fricMu = fricMu,
         kinematicSatisfied = solver.kinematicSatisfied
         ] __device__(int feei) mutable {
            auto fee = FEE[feei];
            auto e0 = vData.pack(VProps::xn, fee[0]) - vData.pack(VProps::xHat, fee[0]);
            auto e1 = vData.pack(VProps::xn, fee[1]) - vData.pack(VProps::xHat, fee[1]);
            auto e2 = vData.pack(VProps::xn, fee[2]) - vData.pack(VProps::xHat, fee[2]);
            auto e3 = vData.pack(VProps::xn, fee[3]) - vData.pack(VProps::xHat, fee[3]);

            auto basis = fricEE.pack(FEEProps::basis, feei);
            auto fn = fricEE(FEEProps::fn, feei);
            auto gammas = fricEE.pack(FEEProps::gamma, feei);
            auto relDX3D = edge_edge_rel_dx(e0, e1, e2, e3, gammas[0], gammas[1]);
            auto relDX = basis.transpose() * relDX3D;
            auto relDXNorm2 = relDX.l2NormSqr();
            auto relDXNorm = zs::sqrt(relDXNorm2);
            auto f1_div_relDXNorm = zs::f1_SF_div_rel_dx_norm(relDXNorm2, epsvh);
            relDX *= f1_div_relDXNorm * fricMu * fn;
            auto TTTDX =
                -edge_edge_rel_dx_tan_to_mesh(relDX, basis, gammas[0], gammas[1]);
            // gradient
            int bodyA = reinterpret_bits<Ti>(vData(VProps::body, fee[0]));
            int bodyB = reinterpret_bits<Ti>(vData(VProps::body, fee[2]));
            auto Jea0 = vData.pack(VProps::J, fee[0]);
            auto Jea1 = vData.pack(VProps::J, fee[1]);
            auto Jeb0 = vData.pack(VProps::J, fee[2]);
            auto Jeb1 = vData.pack(VProps::J, fee[3]);

            vec12 grad;
            for (int vi = 0; vi < 4; vi++)
                for (int di = 0; di < 3; di++)
                    grad(vi * 3 + di) = TTTDX(vi, di);
            scatterContactForce(fee, grad, vData);

            relDX = basis.transpose() * relDX3D;
            auto TT = edge_edge_TT(basis, gammas[0], gammas[1]); // 2x12
            auto f2_term = f2_SF_term(relDXNorm2, epsvh);
            using HessT = zs::vec<T, 12, 12>;
            auto hess = HessT::zeros();
            if (relDXNorm2 >= epsvh * epsvh) {
                zs::vec<T, 2> ubar{-relDX[1], relDX[0]};
                hess = dyadic_prod(
                    TT.transpose() *
                        ((fricMu * fn * f1_div_relDXNorm / relDXNorm2) * ubar),
                    ubar * TT);
            } else {
                if (relDXNorm == 0) {
                    if (fn > 0)
                        hess = fricMu * fn * f1_div_relDXNorm * TT.transpose() * TT;
                    // or ignored
                } else {
                    auto innerMtr = dyadic_prod((f2_term / relDXNorm) * relDX, relDX);
                    innerMtr(0, 0) += f1_div_relDXNorm;
                    innerMtr(1, 1) += f1_div_relDXNorm;
                    innerMtr *= fricMu * fn;
                    //
                    make_pd(innerMtr);
                    hess = TT.transpose() * innerMtr * TT;
                }
            }

            if (bodyA >= 0 && bodyB >= 0) {
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA)); 
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB)); 
                zs::vec<int, 2> segStart {bodyA * 12, bodyB * 12};
                zs::vec<int, 2> segLen {12, 12};
                zs::vec<int, 2> segIsKin {isKinA, isKinB}; 
                mat12x24 J = mat12x24::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; ++j) {
                        J(i, j) = Jea0(i, j);
                        J(i + 3, j) = Jea1(i, j);
                        J(i + 6, j + 12) = Jeb0(i, j);
                        J(i + 9, j + 12) = Jeb1(i, j);
                    }
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q, hess_q, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA < 0 && bodyB < 0)
            {
                zs::vec<int, 4> segStart {(fee[0] - sb_vOffset) * 3 + rbDofs,
                                         (fee[1] - sb_vOffset) * 3 + rbDofs,
                                         (fee[2] - sb_vOffset) * 3 + rbDofs,
                                         (fee[3] - sb_vOffset) * 3 + rbDofs};
                zs::vec<int, 4> segLen {3, 3, 3, 3};
                zs::vec<int, 4> segIsKin {0, 0, 0, 0}; 
                for (int d = 0; d < 4; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, fee[d]));
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad, hess, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyA >= 0)
            {
                int isKinA = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyA)); 
                zs::vec<int, 3> segStart {bodyA * 12,
                                         (fee[2] - sb_vOffset) * 3 + rbDofs,
                                         (fee[3] - sb_vOffset) * 3 + rbDofs};
                zs::vec<int, 3> segLen {12, 3, 3};
                zs::vec<int, 3> segIsKin {isKinA, 0, 0}; 
                for (int d = 1; d < 3; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, fee[d]));
                auto J = zs::vec<T, 12, 18>::zeros();
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 12; ++j) {
                        J(i, j) = Jea0(i, j);
                        J(i + 3, j) = Jea1(i, j);
                    }
                for (int i = 0; i < 6; i++)
                    J(i + 6, i + 12) = 1;

                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q, hess_q, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied);
                return;
            }

            if (bodyB >= 0)
            {
                int isKinB = reinterpret_bits<Ti>(rbData(RBProps::isBC, bodyB)); 
                zs::vec<int, 3> segStart {(fee[0] - sb_vOffset) * 3 + rbDofs,
                                         (fee[1] - sb_vOffset) * 3 + rbDofs,
                                         bodyB * 12};
                zs::vec<int, 3> segLen {3, 3, 12};
                zs::vec<int, 3> segIsKin {0, 0, isKinB}; 
                for (int d = 0; d < 2; d++)
                    segIsKin[d] = reinterpret_bits<Ti>(vData(VProps::isBC, fee[d]));
                auto J = zs::vec<T, 12, 18>::zeros();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 12; j++)
                    {
                        J(6 + i, 6 + j) = Jeb0(i, j);
                        J(9 + i, 6 + j) = Jeb1(i, j);
                    }
                for (int i = 0; i < 6; i++)
                    J(i, i) = 1;
                auto grad_q = J.transpose() * grad;
                auto hess_q = J.transpose() * hess * J;
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad_q, hess_q, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied);
                return;
            }
        });

    return;
}
#endif // s_enableContact

template <typename T>
void ABDSolver::KinematicConstraintEnergy<T>::update(ABDSolver &solver, pol_t &pol, bool forGrad)
{
    static_assert(
        std::is_same_v<T, ABDSolver::T>,
        "The value type of InertialEnergy does not match the solver's.");

    solver.computeKinematicConstraints(pol);  
}

template <typename T>
T ABDSolver::KinematicConstraintEnergy<T>::energy(ABDSolver &solver, pol_t &pol)
{   
    static_assert(
        std::is_same_v<T, ABDSolver::T>,
        "The value type of InertialEnergy does not match the solver's.");

    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    T E = 0;

    if (!solver.kinematicSatisfied)
    {
        auto &es = solver.temp;
        auto &vData = solver.vData;
        auto &rbData = solver.rbData;

        auto boundaryKappa = solver.boundaryKappa;
        auto kinematicALCoef = solver.kinematicALCoef;

        es.resize(count_warps(rbData.size()));
        es.reset(0);
        pol(range(rbData.size()), 
            [es = proxy<space>(es), 
            rbData = view<space>(rbData), 
            boundaryKappa = boundaryKappa, 
            w = kinematicALCoef,
            N = rbData.size()] __device__ (int bi) mutable {
                T e = 0; 
                int isBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, bi)); 
                if (isBC)
                {
                    auto custom_stiffness = rbData(RBProps::BCStiffness, bi);
                    auto stiffness = custom_stiffness > 0 ? 
                                     custom_stiffness : boundaryKappa;
                    auto lambda = rbData.pack(RBProps::lambda, bi);
                    auto cons = rbData.pack(RBProps::cons, bi);
                    auto vol = rbData(RBProps::vol, bi);
                    e = lambda.dot(cons) + (T)0.5 * stiffness * cons.l2NormSqr();
                    // printf("Rigid lambda[%d] = %f %f %f %f %f %f %f %f %f %f %f %f\n", bi, 
                    //     (float)lambda[0], (float)lambda[1], (float)lambda[2], 
                    //     (float)lambda[3], (float)lambda[4], (float)lambda[5], 
                    //     (float)lambda[6], (float)lambda[7], (float)lambda[8], 
                    //     (float)lambda[9], (float)lambda[10], (float)lambda[11]);
                    // printf("Rigid cons[%d] = %f %f %f %f %f %f %f %f %f %f %f %f\n", bi, 
                    //     (float)cons[0], (float)cons[1], (float)cons[2], 
                    //     (float)cons[3], (float)cons[4], (float)cons[5], 
                    //     (float)cons[6], (float)cons[7], (float)cons[8], 
                    //     (float)cons[9], (float)cons[10], (float)cons[11]);
                    // e *= w * vol;
                }
                reduce_to(bi, N, e, es[bi / 32]);
        }); 
        E += reduce(pol, es);


        for (auto &softBody : solver.softBodies)
        {
            auto &verts = softBody.verts();
            auto customBCStiffness = softBody.bodyTypeProperties().customBCStiffness;
            es.resize(count_warps(verts.size()));
            es.reset(0); 
            pol(range(verts.size()), 
                [vOffset = softBody.voffset(), 
                es = proxy<space>(es), 
                vData = view<space>(vData, false_c, "vData"),
                n = verts.size(), 
                boundaryKappa = zs::max(customBCStiffness, boundaryKappa)] __device__ (int v_ofs) mutable {
                    // printf("v_ofs: %d, vOffset: %d\n", v_ofs, (int)vOffset);
                    int vi = v_ofs + vOffset; 
                    int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi)); 
                    T e = 0; 
                    if (isBC)
                    {
                        auto lambda = vData.pack(VProps::lambda, vi); 
                        auto cons = vData.pack(VProps::cons, vi); 
                        auto w = vData(VProps::ws, vi);
                        e = lambda.dot(cons) + (T)0.5 * boundaryKappa * cons.l2NormSqr(); 
                        // printf("lambda[%d] = %f %f %f\n", vi, (float)lambda[0], (float)lambda[0], (float)lambda[2]);
                        // printf("cons[%d] = %f %f %f\n", vi, (float)cons[0], (float)cons[0], (float)cons[2]);
                        // printf("boundaryKappa[%d] = %f\n", vi, (float)boundaryKappa);
                        // printf("energy kin[%d] = %f\n", vi, (float)e);
                        // e *= w;
                    }
                    reduce_to(v_ofs, n, e, es[v_ofs / 32]);
                }); 
            E += reduce(pol, es); // * kinematicALCoef;
        }
    }

#if LOG_ENERGY
    fmt::print("KinematicConstraintEnergy: {}\n", E);
#endif 
    return E;
}

template <typename T>
void ABDSolver::KinematicConstraintEnergy<T>::addGradientAndHessian(ABDSolver &solver, pol_t &pol)
{
    static_assert(
        std::is_same_v<T, ABDSolver::T>,
        "The value type of InertialEnergy does not match the solver's.");

    using namespace zs;
    using mat3 = ABDSolver::mat3;
    using mat12 = ABDSolver::mat12;
    constexpr auto space = execspace_e::cuda;

    auto &rbData = solver.rbData;
    auto &vData = solver.vData;
    auto &dofData = solver.dofData;
    auto &hess = solver.sysHess;
    auto kinematicALCoef = solver.kinematicALCoef;
    auto boundaryKappa = solver.boundaryKappa;

    if (!solver.kinematicSatisfied)
    {
        pol(range(rbData.size()), [rbData = view<space>(rbData),
                                dofData = view<space>(dofData),
                                sysHess = port<space>(hess),
                                w = kinematicALCoef,
                                boundaryKappa = boundaryKappa] __device__ (int bi) mutable {
            int isBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, bi)); 
            if (!isBC)
                return; 
            auto custom_stiffness = rbData(RBProps::BCStiffness, bi); 
            auto stiffness = custom_stiffness > 0 ? custom_stiffness : boundaryKappa; 
            auto vol = rbData(RBProps::vol, bi);
            auto H = mat12::zeros();
            for (int d = 0; d < 12; d++) {
                dofData(DOFProps::grad, bi * 12 + d) -= zs::sqrt(w * vol) * (rbData(RBProps::lambda, d, bi) + rbData(RBProps::cons, d, bi) * stiffness);
                H(d, d) = w * vol * stiffness;
#if !s_enableAutoPrecondition
                int di = d % 3; 
                rbData(RBProps::Pre, (d / 3) * 9 + 3 * di + di, bi) += w * vol * stiffness; 
#endif
            }

            // printf("Rigid kinematic hess[%d] norm: %f\n", bi, H.norm());
            sysHess.addRigidHessNoTile(bi, H, true, rbData);
        });        

        for (auto &softBody : solver.softBodies)
        {
            auto &verts = softBody.verts();
            auto stiffness = softBody.bodyTypeProperties().customBCStiffness > 0 ? 
                             softBody.bodyTypeProperties().customBCStiffness : boundaryKappa;

            pol(range(verts.size()), 
                [vData = view<space>(vData), 
                vOffset = softBody.voffset(),
                dofData = view<space>(dofData), 
                sysHess = port<space>(hess),
                kinematicALCoef = kinematicALCoef, 
                boundaryKappa = stiffness,
                sb_vOffset = RigidBodyHandle::vNum,
                rbDofs = RigidBodyHandle::bodyNum * 12] __device__ (int vi_ofs) mutable {
                    int vi = vi_ofs + vOffset; 
                    int sb_vi = vi - sb_vOffset;
                    int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi)); 
                    if (!isBC)
                        return; 
                    auto w = vData(VProps::ws, vi); 
                    auto H = mat3::zeros();
                    for (int d = 0; d < 3; d++)
                    {
                        dofData(DOFProps::grad, rbDofs + sb_vi * 3 + d) -= 
                            zs::sqrt(w * kinematicALCoef) * (vData(VProps::lambda, d, vi) + vData(VProps::cons, d, vi) * boundaryKappa); 
                        H(d, d) = w * kinematicALCoef * boundaryKappa;
#if !s_enableAutoPrecondition
                        vData(VProps::Pre, d * 3 + d, vi) += w * kinematicALCoef * boundaryKappa;  // TODO: check if correct
#endif
                    }
                    // printf("Vert ws[%d] = %f\n", vi, (float)w);
                    // printf("Soft vertex kinematic hess[%d] norm: %f\n", sb_vi, H.norm());
                    sysHess.addSoftHessNoTile(sb_vi, H, true, vData, sb_vOffset);
                }); 
        }
    } 
    else 
    {
        pol(range(rbData.size()), [rbData = view<space>(rbData),
                                sysHess = port<space>(hess),
                                w = kinematicALCoef,
                                boundaryKappa = boundaryKappa] __device__ (int bi) mutable {
            int isBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, bi)); 
            if (!isBC)
                return;
#if !s_enableAutoPrecondition
            for (int d = 0; d < 12; d++) {
                int di = d % 3; 
                rbData(RBProps::Pre, (d / 3) * 9 + 3 * di + di, bi) += 1.0f;
            }
#endif
            sysHess.addRigidHessNoTile(bi, mat12::identity(), true, rbData);  // TODO: check if correct
        });   
        pol(range(SoftBodyHandle::vNum), 
            [vData = view<space>(vData), 
            sysHess = port<space>(hess),
            kinematicALCoef = kinematicALCoef, 
            boundaryKappa = boundaryKappa, 
            sb_vOffset = RigidBodyHandle::vNum,
            rbDofs = RigidBodyHandle::bodyNum * 12] __device__ (int vi_ofs) mutable {
                int vi = vi_ofs + sb_vOffset; 
                int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi)); 
                if (!isBC)
                    return; 
#if !s_enableAutoPrecondition
                for (int d = 0; d < 3; d++)
                {
                    vData(VProps::Pre, d * 3 + d, vi) += 1.0f; 
                }
#endif
                // printf("Soft vertex kinematic hess[%d] norm: %f\n", vi_ofs, sqrt(3));
                sysHess.addSoftHessNoTile(vi_ofs, mat3::identity(), true, vData, sb_vOffset); // TODO: check if correct
        }); 
    }
}

template <typename T>
T ABDSolver::SpringConstraintEnergy<T>::energy(ABDSolver &solver, pol_t &pol)
{
    static_assert(
        std::is_same_v<T, ABDSolver::T>,
        "The value type of InertialEnergy does not match the solver's.");

    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    T E = 0;

    auto &es = solver.temp;
    auto &vData = solver.vData;
    auto &rbData = solver.rbData;

    auto springStiffness = solver.springStiffness;
    auto abdSpringStiffness = solver.abdSpringStiffness;

    if (springStiffness)
    {
        es.resize(count_warps(vData.size()));
        es.reset(0); 
        pol(range(vData.size()), 
            [vData = view<space>(vData), 
            n = vData.size(), 
            es = proxy<space>(es), 
            k = springStiffness] __device__ (int vi) mutable {
                T e = 0; 
                int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi)); 
                if (vData(VProps::springW, vi) > 0 && !isBC)
                {
                    auto m = vData(VProps::m, vi); 
                    auto xs = vData.pack(VProps::springTarget, vi); 
                    auto x = vData.pack(VProps::xn, vi); 
                    e = m * k * (x - xs).l2NormSqr(); 
                }
                reduce_to(vi, n, e, es[vi / 32]);
            }); 
        E += reduce(pol, es);
    }

    if (abdSpringStiffness)
    {
        es.resize(count_warps(rbData.size()));
        es.reset(0); 
        pol(range(rbData.size()),
            [rbData = view<space>(rbData), 
             k_spring = abdSpringStiffness, 
             es = proxy<space>(es), 
             n = rbData.size()
            ] __device__ (int bi) mutable {
                T e = 0; 
                auto has_spring = rbData(RBProps::hasSpring, bi); 
                if (has_spring)
                {
                    auto q = rbData.pack(RBProps::qn, bi); 
                    auto q_target = rbData.pack(RBProps::springTarget, bi); 
                    e += 0.5 * k_spring * (q - q_target).l2NormSqr(); 
                }
                reduce_to(bi, n, e, es[bi / 32]);
            }); 
        E += reduce(pol, es);
    }

#if LOG_ENERGY
    fmt::print("SpringConstraintEnergy: {}\n", E);
#endif 
    return E;
}

template <typename T>
void ABDSolver::SpringConstraintEnergy<T>::addGradientAndHessian(ABDSolver &solver, pol_t &pol)
{
    static_assert(
        std::is_same_v<T, ABDSolver::T>,
        "The value type of SpringConstraintEnergy does not match the solver's.");

    using namespace zs;
    using mat3 = ABDSolver::mat3;
    using mat12 = ABDSolver::mat12;
    constexpr auto space = execspace_e::cuda;

    auto &rbData = solver.rbData;
    auto &vData = solver.vData;
    auto &dofData = solver.dofData;
    auto &hess = solver.sysHess;

    auto kinematicSatisfied = solver.kinematicSatisfied;
    auto springStiffness = solver.springStiffness;
    auto abdSpringStiffness = solver.abdSpringStiffness;

    if (springStiffness)
        pol(range(vData.size()), 
            [rbData = view<space>(rbData), 
            vData = view<space>(vData), 
            dofData = view<space>(dofData), 
            sysHess = port<space>(hess), 
            k = springStiffness, 
            rbDofs = RigidBodyHandle::bodyNum * 12,
            sb_vOffset = RigidBodyHandle::vNum] __device__ (int vi) mutable {
                int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi)); 
                if (isBC)
                    return; 
                if (vData(VProps::springW, vi) > 0)
                {
                    int bi = reinterpret_bits<Ti>(vData(VProps::body, vi)); 
                    auto m = vData(VProps::m, vi); 
                    auto xs = vData.pack(VProps::springTarget, vi); 
                    auto x = vData.pack(VProps::xn, vi); 
                    auto coef = m * k * 2.0; 
                    auto grad_x = - coef * (x - xs);                          
                    if (bi < 0)
                    {
                        // flexible objects 
                        auto hess_x = mat3::identity() * coef;
                        for (int d = 0; d < 3; d++)
                        {
                            atomic_add(exec_cuda, &dofData(DOFProps::grad, rbDofs + (vi - sb_vOffset) * 3 + d), grad_x(d)); 
#if !s_enableAutoPrecondition
                            {
                                atomic_add(exec_cuda, &vData(VProps::Pre, 3 * d + d, vi), coef); 
                            }
#endif
                        }
                        sysHess.addSoftHessNoTile(vi - sb_vOffset, hess_x, true, vData, sb_vOffset);
                    } else {
                        // get body num; TODO: for flexible objects
                        // add spring energy and gradient 
                        // m * k * (x - xs) ** 2
                        auto grad_q = ABD_x2q(vData.pack(VProps::JVec, vi), grad_x); 
                        auto J = vData.pack(VProps::J, vi); 
                        auto hess_q = coef * J.transpose() * J;  
                        for (int d = 0; d < 12; d++)
                            atomic_add(exec_cuda, &dofData(DOFProps::grad, bi * 12 + d), grad_q(d)); 
#if !s_enableAutoPrecondition
                        {
                            for (int k = 0; k < 4; k++)
                                for (int di = 0; di < 3; di++)
                                    for (int dj = 0; dj < 3; dj++)
                                        atomic_add(exec_cuda, &rbData(RBProps::Pre, k * 9 + di * 3 + dj, bi), 
                                            hess_q(k * 3 + di, k * 3 + dj)); 
                        } 
#endif
                        sysHess.addRigidHessNoTile(bi, hess_q, true, rbData);
                    }
                }
            }); 
    
    if (abdSpringStiffness)
    {
        pol(range(RigidBodyHandle::bodyNum), 
            [dofData = view<space>(dofData), 
             rbData = view<space>(rbData), 
             vData = view<space>(vData), 
             sysHess = port<space>(hess), 
             rbDofs = RigidBodyHandle::bodyNum * 12,
             sb_vOffset = RigidBodyHandle::vNum,
             kinematicSatisfied = kinematicSatisfied, 
             k_spring = abdSpringStiffness] __device__ (int bi) mutable {
                auto has_spring = rbData(RBProps::hasSpring, bi); 
                if (!has_spring)
                    return; 
                int isBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, bi)); 
                auto q = rbData.pack(RBProps::qn, bi); 
                auto q_target = rbData.pack(RBProps::springTarget, bi); 
                auto grad = - (q - q_target) * k_spring; 
                auto hess = mat12::zeros(); 
                for (int d = 0; d < 12; d++)
                    hess(d, d) = k_spring; 
                vec<int, 1> segStart {bi * 12};
                vec<int, 1> segLen {12};
                vec<int, 1> segIsKin {isBC}; 
                scatterGradientAndHessian(segStart, segLen, segIsKin, grad, hess, dofData, rbData, vData, rbDofs, sb_vOffset, sysHess, kinematicSatisfied, true, false); 
            }); 
    }
}

template struct ABDSolver::InertialEnergy<ABDSolver::T>;
template struct ABDSolver::AffineEnergy<ABDSolver::T>;
template struct ABDSolver::SoftEnergy<ABDSolver::T>;
template struct ABDSolver::GroundBarrierEnergy<ABDSolver::T>;
template struct ABDSolver::BarrierEnergy<ABDSolver::T>;
template struct ABDSolver::KinematicConstraintEnergy<ABDSolver::T>;
template struct ABDSolver::SpringConstraintEnergy<ABDSolver::T>;
} // namespace tacipc
