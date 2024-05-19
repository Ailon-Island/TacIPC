#include <tacipc/solver/Solver.cuh>
#include <tacipc/solver/LinearSolver.hpp>
#include <zensim/math/matrix/Eigen.hpp>
#include <zensim/math/matrix/SparseMatrixOperations.hpp>
// CG/Direct Solve

#define DIRECTSOLVER_PRINT_GRAD 0
#define DIRECTSOLVER_PRINT_HESS 0
#define DIRECTSOLVER_PRINT_DIR 0
#define CGSOLVER_PRINT_PRE 0
#define CGSOLVER_PRINT_GRAD 0
#define CGSOLVER_PRINT_HESS 0
#define CGSOLVER_PRINT_DIR 0

namespace tacipc
{
namespace 
{
    Logger solveLogger{"tacipc.log"};
} // namespace anonymous
void ABDSolver::directSolve(pol_t &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    constexpr auto hostSpace = execspace_e::host;
    auto hostPol = OmpExecutionPolicy();

    DirectSolver solver;
    auto nnz = sysHess.spmat.nnz();
    auto dof = ABDSolver::dof();
    solver.allocate(dof, dof, nnz * 9 - dof); 
    solver.setNNZ(nnz * 9 - dof); 
    auto [ai, aj, ax] = solver.matData();
    auto b = solver.vecData();

#if DIRECTSOLVER_PRINT_HESS
    auto matHess = vec<T, 24, 24>::zeros();
#endif

    // copy hess to host and then to host solver
    auto spmat = sysHess.spmat.clone({zs::memsrc_e::host, -1});
    std::size_t k = 0;
    for (std::size_t row = 0; row != spmat._ptrs.size() - 1; ++row)
    {
        for (std::size_t loc = spmat._ptrs[row]; loc != spmat._ptrs[row + 1]; ++loc)
        {
            auto col = spmat._inds[loc];
            auto mat = spmat._vals[loc];
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = (row == col)?i:0; j < 3; ++j)
                {
                    ai[k] = row * 3 + i;
                    aj[k] = col * 3 + j;
                    ax[k] = mat(i, j);
                    // if (row * 3 +i < 24 && col * 3 + j < 24)
                    // {
                    //     matHess(row * 3 + i, col * 3 + j) = mat(i, j);
                    //     matHess(col * 3 + j, row * 3 + i) = mat(i, j);
                    // }
                    ++k;
                }
        }
    }

    // fmt::print("hess (first 24x24): \n");
    // for (int i = 0; i < 24; ++i)
    // {
    //     for (int j = 0; j < 24; ++j)
    //     {
    //         fmt::print("{:.15f}\t", matHess(i, j));
    //     }
    //     fmt::print("\n");
    // }

    // auto [eivals, eivecs] = eigen_decomposition(matHess);
    // for (int i = 0; i != 24; ++i) {
    //     if (eivals[i] < detail::deduce_numeric_epsilon<T>()) 
    //         fmt::print("NOT POSITIVE DEFINITE\n");
    //     fmt::print("eival[{}]: {:.15f}\n", i, eivals[i]);

    // }
    // fmt::print("hess diag (first 24): \n");
    // for (int i = 0; i < 24; ++i)
    // {
    //     fmt::print("{:.20f}\t", matHess(i, i));
    // }

    // copy grad to host and then to host solver
    temp.resize(dof);
    pol(range(dof),
        [temp = view<space>(temp),
         dofData = view<space>(dofData)] __device__ (int tid) mutable {
            temp(tid) = dofData(DOFProps::grad, tid);
        });
    auto h_temp = temp.clone({zs::memsrc_e::host, -1});
#if DIRECTSOLVER_PRINT_GRAD
    fmt::print("grad: \n");
#endif
    for (std::size_t i = 0; i < dof; ++i)
    {
        b[i] = h_temp.getVal(i);
#if DIRECTSOLVER_PRINT_GRAD
        fmt::print("{}\t", b[i]);
#endif
    }
#if DIRECTSOLVER_PRINT_GRAD
    fmt::print("\n");
#endif

    // solve
    solver.solve();

    // read data from solver and then copy to device
    auto result = solver.resultData();

#if DIRECTSOLVER_PRINT_DIR
    fmt::print("dir: \n");
#endif
    for (std::size_t i = 0; i < dof; ++i)
    {
        h_temp.setVal(result[i], i);
#if DIRECTSOLVER_PRINT_DIR
        fmt::print("{}\t", result[i]);
#endif
    }
#if DIRECTSOLVER_PRINT_DIR
    fmt::print("\n");
#endif
    temp = h_temp.clone({zs::memsrc_e::device, 0});
    pol(range(dof), 
        [temp = view<space>(temp),
         dofData = view<space>(dofData)] __device__ (int tid) mutable {
            dofData(DOFProps::dir, tid) = temp(tid);
        });
}

void ABDSolver::precondition(pol_t &pol)
{
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    pol(range(rbData.size()), 
        [dofData = view<space>(dofData), 
        rbData = view<space>(rbData)] __device__ (int bi) mutable {
            for (int d = 0; d < 12; d++)
                dofData(DOFProps::q, bi * 12 + d) = 0; 
            for (int d = 0; d < 4; d++)
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        dofData(DOFProps::q, bi * 12 + d * 3 + i) += 
                            rbData(RBProps::Pre, d * 9 + 3 * i + j, bi) * dofData(DOFProps::r, bi * 12 + d * 3 + j); 
        }); 
    pol(range(SoftBodyHandle::vNum), 
        [dofData = view<space>(dofData), 
        vData = view<space>(vData), 
        rbDofs = RigidBodyHandle::bodyNum * 12, 
        sb_vOffset = RigidBodyHandle::vNum] __device__ (int vi_ofs) mutable {
            // 'ofs' for 'offset' 
            for (int d = 0; d < 3; d++)
                dofData(DOFProps::q, rbDofs + vi_ofs * 3 + d) = 0; 
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    dofData(DOFProps::q, rbDofs + vi_ofs * 3 + i) += 
                        vData(VProps::Pre, 3 * i + j, sb_vOffset + vi_ofs) * dofData(DOFProps::r, rbDofs + vi_ofs * 3 + j); 
        }); 
}

void ABDSolver::multiplyHessian(pol_t &pol)
{
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    auto dof_3 = dof() / 3;

    auto &dx = temp3; // p
    auto &b = temp3_1;
    dx.resize(dof_3);
    b.resize(dof_3);

    // copy p->dx
    pol(range(dof_3),
        [dofData = view<space>(dofData),
         dx = view<space>(dx)] __device__ (int tid) mutable {
            dx(tid) = {
                dofData(DOFProps::p, 3 * tid),
                dofData(DOFProps::p, 3 * tid + 1),
                dofData(DOFProps::p, 3 * tid + 2),
            };
        });

    // multiply
#if s_enableZpcSpmv
    static_assert(true, "zpc SpMV does not support upper triangular matrix now!");
    spmv_classic(pol, sysHess.spmat, dx, b);
#else
    pol(range(dof_3),
        [b = view<space>(b)] __device__ (int tid) mutable {
            b(tid) = vec3::zeros();
        });
    pol(range(dof_3), 
        [hess = tacipc::ABDSolver::port<space>(sysHess),
        dx = view<space>(dx),
        b = view<space>(b)] __device__ (int row) mutable {
            auto bg = hess.spmat._ptrs[row];
            auto ed = hess.spmat._ptrs[row + 1];

            auto sum = vec3::zeros();
            for (auto i = bg; i < ed; ++i) 
            {
                auto col = hess.spmat._inds[i];
                if (row > col)
                    continue;
                auto mat = hess.spmat._vals[i];
                sum += mat * dx[col];
                if (row < col)
                {
                    auto symProd = mat.transpose() * dx[row];
                    for (int d = 0; d != 3; ++d)
                        atomic_add(zs::exec_cuda, &b[col].val(d), symProd.val(d));
                }
            }

            for (int d = 0; d != 3; ++d)
                atomic_add(zs::exec_cuda, &b[row].val(d), sum.val(d));
        });
#endif

    // copy b->temp
    pol(range(dof_3),
        [dofData = view<space>(dofData),
         b = view<space>(b)] __device__ (int tid) mutable {
            auto bi = b(tid);
            dofData(DOFProps::temp, 3 * tid) = bi[0];
            dofData(DOFProps::temp, 3 * tid + 1) = bi[1];
            dofData(DOFProps::temp, 3 * tid + 2) = bi[2];
        });
}

void ABDSolver::cgSolve(pol_t &pol) {
    // input "grad", multiply, constraints
    // output "dir"
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

#if CGSOLVER_PRINT_PRE
    auto matPre = vec<T, 24, 24>::zeros();
    {
        auto h_rbData_ = rbData.clone({zs::memsrc_e::host, -1});
        auto h_rbData = view<execspace_e::host>(h_rbData_);
        for (int bi = 0; bi < min(RigidBodyHandle::bodyNum, (std::size_t)2); bi++)
        {
            for (int d = 0; d < 4; d++)
            {
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        matPre(bi * 12 + d * 3 + i, bi * 12 + d * 3 + j) = h_rbData(RBProps::Pre, d * 9 + i * 3 + j, bi);
            }
        }
    }
    {
        auto h_vData_ = vData.clone({zs::memsrc_e::host, -1});
        auto h_vData = view<execspace_e::host>(h_vData_);
        for (int vi = 0; vi < min(SoftBodyHandle::vNum, 8 - RigidBodyHandle::bodyNum * 4); vi++)
        {
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    matPre(RigidBodyHandle::bodyNum * 12 + vi * 3 + i, RigidBodyHandle::bodyNum * 12 + vi * 3 + j) = h_vData(VProps::Pre, i * 3 + j, RigidBodyHandle::vNum + vi);
        }
    }
    solveLogger.log("preconditioner (first 24x24): \n");
    for (int i = 0; i < 24; ++i)
    {
        for (int j = 0; j < 24; ++j)
        {
            solveLogger.log("{:.15f}\t", matPre(i, j));
        }
        solveLogger.log("\n");
    }
#endif
#if CGSOLVER_PRINT_HESS
    auto matHess = vec<T, 24, 24>::zeros();
    {
        auto h_spmat = sysHess.spmat.clone({zs::memsrc_e::host, -1});
        for (std::size_t row = 0; row != h_spmat._ptrs.size() - 1; ++row)
        {
            for (std::size_t loc = h_spmat._ptrs[row]; loc != h_spmat._ptrs[row + 1]; ++loc)
            {
                auto col = h_spmat._inds[loc];
                auto mat = h_spmat._vals[loc];
                for (std::size_t i = 0; i < 3; ++i)
                    for (std::size_t j = (row == col)?i:0; j < 3; ++j)
                    {
                        if (row * 3 + i < 24 && col * 3 + j < 24)
                        {
                            matHess(row * 3 + i, col * 3 + j) = mat(i, j);
                            matHess(col * 3 + j, row * 3 + i) = mat(i, j);
                        }
                    }
            }
        }
    }
    solveLogger.log("hess (first 24x24): \n");
    for (int i = 0; i < 24; ++i)
    {
        for (int j = 0; j < 24; ++j)
        {
            solveLogger.log("{:.15f}\t", matHess(i, j));
        }
        solveLogger.log("\n");
    }
#endif
#if CGSOLVER_PRINT_GRAD
    {
        auto h_dofData_ = dofData.clone({zs::memsrc_e::host, -1});
        auto h_dofData = view<execspace_e::host>(h_dofData_);
        solveLogger.log("grad: \n");
        for (int i = 0; i < h_dofData.size(); ++i)
        {
            solveLogger.log("{:.15f}\t", h_dofData(DOFProps::grad, i));
        }
        solveLogger.log("\n");
    }
#endif

    // prepare for preconditioner 
    pol(range(rbData.size()), 
        [rbData = view<space>(rbData)] __device__ (int bi) mutable {
            for (int d = 0; d < 4; d++)
            {
                mat3 pre; 
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        pre(i, j) = rbData(RBProps::Pre, d * 9 + i * 3 + j, bi); 
                pre = inverse(pre);
                // TODO: float cgtemp? 
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        rbData(RBProps::Pre, d * 9 + i * 3 + j, bi) = pre(i, j); 
            }
        });
    pol(range(SoftBodyHandle::vNum), 
        [vData = view<space>(vData), 
        sb_vOffset = RigidBodyHandle::vNum] __device__ (int vi_ofs) mutable {
            vData.tuple(VProps::Pre, sb_vOffset + vi_ofs) = 
                inverse(vData.pack(VProps::Pre, sb_vOffset + vi_ofs)); 
        }); 

    // start cg 
    pol(range(dofData.size()),
            [dofData = view<space>(dofData)] __device__ (int i) mutable {
                dofData(DOFProps::dir, i) = 0;
                dofData(DOFProps::temp, i) = 0; // should be H * dir
            }); 
    pol(range(dofData.size()),
            [dofData = view<space>(dofData)] __device__ (int i) mutable {
                dofData(DOFProps::r, i) =
                    dofData(DOFProps::grad, i) - dofData(DOFProps::temp, i);
            });
    precondition(pol); // q = M^-1 * r
    pol(range(dofData.size()),
            [dofData = view<space>(dofData)] __device__ (int i) mutable {
                dofData(DOFProps::p, i) = dofData(DOFProps::q, i);
            });
    auto zTrk = dot(pol, dofData, DOFProps::r, DOFProps::q);
    auto residualPreconditionedNorm2 = zTrk;
    auto localTol2 = cgRel * cgRel * residualPreconditionedNorm2;
    int iter = 0;
    auto [npp, npe, npt, nee, nppm, npem, neem, ncspt, ncsee] = getCnts();

    for (; iter != CGCap; ++iter) {
        if (iter % (CGCap / 5) == 0)
        // if (iter % 1 == 0)
        {
            fmt::print("cg iter: {}, norm2: {} (zTrk: {}) npp: {}, npe: {}, "
                       "npt: {}, nee: {}, nppm: {}, npem: {}, neem: {}, ncspt: "
                       "{}, ncsee: {}\n",
                       iter, residualPreconditionedNorm2, zTrk, npp, npe, npt,
                       nee, nppm, npem, neem, ncspt, ncsee);
            fmt::print("localTol2: {}\n", localTol2);
        }
        if (iter < 25 || iter % 25 == 0)
        {
#if CGSOLVER_PRINT_DIR
            {
                auto h_dofData_ = dofData.clone({zs::memsrc_e::host, -1});
                auto h_dofData = view<execspace_e::host>(h_dofData_);
                solveLogger.log("[iter {}] dir: \n", iter);
                for (int i = 0; i < h_dofData.size(); ++i)
                {
                    solveLogger.log("{:.15f}\t", h_dofData(DOFProps::dir, i));
                }
                solveLogger.log("\n");
            }
#endif
        }

        if (residualPreconditionedNorm2 <= localTol2)
            break;
        multiplyHessian(pol); // temp = H * p
        T alpha = zTrk / dot(pol, dofData, DOFProps::temp, DOFProps::p);
        pol(range(dofData.size()),
                [dofData = view<space>(dofData),
                 alpha] __device__ (int i) mutable {
                    dofData(DOFProps::dir, i) += alpha * dofData(DOFProps::p, i);
                    dofData(DOFProps::r, i) -= alpha * dofData(DOFProps::temp, i);
                });
        precondition(pol); // r, q 
        auto zTrkLast = zTrk;
        zTrk = dot(pol, dofData, DOFProps::q, DOFProps::r);
        auto beta = zTrk / zTrkLast;
        pol(range(dofData.size()),
                [dofData = view<space>(dofData),
                 beta] __device__ (int zi) mutable {
                    dofData(DOFProps::p, zi) = dofData(DOFProps::q, zi) + beta * dofData(DOFProps::p, zi);
                });
        residualPreconditionedNorm2 = zTrk;
    } // end cg step
    fmt::print("[cg solver] ends at iter: {}, norm2: {} (zTrk: {}), localTol2: {}, npp: {}, npe: {}, "
                "npt: {}, nee: {}, nppm: {}, npem: {}, neem: {}, ncspt: "
                "{}, ncsee: {}\n",
                iter, residualPreconditionedNorm2, zTrk, localTol2, npp, npe, npt,
                nee, nppm, npem, neem, ncspt, ncsee);
}

} // namespace tacipc
