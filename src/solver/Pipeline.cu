#include <tacipc/solver/Solver.cuh>
// newton iteration...

namespace tacipc
{

void ABDSolver::step(pol_t &pol, bool updateBody)
{
    zs::CppTimer timer;
    timer.tick();
    stepInitialize(pol);
    fmt::print("[kappa-dhat-check]\tcheck kappa: {}, dHat: {}\n", kappa, dHat); 
    if (!useAbsKappaDhat)
        suggestKappa(pol);
    for (int i = 0; i < frameSubsteps; i++)
    {
        substep(pol);
    }
    timer.tock();
    totalSimulationElapsed += timer.elapsed();

    // if necessary, update data to bodies
    timer.tick();
    if (updateBody)
    {
        update(pol);
    }
    timer.tock("update CPU data");
    totalCPUUpdateElapsed += timer.elapsed();
    totalSimulationElapsed += timer.elapsed();
}
void ABDSolver::step(pol_t &pol, T frameDt, bool updateBody)
{
    this->frameDt = frameDt;
    this->dt = frameDt / frameSubsteps;
    step(pol, updateBody);
} 
void ABDSolver::step(pol_t &pol, T frameDt, int frameSubsteps, bool updateBody)
{
    this->frameDt = frameDt;
    this->frameSubsteps = frameSubsteps;
    this->dt = frameDt / frameSubsteps;
    step(pol, updateBody);
}
void ABDSolver::step(pol_t &pol, int frameSubsteps, bool updateBody)
{
    this->frameDt = dt * frameSubsteps;
    this->frameSubsteps = frameSubsteps;
    step(pol, updateBody);
}

void ABDSolver::substep(pol_t &pol)
{
    bodyUpToDate = false; // tag for body update
    substepInitialize(pol);
    fmt::print("substep: {}\n", substeps);
    int numFricSolve = (enableFriction && fricMu != 0) ? 2 : 1;
    int nkCnt = 0;
    while (numFricSolve-- > 0)
    {
        updateBasis = true;
        auto success = newtonKrylov(pol);
    }
    updateVelocities(pol);
}

ABDSolver::T ABDSolver::kinematicConstraintResidual(pol_t &pol)
{
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    Vector<T> num{vData.get_allocator(), rbData.size() + 1}, 
              den{vData.get_allocator(), rbData.size() + 1};
    pol(range(rbData.size() + 1), 
        [rbData = view<space>(rbData),
        w = kinematicALCoef,
        num = proxy<space>(num),
        den = proxy<space>(den), 
        rbDataSize = rbData.size()] __device__ (int bi) mutable {
            if (bi == rbDataSize)
            {
                num[bi] = 0; 
                den[bi] = 0; 
                return; 
            }
            int isBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, bi)); 
            if (isBC == 0)
            {
                num[bi] = 0; 
                den[bi] = 0; 
            } else {
                auto qn = rbData.pack(RBProps::qn, bi);
                auto qhat = rbData.pack(RBProps::qHat, bi);
                auto q_tilde = rbData.pack(RBProps::qTilde, bi);
                num[bi] = (qn - q_tilde).l2NormSqr();
                den[bi] = (qhat - q_tilde).l2NormSqr(); 
            }
        });
    // TODO: improve performance by mapping BCVerts index to num, den index instead of atomic_add 
    pol(range(SoftBodyHandle::vNum), 
        [vData = view<space>(vData), 
        num = proxy<space>(num), 
        den = proxy<space>(den), 
        ofs = RigidBodyHandle::bodyNum, sb_vOffset = RigidBodyHandle::vNum] __device__ (int v_ofs) mutable {
            int vi = v_ofs + sb_vOffset;
            int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi));
            if (!isBC)
                return; 
            auto xn = vData.pack(VProps::xn, vi); 
            auto xhat = vData.pack(VProps::xHat, vi);
            auto xtilde = vData.pack(VProps::xTilde, vi); 
            atomic_add(exec_cuda, &num[ofs], (xn - xtilde).l2NormSqr()); 
            atomic_add(exec_cuda, &den[ofs], (xhat - xtilde).l2NormSqr()); 
        }); 
    auto nsqr = reduce(pol, num);
    auto dsqr = reduce(pol, den);
    fmt::print("nsqr: {}, dsqr: {}\n", nsqr, dsqr); 
    T ret = 0;
    if (dsqr == 0)
        ret = std::sqrt(nsqr);
    else
        ret = std::sqrt(nsqr / dsqr);
    return ret < consTol ? 0 : ret;
    // return ret < 1e-2 ? 0 : ret;
}

void ABDSolver::computeKinematicConstraints(pol_t &pol)
{
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    pol(range(rbData.size()), [rbData = view<space>(rbData), 
                            w = kinematicALCoef] __device__ (int bi) mutable {
        int isBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, bi));
        if (!isBC)
            return; 
        auto vol = rbData(RBProps::vol, bi);
        rbData.tuple(RBProps::cons, bi) = zs::sqrt(w * vol) * (rbData.pack(RBProps::qn, bi) - rbData.pack(RBProps::qTilde, bi));
    });
    // vData isBC
    pol(range(SoftBodyHandle::vNum), 
        [vData = view<space>(vData), 
        kinematicALCoef = kinematicALCoef, 
        sb_vOffset = RigidBodyHandle::vNum] __device__ (int vi_ofs) mutable {
            int vi = vi_ofs + sb_vOffset;
            auto w = vData(VProps::ws, vi);
            int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi));
            if (!isBC)
                return; 
            vData.tuple(VProps::cons, vi) = zs::sqrt(kinematicALCoef * w) * 
                (vData.pack(VProps::xn, vi) - vData.pack(VProps::xTilde, vi)); 
        }); 
}

void ABDSolver::updateKinematicConstraintsLambda(pol_t &pol)
{
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    computeKinematicConstraints(pol); 
    fmt::print("\t\tupdateKinematicConstraintsLambda\n"); 
    pol(range(rbData.size()), [rbData = view<space>(rbData), 
                            boundaryKappa = boundaryKappa, 
                            w = kinematicALCoef] __device__ (int bi) mutable {
        int isBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, bi));
        if (!isBC)
            return; 
        
        auto cons = rbData.pack(RBProps::cons, bi);
        auto lambda = rbData.pack(RBProps::lambda, bi);
        rbData.tuple(RBProps::lambda, bi) = lambda + 
            boundaryKappa * rbData.pack(RBProps::cons, bi);
    });
    pol(range(SoftBodyHandle::vNum), 
        [vData = view<space>(vData), 
        boundaryKappa = boundaryKappa, 
        sb_vOffset = RigidBodyHandle::vNum] __device__ (int v_ofs) mutable {
            int vi = v_ofs + sb_vOffset; 
            int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi));
            if (!isBC)
                return; 

            auto cons = vData.pack(VProps::cons, vi);
            auto lambda = vData.pack(VProps::lambda, vi);
            vData.tuple(VProps::lambda, vi) = vData.pack(VProps::lambda, vi) + 
                boundaryKappa * vData.pack(VProps::cons, vi);
        }); 
}

// get gradient and hessian
// solve update direction
// line search
// update
bool ABDSolver::newtonKrylov(pol_t &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    float totalNewtonPrecomputeElapsed = 0;
    float totalNewtonGradHessElapsed = 0;  
    float totalNewtonCGElapsed = 0;  
    float totalNewtonUpdateElapsed = 0; 
    bool success = false; 
    kinematicSatisfied = false; 

    for (int newtonIter = 0; newtonIter != PNCap; ++newtonIter) {
        zs::CppTimer timer; 
        timer.tick(); 
        T cons_res; // constraint residual

        /// compute constraint residual
        if (!enableSoftBC && !kinematicSatisfied)
        {
            cons_res = kinematicConstraintResidual(pol);  
            if (cons_res == 0)
            {
                fmt::print("\t\tkinematicSatisfied!\n"); 
                kinematicSatisfied = true; 
            }
        }

        /// precompute contact
#if !s_enableEnergyUpdate
        if (enableContact) {
            findBarrierCollisions(pol, xi);
            if (!useAbsKappaDhat)
                suggestKappa(pol);
#if s_enableFriction
            if ((enableFriction || enableBoundaryFriction) && updateBasis)
                if (fricMu != 0) {
                    precomputeFrictions(pol, xi);
                }
#endif
        }
#endif

        auto check_grad = [&dofData = dofData, &pol](const SmallString& info)
        {
            pol(range(dofData.size()),
                     [dofData = view<space>(dofData), 
                      info] __device__ (int di)
            {
                printf("%s: grad(%d) = %f\n", info.asChars(), di, (float)dofData(DOFProps::grad, di));
            });
        };
        auto check_dir = [&dofData = dofData, &pol](const SmallString& info)
        {
            pol(range(dofData.size()),
                     [dofData = view<space>(dofData), 
                      info] __device__ (int di)
            {
                printf("%s: dir(%d) = %f\n", info.asChars(), di, (float)dofData(DOFProps::dir, di));
            });
        };
        // TODO: make sure that rbData's "grad" is only for affine body contact force computation 
       pol(zs::range(rbData.size()), [rbData = view<space>(rbData)
            ] __device__(int bi) mutable {
            rbData.tuple(RBProps::Pre, bi) = zs::vec<T, 3, 3, 4>::zeros();
            rbData.tuple(RBProps::contact, bi) = vec12::zeros();  // TODO: use tag name "force" instead of "grad"
       });
       pol(zs::range(vData.size()), [vData = view<space>(vData)
            ] __device__ (int vi) mutable {
            vData.tuple(VProps::Pre, vi) = zs::vec<T, 3, 3>::zeros(); 
            vData.tuple(VProps::contact, vi) = vec3::zeros(); 
       }); 
        dofData.fillZeros();
        timer.tock("newton_precompute"); 
        totalNewtonPrecomputeElapsed += timer.elapsed(); 

        timer.tick(); 
        addGradientAndHessian(pol);
        timer.tock("newton_grad_hess");
        totalNewtonGradHessElapsed += timer.elapsed(); 
        totalGradHessElapsed += timer.elapsed();
        
        timer.tick(); 
        if (enableCG)
        {
#if s_enableCGSolver 
            if constexpr (false)
            { // remove precondition for debug
                pol(range(rbData.size()),
                    [rbData = view<space>(rbData)] __device__ (int bi) mutable {
                        rbData.tuple(RBProps::Pre, bi) = zs::vec<T, 3, 3, 4>::zeros();
                        for (int k = 0; k < 4; k++)
                            for (int i = 0; i < 3; i++)
                                rbData(RBProps::Pre, k * 9 + i * 3 + i, bi) = 1;
                    });
                pol(range(SoftBodyHandle::vNum),
                    [vData = view<space>(vData),
                    sb_vOffset = RigidBodyHandle::vNum] __device__ (int id) mutable {
                        int vi = sb_vOffset + id;
                        vData.tuple(VProps::Pre, vi) = zs::vec<T, 3, 3>::identity();
                    });
            }
            cgSolve(pol);
#else 
            fmt::print("CG solver is disabled at compile time.\n");
            directSolve(pol);
#endif 
        }
        else
            directSolve(pol);

        timer.tock("newton_linear_system_solve"); 
        totalNewtonCGElapsed += timer.elapsed(); 
        totalLinearSolveElapsed += timer.elapsed();

        // project kinematic dof is kinematic constraints are already satisfied; NOTE: assuming NO LINEAR CONSTRAINTS EXISTS
        // remove newton directions for BC objects
        if (kinematicSatisfied)
        {
            pol(range(rbData.size()), 
                [dofData = view<space>(dofData), 
                rbData = view<space>(rbData)] __device__ (int bi) mutable {
                    int isBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, bi)); 
                    if (isBC == 0)
                        return; // skip non-kinematic objects 
                    for (int d = 0; d < 12; d++)
                        dofData(DOFProps::dir, bi * 12 + d) = 0; 
            }); 
            pol(range(SoftBodyHandle::vNum), 
                [dofData = view<space>(dofData), 
                vData = view<space>(vData), 
                rbDofs = RigidBodyHandle::bodyNum * 12, sb_vOffset = RigidBodyHandle::vNum] __device__ (int id) mutable {
                    int vi = sb_vOffset + id;
                    int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi)); 
                    if (isBC)
                        for (int d = 0; d < 3; d++)
                            dofData(DOFProps::dir, rbDofs + id * 3 + d) = 0; 
                }); 
        }
        // check_grad("gradient check");
        // check_dir("direction check");

        timer.tick(); 
        T res = infNorm(pol, dofData, DOFProps::dir) / dt;
        if (!enableSoftBC)
        {
            cons_res = kinematicConstraintResidual(pol); 
            fmt::print("\tcons_res: {}\n", cons_res);             
        }
        if (res < targetGRes && (enableSoftBC || (cons_res == 0))) {
            success = true;
            fmt::print("\t# newton optimizer ends in {} iters with residual {}\n", newtonIter, res);
            fmt::print(fg(fmt::color::cyan), "{}: {} ms\n", "Total Newton Precompute", totalNewtonPrecomputeElapsed);
            fmt::print(fg(fmt::color::cyan), "{}: {} ms\n", "Total Newton Grad Hess", totalNewtonGradHessElapsed);
            fmt::print(fg(fmt::color::cyan), "{}: {} ms\n", "Total Newton CG", totalNewtonCGElapsed);
            fmt::print(fg(fmt::color::cyan), "{}: {} ms\n", "Total Newton Update", totalNewtonUpdateElapsed);
            break;
        }

        fmt::print(fg(fmt::color::aquamarine),
                   "newton iter {}: direction residual(/dt) {}, "
                   "grad residual {}, targetGRes: {}\n",
                   newtonIter, res, infNorm(pol, dofData, DOFProps::dir), targetGRes);

        pol(zs::range(vData.size()),
                [vData = view<space>(vData), rbData = view<space>(rbData),
                 dofData = view<space>(dofData), sb_vOffset = RigidBodyHandle::vNum, rbDofs = RigidBodyHandle::bodyNum * 12] __device__(int vi) mutable {
                    if (vi < sb_vOffset)
                    {
                        int bi = reinterpret_bits<Ti>(vData(VProps::body, vi));
                        auto J = vData.pack(VProps::J, vi);
                        vec12 q_dir;
                        for (int d = 0; d < 12; d++)
                            q_dir(d) = dofData(DOFProps::dir, bi * 12 + d);
                        vData.tuple(VProps::dir, vi) = J * q_dir;
                    } else {
                        for (int d = 0; d < 3; d++)
                            vData(VProps::dir, d, vi) = dofData(DOFProps::dir, rbDofs + (vi - sb_vOffset) * 3 + d);
                    }
        });
    
        pol(zs::range(vData.size()), [vData = view<space>(vData)] __device__(int i) mutable {
            vData.tuple(VProps::xn0, i) = vData.pack(VProps::xn, i);
        });
        pol(zs::range(rbData.size()), [rbData = view<space>(rbData)] __device__(int ai) mutable {
            rbData.tuple(RBProps::qn0, ai) = rbData.pack(RBProps::qn, ai);
            // printf("bi: %d, qn:", ai);
            // for (int i = 0; i < 12; i++)
            //     printf("%f ", rbData.pack<12>("qn", ai)[i]);
            // printf("\n");
        });
        // line search
        auto alpha = prepareLineSearch(pol);
#if s_enableLineSearch
        alpha = lineSearch(pol, alpha);
        fmt::print("Newton stepsize: {}\n", alpha);
#endif
        // ABD: update q and x
        pol(zs::range(RigidBodyHandle::bodyNum),
                [dofData = view<space>(dofData),
                 rbData = view<space>(rbData),
                 alpha = alpha] __device__ (int bi) mutable {
                    for (int d = 0; d < 12; d++)
                    {
                        rbData(RBProps::qn, d, bi) = rbData(RBProps::qn0, d, bi) + alpha * dofData(DOFProps::dir, bi * 12 + d);
                    }
                });
        pol(zs::range(SoftBodyHandle::vNum),
                [dofData = view<space>(dofData),
                 rbDofs = RigidBodyHandle::bodyNum * 12, alpha = alpha, sb_vOffset = RigidBodyHandle::vNum,
                 vData = view<space>(vData)] __device__ (int id) mutable {
                    for (int d = 0; d < 3; d++)
                        vData(VProps::xn, d, id + sb_vOffset) =
                            vData(VProps::xn0, d, id + sb_vOffset)
                            + alpha * dofData(DOFProps::dir, rbDofs + id * 3 + d);
                });
        // for affine bodies, update vData's x from q
        pol(zs::range(RigidBodyHandle::vNum),
                [rbData = view<space>(rbData),
                 vData = view<space>(vData)] __device__ (int vi) mutable {
                    int bi = reinterpret_bits<Ti>(vData(VProps::body, vi));
                    auto J = vData.pack(VProps::J, vi);
                    vData.tuple(VProps::xn, vi) = J * rbData.pack(RBProps::qn, bi);
        });
        
        // for plastic bodies, update IB matrix 
        // updatePlasticDeformation(pol); 
        timer.tock("newton_update"); 
        totalNewtonUpdateElapsed += timer.elapsed(); 

        if (!enableSoftBC)
        {
            cons_res = kinematicConstraintResidual(pol); 
            fmt::print("res * dt: {}, kinematicALTol: {}, cons_res: {}, consTol: {}\n", 
                res * dt, kinematicALTol, cons_res, consTol); 
            if (!kinematicSatisfied && res * dt < kinematicALTol && cons_res > consTol)
                updateKinematicConstraintsLambda(pol);             
        }
    } // end newton step

    return success;
}

void ABDSolver::updateEnergy(ABDSolver::pol_t &pol, bool forGrad)
{
    for (auto &e : energies)
    {
        e->update(*this, pol, forGrad);
    }
}

ABDSolver::T ABDSolver::energy(ABDSolver::pol_t &pol)
{
    zs::CppTimer timer;
    timer.tick();
#if s_enableEnergyUpdate
    updateEnergy(pol, false); // forGrad = false
#endif

    T E = 0;
    for (auto &e : energies)
    {
        E += e->energy(*this, pol);
    }
    timer.tock("energy");
    totalEnergyElapsed += timer.elapsed();
    return E;
}

void ABDSolver::addGradientAndHessian(ABDSolver::pol_t &pol)
{
#if s_enableEnergyUpdate
    updateEnergy(pol, true); // forGrad = true
#endif

    sysHess.clear();
    for (auto &e : energies)
    {
        e->addGradientAndHessian(*this, pol);
    }
    sysHess.buildInit();
    sysHess.build();
}

ABDSolver::T ABDSolver::prepareLineSearch(ABDSolver::pol_t &pol)
{
    zs::CppTimer timer;
    timer.tick();
    T alpha = 1.;
    if (enableGround) {
        alpha = groundCCD(pol, alpha);
        fmt::print("\tstepsize after ground ccd: {}\n", alpha);
    }
    if (enableContact)
    {
        findCCDCollisions(pol, alpha, xi);
        auto [npp, npe, npt, nee, nppm, npem, neem, ncspt, ncsee] = getCnts();
        alpha = CCD(pol, (T)0.2, alpha);
        fmt::print("\tstepsize after ccd: {}. (ncspt: {}, ncsee: {})\n", alpha, ncspt, ncsee);
    }
    if (enableInversionPrevention)
    {
        alpha = inversionPreventCCD(pol, alpha);
        fmt::print("\tstepsize after inversion prevention: {}\n", alpha);
    }
    timer.tock("CCD linesearch");
    totalCCDElapsed += timer.elapsed();

    return alpha;
}

ABDSolver::T ABDSolver::lineSearch(ABDSolver::pol_t &pol, ABDSolver::T alpha)
{
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    // initial energy 
    T E0 = energy(pol);

    T E{E0};
    T c1m = 0;
    int lsIter = 0;
    c1m = armijoParam * dot(pol, dofData, DOFProps::dir, DOFProps::grad);
    if (c1m < 0) {
        fmt::print(fg(fmt::color::light_yellow), "c1m < 0!!!\n");
        c1m = 0; 
    }
    fmt::print(fg(fmt::color::white), "c1m : {}\n", c1m);
    do {
        pol(zs::range(rbData.size()),
                [dofData = view<space>(dofData),
                 rbData = view<space>(rbData),
                 alpha = alpha] __device__ (int bi) mutable {
                    for (int d = 0; d < 12; d++)
                        rbData(RBProps::qn, d, bi) = rbData(RBProps::qn0, d, bi) + alpha * dofData(DOFProps::dir, bi * 12 + d);
                });
        pol(zs::range(SoftBodyHandle::vNum),
                [dofData = view<space>(dofData),
                 rbDofs = RigidBodyHandle::bodyNum * 12, alpha = alpha, sb_vOffset = RigidBodyHandle::vNum,
                 vData = view<space>(vData)] __device__ (int id) mutable {
                    for (int d = 0; d < 3; d++)
                        vData(VProps::xn, d, sb_vOffset + id) =
                            vData(VProps::xn0, d, sb_vOffset + id)
                            + alpha * dofData(DOFProps::dir, rbDofs + id * 3 + d);
                });
        pol(zs::range(RigidBodyHandle::vNum),
                [vData = view<space>(vData),
                 rbData = view<space>(rbData)]__device__(int vi) mutable {
            int bi = reinterpret_bits<Ti>(vData(VProps::body, vi));
            auto J = vData.pack(VProps::J, vi);
            vData.tuple(VProps::xn, vi) = J * rbData.pack(RBProps::qn, bi);
        });
#if !s_enableEnergyUpdate
#if s_enableContact 
        if (enableContact)
            findBarrierCollisions(pol, xi);
#endif
#endif
        E = energy(pol);

        fmt::print("E: {} at alpha {}. E0 {}\n", E, alpha, E0);
        if (E <= E0 + alpha * c1m)
            break;

        if (lsIter > 10) {
            fmt::print(fg(fmt::color::light_yellow), "linesearch early exit with alpha {}\n", alpha);
            break;
        }
        alpha /= 2;
        if (++lsIter > 30) {
            // ABD： remove constrain residue output
            fmt::print(
                "too small stepsize at iteration [{}]! alpha: {}n",
                lsIter, alpha);
            getchar();
        }
    } while (true);

    return alpha;
}

void ABDSolver::updateVelocities(ABDSolver::pol_t &pol)
{
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    // ABD: update q_dot_n instead of vn
    pol(range(rbData.size()),
        [rbData = view<space>(rbData),
         dt = dt] __device__ (int ai) mutable {
            int isBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, ai)); 
            if (isBC)
                return; 
            auto newQ = rbData.pack(RBProps::qn, ai);
            auto dq_dot = (newQ - rbData.pack(RBProps::qTilde, ai)) / dt;
            auto q_dot = rbData.pack(RBProps::qDot, ai);
            q_dot += dq_dot;
            rbData.tuple(RBProps::qDot, ai) = q_dot;
            // printf("q_dot[%d]: %f %f %f %f %f %f %f %f %f %f %f %f\n", (int)ai, (float)q_dot[0], (float)q_dot[1], (float)q_dot[2], (float)q_dot[3], (float)q_dot[4], (float)q_dot[5], (float)q_dot[6], (float)q_dot[7], (float)q_dot[8], (float)q_dot[9], (float)q_dot[10], (float)q_dot[11]);
    });

    pol(range(SoftBodyHandle::vNum),
        [vData = view<space>(vData),
        dt = dt, rb_vOffset = RigidBodyHandle::vNum] __device__ (int id) mutable {
            auto vi = rb_vOffset + id;
            int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi)); 
            if (isBC)
                return; 
            auto newX = vData.pack(VProps::xn, vi);
            auto dv = (newX - vData.pack(VProps::xTilde, vi)) / dt;
            auto vn = vData.pack(VProps::vn, vi);
            vn += dv;
            vData.tuple(VProps::vn, vi) = vn;

            // printf("v mass[%d]: %f\n", (int)vi, (float)vData(VProps::m, vi));
            // printf("v vn[%d]: %f %f %f\n", (int)vi, (float)vn[0], (float)vn[1], (float)vn[2]);
    });
}

void ABDSolver::update(pol_t &pol) {
    if (bodyUpToDate)
        return;

    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    for (auto& sbHandle : softBodies) {
        auto &verts = sbHandle.verts();
        pol(zs::range(verts.size()),
            [vData = view<space>(vData), verts = view<space>({}, verts),
             xTag = verts.getPropertyOffset("x"),
             vTag = verts.getPropertyOffset("v"),
             contactTag = verts.getPropertyOffset("contact"),
             friction_force_tag = verts.getPropertyOffset("friction_force"),
             collision_force_tag = verts.getPropertyOffset("collision_force"),
             elastic_force_tag = verts.getPropertyOffset("elastic_force"),
             dt = dt, vOffset = sbHandle.voffset()] __device__(int vi) mutable {
                verts.tuple<3>(xTag, vi) = vData.pack(VProps::xn, vOffset + vi);
                verts.tuple<3>(vTag, vi) = vData.pack(VProps::vn, vOffset + vi);
                verts.tuple<3>(contactTag, vi) = vData.pack(VProps::contact, vOffset + vi) / (dt * dt);
                verts.tuple<3>(friction_force_tag, vi) = vData.pack(VProps::friction_force, vOffset + vi) / (dt * dt);
                verts.tuple<3>(collision_force_tag, vi) = vData.pack(VProps::collision_force, vOffset + vi) / (dt * dt);
                verts.tuple<3>(elastic_force_tag, vi) = vData.pack(VProps::elastic_force, vOffset + vi) / (dt * dt);
            });
    }

    temp12.resize(3 * RigidBodyHandle::bodyNum); 
    pol(range(RigidBodyHandle::bodyNum), 
        [temp12 = view<space>(temp12), 
         rbData = view<space>(rbData), 
         dt = dt] __device__ (int rbi) mutable {
            temp12[rbi * 3] = rbData.pack(RBProps::qn, rbi);
            temp12[rbi * 3 + 1] = rbData.pack(RBProps::qDot, rbi); 
            temp12[rbi * 3 + 2] = rbData.pack(RBProps::contact, rbi) / (dt * dt); 
        }); 
    for (auto&& [rbi, rbHandle] : enumerate(rigidBodies)) {
        auto &verts = rbHandle.verts();
        // update velocity and positions
        pol(range(verts.size()),
            [vData = view<space>(vData), verts = view<space>({}, verts),
             xTag = verts.getPropertyOffset("x"),
             vTag = verts.getPropertyOffset("v"),
             contactTag = verts.getPropertyOffset("contact"),
             friction_force_tag = verts.getPropertyOffset("friction_force"),
             collision_force_tag = verts.getPropertyOffset("collision_force"),
             elastic_force_tag = verts.getPropertyOffset("elastic_force"),
             dt = dt, vOffset = rbHandle.voffset(), rbi = rbi,
             rbData = view<space>(rbData)] __device__(int vi) mutable {
                verts.tuple<3>(xTag, vi) = vData.pack(VProps::xn, vOffset + vi);
                verts.tuple<3>(vTag, vi) = ABD_q2x(vData.pack(VProps::JVec, vOffset + vi), 
                    rbData.pack(RBProps::qDot, rbi)); 
                verts.tuple<3>(contactTag, vi) = vData.pack(VProps::contact, vOffset + vi) / (dt * dt);
                verts.tuple<3>(friction_force_tag, vi) = vData.pack(VProps::friction_force, vOffset + vi) / (dt * dt);
                verts.tuple<3>(collision_force_tag, vi) = vData.pack(VProps::collision_force, vOffset + vi) / (dt * dt);
                verts.tuple<3>(elastic_force_tag, vi) = vec3::zeros(); 
            });
        auto &bodyTypeProperties = rbHandle.bodyTypeProperties();
        bodyTypeProperties.q = temp12.getVal(rbi * 3);
        bodyTypeProperties.v = temp12.getVal(rbi * 3 + 1);
        bodyTypeProperties.contactForce = temp12.getVal(rbi * 3 + 2);
    }
    for (auto&& [bi, sbHanlde] : enumerate(softBodies)) {
        auto &verts = sbHanlde.verts();
        temp.resize(count_warps(verts.size())); 
        temp.reset(0); 
        pol(range(verts.size()), 
            [vData = view<space>(vData), 
             vOffset = sbHanlde.voffset(),
             temp = view<space>(temp), n = verts.size(), dt = dt] __device__ (int i) mutable {
                auto vi = i + vOffset; 
                auto force = vData.pack(VProps::contact, vi) / (dt * dt); 
                reduce_to(i, n, force.l2NormSqr(), temp[i / 32]);
            }); 
        auto forceNorm = sqrt(reduce(pol, temp)); 
        sbHanlde.bodyTypeProperties().contactForce = forceNorm;
        // prims[bi].zsprimPtr->setMeta(s_ABDextForceTag, vec12::zeros()); // TODO: why softbody has ABD external force?
    }

    for (auto &rbHandle : rigidBodies) {
        rbHandle.updateMesh();
    }
    for (auto &sbHandle : softBodies) {
        sbHandle.updateMesh();
    }

    bodyUpToDate = true;
}

} // namespace tacipc
