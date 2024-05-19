#include <tacipc/solver/Solver.cuh>
#include <zensim/geometry/Distance.hpp>
#include <zensim/geometry/Friction.hpp>
#include <zensim/geometry/SpatialQuery.hpp>
// DCD, CCD

namespace tacipc
{
ABDSolver::T ABDSolver::groundCCD(pol_t &pol, T stepSize)
{
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    zs::Vector<T> alpha{vData.get_allocator(), 1};
    alpha.setVal(stepSize);
    pol(Collapse{vData.size()},
        [vData = view<space>(vData),
         // boundary
         gn = groundNormal, alpha = view<space>(alpha),
         stepSize] ZS_LAMBDA(int vi) mutable
        {
            // skip kinematic verts
            int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi));
            if (isBC)
                return;
            // this vert affected by sticky boundary conditions
            auto dir = vData.pack(VProps::dir, vi);
            auto coef = gn.dot(dir);
            if (coef < 0)
            { // impacting direction
                auto x = vData.pack(VProps::xn, vi);
                auto dist = gn.dot(x);
                auto maxAlpha = (dist * 0.8) / (-coef);
                if (maxAlpha < stepSize)
                {
                    atomic_min(exec_cuda, &alpha[0], maxAlpha);
                }
            }
        });
    stepSize = alpha.getVal();
    fmt::print(fg(fmt::color::dark_cyan), "ground alpha: {}\n", stepSize);
    return stepSize;
}

ABDSolver::T ABDSolver::CCD(pol_t &pol, T eta, T stepSize)
{
    return ACCD(pol, T(0), stepSize);
}

ABDSolver::T ABDSolver::ACCD(pol_t &pol, T eta, T stepSize)
{
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    Vector<T> alpha{vData.get_allocator(), 1};
    alpha.setVal(stepSize);
    auto npt = csPT.getCount();
    pol(range(npt),
        [csPT = csPT.port(), vData = view<space>(vData),
         alpha = view<space>(alpha), stepSize, eta] __device__(int pti)
        {
            auto ids = csPT[pti];
            auto p = vData.pack(VProps::xn, ids[0]);
            auto t0 = vData.pack(VProps::xn, ids[1]);
            auto t1 = vData.pack(VProps::xn, ids[2]);
            auto t2 = vData.pack(VProps::xn, ids[3]);
            auto dp = vData.pack(VProps::dir, ids[0]);
            auto dt0 = vData.pack(VProps::dir, ids[1]);
            auto dt1 = vData.pack(VProps::dir, ids[2]);
            auto dt2 = vData.pack(VProps::dir, ids[3]);
            T tmp = alpha[0];
#if 1
            if (accd::ptccd(p, t0, t1, t2, dp, dt0, dt1, dt2, (T)0.2, eta, tmp))
#elif 1
            if (ticcd::ptccd(p, t0, t1, t2, dp, dt0, dt1, dt2, (T)0.2, eta, tmp))
#else
            if (pt_ccd(p, t0, t1, t2, dp, dt0, dt1, dt2, eta, tmp)) // TODO
#endif
                atomic_min(exec_cuda, &alpha[0], tmp);

            // printf("ptccd: %d, %d, %d, %d, alpha: %f\n", ids[0], ids[1],
            // ids[2], ids[3], alpha[0]);
        });
    if (!enableContactEE)
    {
        stepSize = alpha.getVal();
        return stepSize;
    }
    auto nee = csEE.getCount();
    pol(range(nee),
        [csEE = csEE.port(), vData = view<space>(vData),
         alpha = view<space>(alpha), stepSize, eta] __device__(int eei)
        {
            auto ids = csEE[eei];
            auto ea0 = vData.pack(VProps::xn, ids[0]);
            auto ea1 = vData.pack(VProps::xn, ids[1]);
            auto eb0 = vData.pack(VProps::xn, ids[2]);
            auto eb1 = vData.pack(VProps::xn, ids[3]);
            auto dea0 = vData.pack(VProps::dir, ids[0]);
            auto dea1 = vData.pack(VProps::dir, ids[1]);
            auto deb0 = vData.pack(VProps::dir, ids[2]);
            auto deb1 = vData.pack(VProps::dir, ids[3]);
            auto tmp = alpha[0];
#if 1
            if (accd::eeccd(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, (T)0.2,
                            eta, tmp))
#elif 1
            if (ticcd::eeccd(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, (T)0.2, eta, tmp))
#else
            if (ee_ccd(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, eta, tmp)) // TODO
#endif
                atomic_min(exec_cuda, &alpha[0], tmp);
        });
    stepSize = alpha.getVal();
    return stepSize;
}

ABDSolver::T ABDSolver::inversionPreventCCD(pol_t &pol, T stepSize)
{
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    Vector<T> alpha{vData.get_allocator(), 1};
    alpha.setVal(stepSize);

    auto ptccd = [vData = view<space>(vData),
                  alpha = view<space>(alpha)] __device__ (const pair4_t &ids) mutable {
                    auto p = vData.pack(VProps::xn, ids[0]);
                    auto t0 = vData.pack(VProps::xn, ids[1]);
                    auto t1 = vData.pack(VProps::xn, ids[2]);
                    auto t2 = vData.pack(VProps::xn, ids[3]);
                    auto dp = vData.pack(VProps::dir, ids[0]);
                    auto dt0 = vData.pack(VProps::dir, ids[1]);
                    auto dt1 = vData.pack(VProps::dir, ids[2]);
                    auto dt2 = vData.pack(VProps::dir, ids[3]);
                    T tmp = alpha[0];
#if 1
                    if (accd::ptccd(p, t0, t1, t2, dp, dt0, dt1, dt2, (T)0.2, (T)0, tmp))
#elif 1
                    if (ticcd::ptccd(p, t0, t1, t2, dp, dt0, dt1, dt2, (T)0.2, (T)0, tmp))
#else
                    if (pt_ccd(p, t0, t1, t2, dp, dt0, dt1, dt2, (T)0, tmp)) // TODO
#endif
                        atomic_min(exec_cuda, &alpha[0], tmp);
                };
    auto eeccd = [vData = view<space>(vData),
                  alpha = view<space>(alpha)] __device__ (const pair4_t &ids) mutable {
                    auto p = vData.pack(VProps::xn, ids[0]);
                    auto t0 = vData.pack(VProps::xn, ids[1]);
                    auto t1 = vData.pack(VProps::xn, ids[2]);
                    auto t2 = vData.pack(VProps::xn, ids[3]);
                    auto dp = vData.pack(VProps::dir, ids[0]);
                    auto dt0 = vData.pack(VProps::dir, ids[1]);
                    auto dt1 = vData.pack(VProps::dir, ids[2]);
                    auto dt2 = vData.pack(VProps::dir, ids[3]);
                    T tmp = alpha[0];
#if 1
                    if (accd::ptccd(p, t0, t1, t2, dp, dt0, dt1, dt2, (T)0.2, (T)0, tmp))
#elif 1
                    if (ticcd::ptccd(p, t0, t1, t2, dp, dt0, dt1, dt2, (T)0.2, (T)0, tmp))
#else
                    if (pt_ccd(p, t0, t1, t2, dp, dt0, dt1, dt2, (T)0, tmp)) // TODO
#endif
                        atomic_min(exec_cuda, &alpha[0], tmp);
                };

    for (auto &sbHandle : softBodies)
    {
        if (sbHandle.codim() != 3) // only for tets for now
            continue;

        auto &tets = sbHandle.elems();
        pol(range(tets.size()),
            [tets = view<space>({}, tets),
             indsTag = tets.getPropertyOffset("inds"),
             ptccd, eeccd,
             vOffset = sbHandle.voffset()]__device__(int ti)mutable
            {
                auto inds = tets.pack<4>(indsTag, ti).template reinterpret_bits<Ti>() + vOffset;
                ptccd(pair4_t{inds[0], inds[1], inds[2], inds[3]});
                ptccd(pair4_t{inds[1], inds[2], inds[3], inds[0]});
                ptccd(pair4_t{inds[2], inds[3], inds[0], inds[1]});
                ptccd(pair4_t{inds[3], inds[0], inds[1], inds[2]});

                eeccd(pair4_t{inds[0], inds[1], inds[2], inds[3]});
                eeccd(pair4_t{inds[0], inds[2], inds[1], inds[3]});
                eeccd(pair4_t{inds[0], inds[3], inds[1], inds[2]});
            });
    }
   
    stepSize = alpha.getVal();
    return stepSize;
}

void ABDSolver::findBarrierCollisions(pol_t &pol, T xi)
{
    PP.reset();
    PE.reset();
    PT.reset();
    EE.reset();

    PPM.reset();
    PEM.reset();
    EEM.reset();

    csPT.reset();
    csEE.reset();

    zs::CppTimer timer;
    timer.tick();
    {
        if (pureRigidScene)
        {
            {
                using namespace zs;
                constexpr auto space = execspace_e::cuda;
                // update body boxes
                // rbRestBvs -- affine transform -> rbBvs
                pol(range(rbData.size()),
                    [rbData = view<space>(rbData),
                     rbRestBvs = view<space>(rbRestBvs),
                     rbBvs = view<space>(rbBvs)] 
                     __device__ (int rbi) mutable {
                        zs::vec<T, 2, 3> corners;
                        auto &minPos = rbRestBvs[rbi]._min;
                        auto &maxPos = rbRestBvs[rbi]._max;
                        vec3 newMinPos{T_max_c, T_max_c, T_max_c};
                        vec3 newMaxPos{T_min_c, T_min_c, T_min_c};
                        for (int d = 0; d < 3; d++)
                        {
                            corners(0, d) = minPos(d);
                            corners(1, d) = maxPos(d);
                        }
                        auto center = rbData.pack(RBProps::center, rbi);
                        auto qn = rbData.pack(RBProps::qn, rbi);
                        for (int s0 = 0; s0 < 2; s0++)
                            for (int s1 = 0; s1 < 2; s1++)
                                for (int s2 = 0; s2 < 2; s2++)
                                {
                                    vec3 pos{corners(s0, 0), corners(s1, 1), corners(s2, 2)}; 
                                    auto new_pos = ABD_q2x(pos - center, qn); 
                                    for (int d = 0; d < 3; d++)
                                    {
                                        newMaxPos[d] = zs::max(newMaxPos[d],
                                        new_pos[d]); newMinPos[d] =
                                        zs::min(newMinPos[d], new_pos[d]);
                                    }
                                }
                        rbBvs(rbi) = bv_t{newMinPos, newMaxPos};
                     });
            }
            pure_rigid_retrieve_bounding_volumes(pol, vData, rbData, dHat, true, stInds, 0, rbBvs, culledStInds, bvs); 
            rigidStBvhSize = bvs.size(); 
            rigidStBvh.build(pol, bvs); 
            if (enableContactEE)
            {
                pure_rigid_retrieve_bounding_volumes(pol, vData, rbData, dHat, true, seInds, 0, rbBvs, culledSeInds, bvs); 
                rigidSeBvhSize = bvs.size(); 
                rigidSeBvh.build(pol, bvs);
            }
            zs::CppTimer timer;
            timer.tick();
            findBarrierCollisionsImpl(pol, xi, false);
            timer.tock("find constrains impl");
        } else
        {
            retrieve_bounding_volumes(pol, vData, stInds, RigidBodyHandle::stNum, bvs, bvs1);
            rigidStBvhSize = bvs.size();
            softStBvhSize = bvs1.size();
            rigidStBvh.refit(pol, bvs);
            softStBvh.refit(pol, bvs1);
            fmt::print("[DCD] rigidStBvhSize: {}, softStBvhSize: {}\n", rigidStBvhSize, softStBvhSize);
            if (enableContactEE)
            {
                retrieve_bounding_volumes(pol, vData, seInds, RigidBodyHandle::seNum, bvs, bvs1);
                rigidSeBvhSize = bvs.size();
                softSeBvhSize = bvs1.size();
                rigidSeBvh.refit(pol, bvs);
                softSeBvh.refit(pol, bvs1);
                fmt::print("[DCD] rigidSeBvhSize: {}, softSeBvhSize: {}\n", rigidSeBvhSize, softSeBvhSize);
            }
            findBarrierCollisionsImpl(pol, xi, false);
        }
    }
    auto [npt, nee] = getCollisionCnts();
    timer.tock(fmt::format("dcd broad phase [pt, ee]({}, {})", npt, nee));
}

void ABDSolver::findBarrierCollisionsImpl(pol_t &pol, T xi, bool withBoundary)
{
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    const auto dHat2 = dHat * dHat;

    /// pt
    zs::CppTimer timer;
    timer.tick();
    auto findPTCollisions = [&](bvh_t &bvh, int stOffset = 0)
    {
        return
            [svData = view<space>(svData, false_c, "svData"), eles = view<space>(stInds, false_c, "stInds"),
             // eles = view<space>({}, withBoundary ? coEles : stInds),
             vData = view<space>(vData, false_c, "vData"), rbData = view<space>(rbData, false_c, "rbData"),
             // disKinCol = s_disableKinCollision,  // TODO: kinematic
             // collision?
             collisionMat = view<space>(collisionMat),
             bvh = view<space>(bvh), PP = PP.port(), PE = PE.port(),
             PT = PT.port(), csPT = csPT.port(), dHat = this->dHat, xi,
             thickness = xi + dHat, stOffset = stOffset,
             culledStInds = view<space>(culledStInds), 
             pureRigidScene = pureRigidScene
             ] __device__(int svi) mutable
        {
            int vi = reinterpret_bits<Ti>(svData(SVProps::inds, svi));
            const auto dHat2 = zs::sqr(dHat + xi);
            auto p = vData.pack(VProps::xn, vi);
            auto bv = bv_t{get_bounding_box(p - thickness, p + thickness)};
            bvh.iter_neighbors(
                bv,
                [&](int stI)
                {
                    stI += stOffset;
                    if (pureRigidScene) 
                        stI = culledStInds[stI];
                    auto tri = eles.pack(TIProps::inds, stI)
                                   .template reinterpret_bits<Ti>();
                    if (vi == tri[0] || vi == tri[1] || vi == tri[2])
                        return;
                    int layerP = vData(VProps::layer, vi, ti_c);
                    int layerT = vData(VProps::layer, tri[0], ti_c);
                    if (!collisionMat(layerP, layerT))
                        return;
                    int bodyP = reinterpret_bits<Ti>(vData(VProps::body, vi));
                    int bodyT = reinterpret_bits<Ti>(vData(VProps::body, tri[0]));
                    if (bodyP == bodyT && bodyP >= 0)
                        return;
                    int groupP = vData(VProps::exclGrpIdx, vi, ti_c);
                    int groupT = vData(VProps::exclGrpIdx, tri[0], ti_c);
                    if (groupP == groupT && groupP >= 0)
                        return;
                    // if (disKinCol) // TODO
                    // {
                    //     int isPKin = bodyP >= 0 ?
                    //     reinterpret_bits<Ti>(rbData(rbIsBCTag, bodyP)) :
                    //         reinterpret_bits<Ti>(vData(sbIsBCTag, vi));
                    //     int isTKin = bodyT >= 0 ?
                    //     reinterpret_bits<Ti>(rbData(rbIsBCTag, bodyT)) :
                    //         reinterpret_bits<Ti>(vData(sbIsBCTag, tri[0])) +
                    //         reinterpret_bits<Ti>(vData(sbIsBCTag, tri[1])) +
                    //         reinterpret_bits<Ti>(vData(sbIsBCTag, tri[2]));
                    //     if (isPKin && isTKin)
                    //         return;
                    // }
                    //                    printf("find collision constraints PT:
                    //                    %d and %d\n", bodyP, bodyT);
                    // ccd
                    auto t0 = vData.pack(VProps::xn, tri[0]);
                    auto t1 = vData.pack(VProps::xn, tri[1]);
                    auto t2 = vData.pack(VProps::xn, tri[2]);

                    switch (pt_distance_type(p, t0, t1, t2))
                    {
                    case 0:
                    {
                        if (auto d2 = dist2_pp(p, t0); d2 < dHat2)
                        {
                            PP.try_push(pair_t{tri[0], vi});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 1:
                    {
                        if (auto d2 = dist2_pp(p, t1); d2 < dHat2)
                        {
                            PP.try_push(pair_t{tri[1], vi});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 2:
                    {
                        if (auto d2 = dist2_pp(p, t2); d2 < dHat2)
                        {
                            PP.try_push(pair_t{tri[2], vi});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 3:
                    {
                        if (auto d2 = dist2_pe(p, t0, t1); d2 < dHat2)
                        {
                            PE.try_push(pair3_t{vi, tri[0], tri[1]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 4:
                    {
                        if (auto d2 = dist2_pe(p, t1, t2); d2 < dHat2)
                        {
                            PE.try_push(pair3_t{vi, tri[1], tri[2]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 5:
                    {
                        if (auto d2 = dist2_pe(p, t2, t0); d2 < dHat2)
                        {
                            PE.try_push(pair3_t{vi, tri[2], tri[0]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    case 6:
                    {
                        if (auto d2 = dist2_pt(p, t0, t1, t2); d2 < dHat2)
                        {
                            PT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                            csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                        }
                        break;
                    }
                    default:
                        break;
                    }
                });
        };
    };
    if (withBoundary)
    {
        pol(Collapse{svData.size()}, findPTCollisions(bouStBvh));
    }
    else
    {
        if (rigidStBvhSize)
            pol(Collapse{svData.size()}, findPTCollisions(rigidStBvh));
        if (softStBvhSize)
            pol(Collapse{svData.size()}, findPTCollisions(softStBvh, RigidBodyHandle::stNum));
    }
    timer.tock("find barrier collision impl pt");

    if (!enableContactEE)
        return;
    /// ee
    fmt::print("seInds.size(): {}\n", seInds.size());
    timer.tick();
    auto findEECollisions = [&](bvh_t &bvh, int seOffset = 0)
    {
        return
            [seInds = view<space>(seInds, false_c, "seInds"),
             sedges = view<space>(seInds, false_c, "sedges"),
             //                                          sedges = view<space>(
             //                                              {}, withBoundary ?
             //                                              coEdges : seInds),
             vData = view<space>(vData, false_c, "vData"), rbData = view<space>(rbData, false_c, "rbData"),
             // disKinCol = s_disableKinCollision, // TODO
             collisionMat = view<space>(collisionMat),
             bvh = view<space>(bvh),
             PP = PP.port(), PE = PE.port(), EE = EE.port(),
#if s_enableMollification
             // mollifier
             PPM = PPM.port(), PEM = PEM.port(), EEM = EEM.port(),
#endif
             //
             csEE = csEE.port(), dHat = this->dHat, xi, thickness = xi + dHat,
             enableMollification = this->enableMollification, seOffset = seOffset,
             culledSeInds = view<space>(culledSeInds), 
             pureRigidScene = pureRigidScene
        ] __device__(int sei) mutable
        {
            if (pureRigidScene) 
                sei = culledSeInds[sei];
            const auto dHat2 = zs::sqr(dHat + xi);
            auto eiInds = seInds.pack(EIProps::inds, sei)
                              .template reinterpret_bits<Ti>();
            // printf("[dcd] testing ee for (%d-%d), thickness: %f, dHat: %f, dHat2 * 1000: %f\n", (int)eiInds[0], (int)eiInds[1], (float)thickness, (float)dHat, (float)dHat2 * 1000);
            auto v0 = vData.pack(VProps::xn, eiInds[0]);
            auto v1 = vData.pack(VProps::xn, eiInds[1]);
            auto rv0 = vData.pack(VProps::x0, eiInds[0]);
            auto rv1 = vData.pack(VProps::x0, eiInds[1]);
            auto [mi, ma] = get_bounding_box(v0, v1);
            auto bv = bv_t{mi - thickness, ma + thickness};
            bvh.iter_neighbors(
                bv,
                [&](int sej)
                {
                    sej += seOffset;
                    if (pureRigidScene) 
                        sej = culledSeInds[sej];
                    if (sei > sej)
                        return;
                    
                    auto ejInds = sedges.pack(EIProps::inds, sej)
                                      .template reinterpret_bits<Ti>();
                    // if (seOffset > 0)
                    //     printf("checking EE (%d, %d-%d, %d)\n", (int)eiInds[0], (int)eiInds[1], (int)ejInds[0], (int)ejInds[1]);
                    if (eiInds[0] == ejInds[0] || eiInds[0] == ejInds[1] ||
                        eiInds[1] == ejInds[0] || eiInds[1] == ejInds[1])
                        return;
                    int layerE1 = vData(VProps::layer, eiInds[0], ti_c);
                    int layerE2 = vData(VProps::layer, ejInds[0], ti_c);
                    if (!collisionMat(layerE1, layerE2))
                        return;
                    int bodyE1 = reinterpret_bits<Ti>(vData(VProps::body, eiInds[0]));
                    int bodyE2 = reinterpret_bits<Ti>(vData(VProps::body, ejInds[0]));
                    if (bodyE1 == bodyE2 && bodyE1 >= 0)
                        return;
                    int groupE1 = vData(VProps::exclGrpIdx, eiInds[0], ti_c);
                    int groupE2 = vData(VProps::exclGrpIdx, ejInds[0], ti_c);
                    if (groupE1 == groupE2 && groupE1 >= 0)
                        return;
                    // if (disKinCol) // TODO
                    // {
                    //     int isE1Kin = bodyE1 >= 0 ?
                    //     reinterpret_bits<Ti>(rbData("isBC", bodyE1)) :
                    //         reinterpret_bits<Ti>(vData("isBC", eiInds[0])) +
                    //         reinterpret_bits<Ti>(vData("isBC", eiInds[1]));
                    //     int isE2Kin = bodyE2 >= 0 ?
                    //     reinterpret_bits<Ti>(rbData("isBC", bodyE2)) :
                    //         reinterpret_bits<Ti>(vData("isBC", ejInds[0])) +
                    //         reinterpret_bits<Ti>(vData("isBC", ejInds[1]));
                    //     if (isE1Kin && isE2Kin)
                    //         return;
                    // }
                    // ccd
                    auto v2 = vData.pack(VProps::xn, ejInds[0]);
                    auto v3 = vData.pack(VProps::xn, ejInds[1]);
                    auto rv2 = vData.pack(VProps::x0, ejInds[0]);
                    auto rv3 = vData.pack(VProps::x0, ejInds[1]);

#if s_enableMollification
                    // IPC (24)
                    T c = cn2_ee(v0, v1, v2, v3);
                    T epsX = mollifier_threshold_ee(rv0, rv1, rv2, rv3);
                    bool mollify = c < epsX;
#endif

                    switch (ee_distance_type(v0, v1, v2, v3))
                    {
                    case 0:
                    {
                        // printf("ee (%d-%d,%d-%d) dist2 * 1000: %f, dhat2 * 1000: %f\n", (int)eiInds[0], (int)eiInds[1], (int)ejInds[0], (int)ejInds[1], (float)dist2_pp(v0, v2) * 1000, (float)dHat2 * 1000);
                        if (auto d2 = dist2_pp(v0, v2);  d2 < dHat2)
                        {
                            {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1],
                                                      ejInds[0], ejInds[1]});
#if s_enableMollification
                                if (enableMollification)
                                {
                                    if (mollify)
                                    {
                                        PPM.try_push(
                                            pair4_t{eiInds[0], eiInds[1],
                                                    ejInds[0], ejInds[1]});
                                        break;
                                    }
                                }
#endif
                                {
                                    PP.try_push(pair_t{eiInds[0], ejInds[0]});
                                }
                            }
                        }
                        break;
                    }
                    case 1:
                    {
                        // printf("ee (%d-%d,%d-%d) dist2 * 1000: %f, dhat2 * 1000: %f\n", (int)eiInds[0], (int)eiInds[1], (int)ejInds[0], (int)ejInds[1], (float)dist2_pp(v0, v3) * 1000, (float)dHat2 * 1000);
                        if (auto d2 = dist2_pp(v0, v3); d2 < dHat2)
                        {
                            {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1],
                                                      ejInds[0], ejInds[1]});
#if s_enableMollification
                                if (enableMollification)
                                {
                                    if (mollify)
                                    { // TODO: ? why not aligned?
                                        PPM.try_push(
                                            pair4_t{eiInds[0], eiInds[1],
                                                    ejInds[1], ejInds[0]});
                                        break;
                                    }
                                }
#endif
                            }
                            {
                                PP.try_push(pair_t{eiInds[0], ejInds[1]});
                            }
                        }
                        break;
                    }
                    case 2:
                    {
                        // printf("ee (%d-%d,%d-%d) dist2 * 1000: %f, dhat2 * 1000: %f\n", (int)eiInds[0], (int)eiInds[1], (int)ejInds[0], (int)ejInds[1], (float)dist2_pe(v0, v2, v3) * 1000, (float)dHat2 * 1000);
                        if (auto d2 = dist2_pe(v0, v2, v3); d2 < dHat2)
                        {
                            {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1],
                                                      ejInds[0], ejInds[1]});
#if s_enableMollification
                                if (enableMollification)
                                {
                                    if (mollify)
                                    {
                                        PEM.try_push(
                                            pair4_t{eiInds[0], eiInds[1],
                                                    ejInds[0], ejInds[1]});
                                        break;
                                    }
                                }
#endif
                                {
                                    PE.try_push(pair3_t{eiInds[0], ejInds[0],
                                                        ejInds[1]});
                                }
                            }
                        }
                        break;
                    }
                    case 3:
                    {
                        // printf("ee (%d-%d,%d-%d) dist2 * 1000: %f, dhat2 * 1000: %f\n", (int)eiInds[0], (int)eiInds[1], (int)ejInds[0], (int)ejInds[1], (float)dist2_pp(v1, v2) * 1000, (float)dHat2 * 1000);
                        if (auto d2 = dist2_pp(v1, v2); d2 < dHat2)
                        {
                            {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1],
                                                      ejInds[0], ejInds[1]});
#if s_enableMollification
                                if (enableMollification)
                                {
                                    if (mollify)
                                    {
                                        PPM.try_push(
                                            pair4_t{eiInds[1], eiInds[0],
                                                    ejInds[0], ejInds[1]});
                                        break;
                                    }
                                }
#endif
                                {
                                    PP.try_push(pair_t{eiInds[1], ejInds[0]});
                                }
                            }
                        }
                        break;
                    }
                    case 4:
                    {
                        // printf("ee (%d-%d,%d-%d) dist2 * 1000: %f, dhat2 * 1000: %f\n", (int)eiInds[0], (int)eiInds[1], (int)ejInds[0], (int)ejInds[1], (float)dist2_pp(v1, v3) * 1000, (float)dHat2 * 1000);
                        if (auto d2 = dist2_pp(v1, v3); d2 < dHat2)
                        {
                            {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1],
                                                      ejInds[0], ejInds[1]});
#if s_enableMollification
                                if (enableMollification)
                                {
                                    if (mollify)
                                    {
                                        PPM.try_push(
                                            pair4_t{eiInds[1], eiInds[0],
                                                    ejInds[1], ejInds[0]});
                                        break;
                                    }
                                }
#endif
                            }
                            {
                                PP.try_push(pair_t{eiInds[1], ejInds[1]});
                            }
                        }
                        break;
                    }
                    case 5:
                    {
                        // printf("ee (%d-%d,%d-%d) dist2 * 1000: %f, dhat2 * 1000: %f\n", (int)eiInds[0], (int)eiInds[1], (int)ejInds[0], (int)ejInds[1], (float)dist2_pe(v1, v2, v3) * 1000, (float)dHat2 * 1000);
                        if (auto d2 = dist2_pe(v1, v2, v3); d2 < dHat2)
                        {
                            {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1],
                                                      ejInds[0], ejInds[1]});
#if s_enableMollification
                                if (enableMollification)
                                {
                                    if (mollify)
                                    {
                                        PEM.try_push(
                                            pair4_t{eiInds[1], eiInds[0],
                                                    ejInds[0], ejInds[1]});
                                        break;
                                    }
                                }
#endif
                                {
                                    PE.try_push(pair3_t{eiInds[1], ejInds[0],
                                                        ejInds[1]});
                                }
                            }
                        }
                        break;
                    }
                    case 6:
                    {
                        // printf("ee (%d-%d,%d-%d) dist2 * 1000: %f, dhat2 * 1000: %f\n", (int)eiInds[0], (int)eiInds[1], (int)ejInds[0], (int)ejInds[1], (float)dist2_ee(v0, v1, v2, v3) * 1000, (float)dHat2 * 1000);
                        if (auto d2 = dist2_pe(v2, v0, v1); d2 < dHat2)
                        {
                            {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1],
                                                      ejInds[0], ejInds[1]});
#if s_enableMollification
                                if (enableMollification)
                                {
                                    if (mollify)
                                    {
                                        PEM.try_push(
                                            pair4_t{ejInds[0], ejInds[1],
                                                    eiInds[0], eiInds[1]});
                                        break;
                                    }
                                }
#endif
                                {
                                    PE.try_push(pair3_t{ejInds[0], eiInds[0],
                                                        eiInds[1]});
                                }
                            }
                        }
                        break;
                    }
                    case 7:
                    {
                        // printf("ee (%d-%d,%d-%d) dist2 * 1000: %f, dhat2 * 1000: %f\n", (int)eiInds[0], (int)eiInds[1], (int)ejInds[0], (int)ejInds[1], (float)dist2_ee(v0, v1, v2, v3) * 1000, (float)dHat2 * 1000);
                        if (auto d2 = dist2_pe(v3, v0, v1); d2 < dHat2)
                        {
                            {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1],
                                                      ejInds[0], ejInds[1]});
#if s_enableMollification
                                if (enableMollification)
                                {
                                    if (mollify)
                                    {
                                        PEM.try_push(
                                            pair4_t{ejInds[1], ejInds[0],
                                                    eiInds[0], eiInds[1]});
                                        break;
                                    }
                                }
#endif
                                {
                                    PE.try_push(pair3_t{ejInds[1], eiInds[0],
                                                        eiInds[1]});
                                }
                            }
                        }
                        break;
                    }
                    case 8:
                    {
                        // printf("ee (%d-%d,%d-%d) dist2 * 1000: %f, dhat2 * 1000: %f\n", (int)eiInds[0], (int)eiInds[1], (int)ejInds[0], (int)ejInds[1], (float)dist2_ee(v0, v1, v2, v3) * 1000, (float)dHat2 * 1000);
                        if (auto d2 = dist2_ee(v0, v1, v2, v3); d2 < dHat2)
                        {
                            {
                                csEE.try_push(pair4_t{eiInds[0], eiInds[1],
                                                      ejInds[0], ejInds[1]});
#if s_enableMollification
                                if (enableMollification)
                                {
                                    if (mollify)
                                    {
                                        EEM.try_push(
                                            pair4_t{eiInds[0], eiInds[1],
                                                    ejInds[0], ejInds[1]});
                                        break;
                                    }
                                }
#endif
                                {
                                    EE.try_push(pair4_t{eiInds[0], eiInds[1],
                                                        ejInds[0], ejInds[1]});
                                }
                            }
                        }
                        break;
                    }
                    default:
                        break;
                    }
                });
        };
    };
    if (pureRigidScene)
    {
        pol(Collapse{culledSeInds.size()}, findEECollisions(rigidSeBvh));
    }
    else 
    {
        if (rigidSeBvhSize && softSeBvhSize)
        {
            pol(Collapse{seInds.size()},
                findEECollisions(softSeBvh, RigidBodyHandle::seNum));
            pol(Collapse{RigidBodyHandle::seNum}, findEECollisions(rigidSeBvh));
        }
        else if (rigidSeBvhSize)
            pol(Collapse{seInds.size()}, findEECollisions(rigidSeBvh));
        else if (softSeBvhSize)
            pol(Collapse{seInds.size()}, findEECollisions(softSeBvh));
    }
    // if (rigidSeBvhSize && softSeBvhSize)
    //     pol(Collapse{softSeBvh.getNumLeaves()},
    //         findEECollisions(&softSeBvh, &rigidSeBvh));
    timer.tock("find barrier collision impl ee");
}

void ABDSolver::findCCDCollisions(pol_t &pol, T alpha, T xi)
{
    csPT.reset();
    csEE.reset();
    {
        if (pureRigidScene)
        {
            {
                using namespace zs;
                constexpr auto space = execspace_e::cuda;
                // update body boxes
                // rbRestBvs -- affine transform -> rbBvs
                pol(range(rbData.size()),
                    [rbData = view<space>(rbData),
                     rbRestBvs = view<space>(rbRestBvs),
                     rbBvs = view<space>(rbBvs),
                     alpha] __device__ (int bi) mutable {
                        zs::vec<T, 2, 3> corners;
                        auto &minPos = rbRestBvs[bi]._min;
                        auto &maxPos = rbRestBvs[bi]._max;
                        bv_t bv{vec3::constant(T_max_c),
                        vec3::constant(T_min_c)}; for (int d = 0; d < 3; d++)
                        {
                            corners(0, d) = minPos(d);
                            corners(1, d) = maxPos(d);
                        }
                        auto center = rbData.pack(RBProps::center, bi);
                        auto qn = rbData.pack(RBProps::qn, bi);
                        auto qDir = rbData.pack(RBProps::qDir, bi);
                        for (int s0 = 0; s0 < 2; s0++)
                            for (int s1 = 0; s1 < 2; s1++)
                                for (int s2 = 0; s2 < 2; s2++)
                                {
                                    vec3 pos{corners(s0, 0), corners(s1, 1),
                                    corners(s2, 2)}; 
                                    merge(bv, ABD_q2x(pos - center, qn)); 
                                    merge(bv, ABD_q2x(pos - center, qn + alpha * qDir));
                                }
                        rbBvs(bi) = bv;
                     });
            }
            zs::CppTimer timer;
            timer.tick();
            pure_rigid_retrieve_bounding_volumes(pol, vData, rbData, dHat, alpha, stInds, 0, rbBvs, culledStInds, bvs); 
            rigidStBvhSize = bvs.size(); 
            rigidStBvh.build(pol, bvs); 
            fmt::print("[pure_rigid_retrieve_bounding_volumes] bvs st size: {}\n", bvs.size()); 
            pure_rigid_retrieve_bounding_volumes(pol, vData, rbData, dHat, alpha, seInds, 0, rbBvs, culledSeInds, bvs);
            fmt::print("[pure_rigid_retrieve_bounding_volumes] bvs se size: {}\n", bvs.size()); 
            rigidSeBvhSize = bvs.size(); 
            rigidSeBvh.build(pol, bvs); 
            timer.tock("ccd affine bvh build");
        } else
        {
            retrieve_bounding_volumes(pol, vData, stInds,
                                      vData, alpha,
                                      RigidBodyHandle::stNum, bvs, bvs1);
            rigidStBvhSize = bvs.size();
            softStBvhSize = bvs1.size();
            rigidStBvh.refit(pol, bvs);
            softStBvh.refit(pol, bvs1);
            retrieve_bounding_volumes(pol, vData, seInds,
                                      vData, alpha,
                                      RigidBodyHandle::seNum, bvs, bvs1);
            rigidSeBvhSize = bvs.size();
            softSeBvhSize = bvs1.size();
            rigidSeBvh.refit(pol, bvs);
            softSeBvh.refit(pol, bvs1);
            fmt::print("[CCD] rigidStBvhSize: {}, softStBvhSize: {}, rigidSeBvhSize: {}, softSeBvhSize: {}\n", rigidStBvhSize, softStBvhSize, rigidSeBvhSize, softSeBvhSize);
        }
    }

    zs::CppTimer timer;
    timer.tick();
    findCCDCollisionsImpl(pol, alpha, xi, false);
    // if (enableInversionPrevention)
    //     findInversionCCDCollisionsImpl(pol, alpha, xi);

    auto checkSize = [this](const auto &cnt, std::string_view msg)
    {
        if (cnt >= estNumCps)
            throw std::runtime_error(
                fmt::format("[{}] cp queue of size {} not enough for {} cps!",
                            msg, estNumCps, cnt));
    };
    checkSize(csPT.getCount(), "PT");
    checkSize(csEE.getCount(), "EE");
    auto [npt, nee] = getCollisionCnts();
    timer.tock(fmt::format("ccd broad phase [pt, ee]({}, {})", npt, nee));
}

void ABDSolver::findCCDCollisionsImpl(pol_t &pol, T alpha, T xi, bool withBoundary)
{
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    const auto dHat2 = dHat * dHat;
    zs::CppTimer timer;
    /// pt
    timer.tick();
    auto findPTCollisions = [&](bvh_t &bvh, int stOffset = 0)
    {
        auto thickness = xi + dHat;
        // auto thickness = xi;
        return
            [svData = view<space>(svData), eles = view<space>(stInds),
             // eles = view<space>({}, withBoundary ? coEles : stInds),
             vData = view<space>(vData), rbData = view<space>(rbData),
             collisionMat = view<space>(collisionMat),
             bvh = view<space>(bvh), PP = PP.port(), PE = PE.port(),
             PT = PT.port(), csPT = csPT.port(), thickness = thickness, alpha, stOffset = stOffset,
             culledStInds = view<space>(culledStInds),
             pureRigidScene = pureRigidScene
             // disKinCol = s_disableKinCollision, // TODO
             ] __device__(int vi) mutable
        {
            vi = reinterpret_bits<Ti>(svData(SVProps::inds, vi));
            // printf("finding CCD PT for %d\n", (int)vi);
            auto p = vData.pack(VProps::xn, vi);
            auto dir = vData.pack(VProps::dir, vi);
            auto bv = bv_t{get_bounding_box(p, p + alpha * dir)};
            bv._min -= thickness;
            bv._max += thickness;
            bvh.iter_neighbors(
                bv,
                [&](int stI)
                {
                    stI += stOffset;
                    if (pureRigidScene) 
                        stI = culledStInds[stI];
                    auto tri = eles.pack(TIProps::inds, stI)
                                   .template reinterpret_bits<Ti>();
                    int layerP = vData(VProps::layer, vi, ti_c);
                    int layerT = vData(VProps::layer, tri[0], ti_c);
                    if (!collisionMat(layerP, layerT))
                        return;
                    int bodyP = reinterpret_bits<Ti>(vData(VProps::body, vi));
                    int bodyT = reinterpret_bits<Ti>(vData(VProps::body, tri[0]));
                    if (bodyP == bodyT && bodyP >= 0)
                        return;
                    if (vi == tri[0] || vi == tri[1] || vi == tri[2])
                        return;
                    int groupP = vData(VProps::exclGrpIdx, vi, ti_c);
                    int groupT = vData(VProps::exclGrpIdx, tri[0], ti_c);
                    if (groupP == groupT && groupP >= 0)
                        return;
                    // if (disKinCol) // TODO
                    // {
                    //     int isPKin = bodyP >= 0 ?
                    //     reinterpret_bits<Ti>(rbData(rbIsBCTag, bodyP)) :
                    //         reinterpret_bits<Ti>(vData(sbIsBCTag, vi));
                    //     int isTKin = bodyT >= 0 ?
                    //     reinterpret_bits<Ti>(rbData(rbIsBCTag, bodyT)) :
                    //         reinterpret_bits<Ti>(vData(sbIsBCTag, tri[0])) +
                    //         reinterpret_bits<Ti>(vData(sbIsBCTag, tri[1])) +
                    //         reinterpret_bits<Ti>(vData(sbIsBCTag, tri[2]));
                    //     if (isPKin && isTKin)
                    //         return;
                    // }
                    // all affected by sticky boundary conditions
                    // ABD: remove BCorder related codes
                    csPT.try_push(pair4_t{vi, tri[0], tri[1], tri[2]});
                    // printf("find CCD PT: (%d, %d-%d-%d)\n", (int)vi, (int)tri[0], (int)tri[1], (int)tri[2]);
                });
        };
    };
    if (withBoundary)
        pol(Collapse{svData.size()}, findPTCollisions(bouStBvh));
    else
    {
        if (rigidStBvhSize)
            pol(Collapse{svData.size()}, findPTCollisions(rigidStBvh, 0));
        if (softStBvhSize)
            pol(Collapse{svData.size()}, findPTCollisions(softStBvh, RigidBodyHandle::stNum));
    }
    timer.tock("ccd bvh query pt");

    if (!enableContactEE)
        return;
    /// ee
    timer.tick();
    auto findEECollisions = [&](bvh_t &bvh, int seOffset = 0) // warning: edges in bvh should are supposed to cover larger indices
    {
        auto thickness = xi + dHat;
        // auto thickness = xi;
        return
            [seInds = view<space>(seInds),
             sedges = view<space>(seInds),
             // sedges = view<space>({}, withBoundary ? coEdges : seInds),
             vData = view<space>(vData), rbData = view<space>(rbData),
             // disKinCol = s_disableKinCollision, // TODO
             collisionMat = view<space>(collisionMat),
             bvh = view<space>(bvh),
             PP = PP.port(), PE = PE.port(), EE = EE.port(), csEE = csEE.port(),
             thickness = thickness, alpha,
             culledSeInds = view<space>(culledSeInds),
             pureRigidScene = pureRigidScene,
             seOffset = seOffset] __device__(int sei) mutable
        {
            if (pureRigidScene) 
                sei = culledSeInds[sei];
            auto eiInds = seInds.pack(EIProps::inds, sei)
                              .template reinterpret_bits<Ti>();
            // ABD: remove BCorder related codes
            auto v0 = vData.pack(VProps::xn, eiInds[0]);
            auto v1 = vData.pack(VProps::xn, eiInds[1]);
            auto dir0 = vData.pack(VProps::dir, eiInds[0]);
            auto dir1 = vData.pack(VProps::dir, eiInds[1]);
            auto bv = bv_t{get_bounding_box(v0, v0 + alpha * dir0)};
            merge(bv, v1);
            merge(bv, v1 + alpha * dir1);
            bv._min -= thickness;
            bv._max += thickness;
            bvh.iter_neighbors(
                bv,
                [&](int sej)
                {
                    sej += seOffset;
                    if (pureRigidScene)
                        sej = culledSeInds[sej];
                    if (sei > sej)
                        return;
                    auto ejInds = sedges.pack(EIProps::inds, sej)
                                      .template reinterpret_bits<Ti>();
                    int layerE1 = vData(VProps::layer, eiInds[0], ti_c);
                    int layerE2 = vData(VProps::layer, ejInds[0], ti_c);
                    if (!collisionMat(layerE1, layerE2))
                        return;
                    int bodyE1 = reinterpret_bits<Ti>(vData(VProps::body, eiInds[0]));
                    int bodyE2 = reinterpret_bits<Ti>(vData(VProps::body, ejInds[0]));
                    if (bodyE1 == bodyE2 && bodyE1 >= 0)
                        return;
                    if (eiInds[0] == ejInds[0] || eiInds[0] == ejInds[1] ||
                        eiInds[1] == ejInds[0] || eiInds[1] == ejInds[1])
                        return;
                    int groupE1 = vData(VProps::exclGrpIdx, eiInds[0], ti_c);
                    int groupE2 = vData(VProps::exclGrpIdx, ejInds[0], ti_c);
                    if (groupE1 == groupE2 && groupE1 >= 0)
                        return;
                    // if (disKinCol) // TODO
                    // {
                    //     int isE1Kin = bodyE1 >= 0 ?
                    //     reinterpret_bits<Ti>(rbData("isBC", bodyE1)) :
                    //         reinterpret_bits<Ti>(vData("isBC", eiInds[0])) +
                    //         reinterpret_bits<Ti>(vData("isBC", eiInds[1]));
                    //     int isE2Kin = bodyE2 >= 0 ?
                    //     reinterpret_bits<Ti>(rbData("isBC", bodyE2)) :
                    //         reinterpret_bits<Ti>(vData("isBC", ejInds[0])) +
                    //         reinterpret_bits<Ti>(vData("isBC", ejInds[1]));
                    //     if (isE1Kin && isE2Kin)
                    //         return;
                    // }
                    // all affected by sticky boundary conditions
                    // ABD: remove BCorder related codes
                    csEE.try_push(
                        pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]});
                    // printf("find CCD EE: (%d-%d, %d-%d)\n", (int)eiInds[0], (int)eiInds[1], (int)ejInds[0], (int)ejInds[1]);
                });
        };
    };
    if (withBoundary)
    {
        pol(Collapse{bouSeBvh.getNumLeaves()}, findEECollisions(bouSeBvh));
    }
    else
    {
        if (pureRigidScene)
        {
            pol(Collapse{culledSeInds.size()},
                findEECollisions(rigidSeBvh));
        }
        else
        {
            if (rigidSeBvhSize && softSeBvhSize)
            {
                pol(Collapse{seInds.size()},
                    findEECollisions(softSeBvh, RigidBodyHandle::seNum)); // find all-to-soft
                pol(Collapse{RigidBodyHandle::seNum},
                    findEECollisions(rigidSeBvh)); // find rigid-to-rigid
            }
            else if (rigidSeBvhSize)
            {
                pol(Collapse{seInds.size()},
                    findEECollisions(rigidSeBvh));
            }
            else if (softSeBvhSize)
            {
                pol(Collapse{seInds.size()},
                    findEECollisions(softSeBvh));
            }
        }
    }
    timer.tock("ccd bvh query ee");
}

// void ABDSolver::findInversionCCDCollisionsImpl(pol_t &pol, T alpha, T xi)
// {
//     using namespace zs; 
//     constexpr auto space = execspace_e::cuda;

//     CppTimer timer;
//     timer.tick();

//     for (auto &sbHandle : softBodies)
//     {
//         if (sbHandle.codim() != 3) // only for tets for now
//             continue;

//         auto &tets = sbHandle.elems();
//         pol(range(tets.size()),
//             [tets = view<space>({}, tets),
//              csPT = csPT.port(),
//              csEE = csEE.port(),
//              indsTag = tets.getPropertyOffset("inds"),
//              vOffset = sbHandle.voffset()]__device__(int ti)mutable
//             {
//                 auto inds = tets.pack<4>(indsTag, ti).template reinterpret_bits<Ti>() + vOffset;
//                 csPT.try_push(pair4_t{inds[0], 
//                                         inds[1], inds[2], inds[3]});
//                 csPT.try_push(pair4_t{inds[1],
//                                         inds[2], inds[3], inds[0]});
//                 csPT.try_push(pair4_t{inds[2],
//                                         inds[3], inds[0], inds[1]});
//                 csPT.try_push(pair4_t{inds[3],
//                                         inds[0], inds[1], inds[2]});

//                 csEE.try_push(pair4_t{inds[0], inds[1],
//                                         inds[2], inds[3]});
//                 csEE.try_push(pair4_t{inds[0], inds[2],
//                                         inds[1], inds[3]});
//                 csEE.try_push(pair4_t{inds[0], inds[3],
//                                         inds[1], inds[2]});
//             });
//     }

//     timer.tock("ccd for inversion");
// }

void ABDSolver::precomputeFrictions(pol_t &pol, T xi)
{
    updateBasis = false;
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    T activeGap2 = dHat * dHat + (T)2.0 * xi * dHat;
    FPP.assignCounterFrom(PP);
    FPE.assignCounterFrom(PE);
    FPT.assignCounterFrom(PT);
    FEE.assignCounterFrom(EE);

    if (enableContact)
    {
        if (enableFriction)
        {
            auto numFPP = FPP.getCount();
            pol(range(numFPP),
                [vData = view<space>(vData), fricPP = view<space>(fricPP), PP = PP.port(),
                 FPP = FPP.port(), xi2 = xi * xi, activeGap2, kappa = kappa] __device__(int fppi) mutable {
                    auto fpp = PP[fppi];
                    FPP[fppi] = fpp;
                    auto x0 = vData.pack(VProps::xn, fpp[0]);
                    auto x1 = vData.pack(VProps::xn, fpp[1]);
                    auto dist2 = dist2_pp(x0, x1);
                    auto bGrad = barrier_gradient(dist2 - xi2, activeGap2, kappa);
                    fricPP(FPPProps::fn, fppi) = -bGrad * 2 * zs::sqrt(dist2);
                    fricPP.tuple<6>(FPPProps::basis, fppi) = point_point_tangent_basis(x0, x1);
                });
            auto numFPE = FPE.getCount();
            pol(range(numFPE),
                [vData = view<space>(vData), fricPE = view<space>(fricPE), PE = PE.port(),
                 FPE = FPE.port(), xi2 = xi * xi, activeGap2, kappa = kappa] __device__(int fpei) mutable {
                    auto fpe = PE[fpei];
                    FPE[fpei] = fpe;
                    auto p = vData.pack(VProps::xn, fpe[0]);
                    auto e0 = vData.pack(VProps::xn, fpe[1]);
                    auto e1 = vData.pack(VProps::xn, fpe[2]);
                    auto dist2 = dist2_pe(p, e0, e1);
                    auto bGrad = barrier_gradient(dist2 - xi2, activeGap2, kappa);
                    fricPE(FPEProps::fn, fpei) = -bGrad * 2 * zs::sqrt(dist2);
                    fricPE(FPEProps::yita, fpei) = point_edge_closest_point(p, e0, e1);
                    fricPE.tuple<6>(FPEProps::basis, fpei) = point_edge_tangent_basis(p, e0, e1);
                });
            auto numFPT = FPT.getCount();
            pol(range(numFPT),
                [vData = view<space>(vData), fricPT = view<space>(fricPT), PT = PT.port(),
                 FPT = FPT.port(), xi2 = xi * xi, activeGap2, kappa = kappa] __device__(int fpti) mutable {
                    auto fpt = PT[fpti];
                    FPT[fpti] = fpt;
                    auto p = vData.pack(VProps::xn, fpt[0]);
                    auto t0 = vData.pack(VProps::xn, fpt[1]);
                    auto t1 = vData.pack(VProps::xn, fpt[2]);
                    auto t2 = vData.pack(VProps::xn, fpt[3]);
                    auto dist2 = dist2_pt(p, t0, t1, t2);
                    auto bGrad = barrier_gradient(dist2 - xi2, activeGap2, kappa);
                    fricPT(FPTProps::fn, fpti) = -bGrad * 2 * zs::sqrt(dist2);
                    fricPT.tuple(FPTProps::beta, fpti) = point_triangle_closest_point(p, t0, t1, t2);
                    fricPT.tuple<6>(FPTProps::basis, fpti) = point_triangle_tangent_basis(p, t0, t1, t2);
                });
            auto numFEE = FEE.getCount();
            pol(range(numFEE),
                [vData = view<space>(vData), fricEE = view<space>(fricEE), EE = EE.port(),
                 FEE = FEE.port(), xi2 = xi * xi, activeGap2, kappa = kappa] __device__(int feei) mutable {
                    auto fee = EE[feei];
                    FEE[feei] = fee;
                    auto ea0 = vData.pack(VProps::xn, fee[0]);
                    auto ea1 = vData.pack(VProps::xn, fee[1]);
                    auto eb0 = vData.pack(VProps::xn, fee[2]);
                    auto eb1 = vData.pack(VProps::xn, fee[3]);
                    auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
                    auto bGrad = barrier_gradient(dist2 - xi2, activeGap2, kappa);
                    fricEE(FEEProps::fn, feei) = -bGrad * 2 * zs::sqrt(dist2);
                    fricEE.tuple(FEEProps::gamma, feei) = edge_edge_closest_point(ea0, ea1, eb0, eb1);
                    fricEE.tuple<6>(FEEProps::basis, feei) = edge_edge_tangent_basis(ea0, ea1, eb0, eb1);
                });
        }
    }

#if s_enableGround
#if s_enableBoundaryFriction
    if (enableGround)
    {
        if (enableBoundaryFriction)
            precomputeBoundaryFrictions(pol, activeGap2);
    }
#endif
#endif
}

void ABDSolver::precomputeBoundaryFrictions(pol_t &pol, T activeGap2)
{
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    // fmt::print("precomputing barrier frictions, activeGap2: {}\n", activeGap2);
    pol (range(svData.size()), 
         [vData = view<space>(vData),
          svData = view<space>(svData),
          kappa = kappa, 
          xi2 = xi * xi,
          activeGap2 = activeGap2,
          gn = groundNormal] __device__ (int svi) mutable
          {
              const auto vi = reinterpret_bits<Ti>(svData(SVProps::inds, svi));
              auto x = vData.pack(VProps::xn, vi);
              auto dist = gn.dot(x);
              auto dist2 = dist * dist;
              if (dist2 < activeGap2)
              {
                  auto bGrad = barrier_gradient(dist2 - xi2, activeGap2, kappa);
                  svData(SVProps::fn, svi) = -bGrad * 2 * dist;
              }
              else
              {
                  svData(SVProps::fn, svi) = 0;
              }
            //   printf("fn[%d]: %f\n", (int)svi, (float)svData(SVProps::fn, svi));
          });
} 

}// namespace tacipc
