#include <tacipc/solver/Solver.cuh>
// initialize, stepInitialize, substepInitialize

namespace tacipc
{
void ABDSolver::resetProfiler()
{
    totalSimulationElapsed = 0;  
    totalDCDElapsed = 0;         
    totalCCDElapsed = 0;         
    totalGradHessElapsed = 0;    
    totalLinearSolveElapsed = 0; 
    totalEnergyElapsed = 0;      
    totalCPUUpdateElapsed = 0;
}

void ABDSolver::suggestKappa(pol_t &pol)
{
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    if (kappa0 == 0) {
        /// kappaMin
        initKappa(pol);
        /// adaptive kappa
        { // tet-oriented
            T H_b = computeHb((T)1e-16 * boxDiagSize2, dHat * dHat);
            kappa = 1e11 * meanNodeMass / (4e-16 * boxDiagSize2 * H_b);
            kappaMax = 1000000 * kappa;
            if (kappa < kappaMin)
                kappa = kappaMin;
            if (kappa > kappaMax)
                kappa = kappaMax;
        }
        { // surf oriented (use framedt here)
            auto kappaSurf = dt * dt * meanSurfArea / 3 * dHat * largestMu();
            if (kappaSurf > kappa && kappaSurf < kappaMax) {
                kappa = kappaSurf;
            }
        }
    }
}

void ABDSolver::initKappa(pol_t &pol)
{
    // should be called after dHat set
    if (!enableContact)
        return;
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    kappaMin = 0;
}

typename ABDSolver::T ABDSolver::averageSurfEdgeLength(pol_t &pol) {
    using T = typename ABDSolver::T;
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    T sumSurfEdgeLengths = 0;
    std::size_t numSE = 0;
    auto accumulate = [&sumSurfEdgeLengths, &numSE, 
                       &temp = this->temp, &pol] (Handle &&bodyHandle){
                           auto&& [verts, edges] = std::visit(
                               [](auto &&bodyHandle){
                                  return std::make_tuple(bodyHandle.verts(),
                                                         bodyHandle.seInds());
                               }, bodyHandle);
                           auto seNum = edges.size();
                           numSE += seNum;
                           temp.resize(seNum);
                           pol(range(seNum),
                               [verts = view<space>({}, verts),
                               x0Tag = verts.getPropertyOffset("x0"),
                               edges = view<space>(edges, false_c, "init_edges"),
                               temp = view<space>(temp, false_c, "init_edges_temp")] __device__ (int i) mutable {
                                   auto inds = edges[i];
                                   temp[i] = (verts.pack<3>(x0Tag, inds[0]) -
                                              verts.pack<3>(x0Tag, inds[1])).norm();
                               });
                           sumSurfEdgeLengths += reduce(pol, temp);
                      };
    for (auto &&rbHandle : rigidBodies) {
        accumulate(rbHandle);
    }
    for (auto &&sbHandle : softBodies) {
        accumulate(sbHandle);
    }
    if (numSE)
        return sumSurfEdgeLengths / numSE;
    else
        return 0;
}
typename ABDSolver::T ABDSolver::averageSurfArea(pol_t &pol) {
    using T = typename ABDSolver::T;
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    
    T sumSurfAreas = 0;
    std::size_t numST = 0;
    auto accumulate = [&sumSurfAreas, &numST, 
                       &temp = this->temp, &pol] (Handle &&bodyHandle){
                           auto&& [verts, stInds] = std::visit(
                               [](auto &&bodyHandle){
                                   return std::make_tuple(bodyHandle.verts(),
                                                          bodyHandle.stInds());
                               }, bodyHandle);
                           auto stNum = stInds.size();
                           numST += stNum;
                           temp.resize(stNum);
                           pol(range(stNum),
                               [verts = view<space>({}, verts),
                                tris = view<space>(stInds, false_c, "tris"),
                                x0Tag = verts.getPropertyOffset("x0"),
                                temp = view<space>(temp , false_c, "accumulate_temp")] __device__ (int i) mutable {
                                    auto&& inds = tris[i];
                                    temp[i] = (verts.pack<3>(x0Tag, inds[1]) - verts.pack<3>(x0Tag, inds[0]))
                                            .cross(verts.pack<3>(x0Tag, inds[2]) - verts.pack<3>(x0Tag, inds[0]))
                                            .norm() / 2;
                               });
                           sumSurfAreas += reduce(pol, temp);
                      };
    for (auto &&rbHandle : rigidBodies) {
        accumulate(rbHandle);
    }
    for (auto &&sbHandle : softBodies) {
        accumulate(sbHandle);
    }
    if (numST)
        return sumSurfAreas / numST;
    else
        return 0;
}
typename ABDSolver::T ABDSolver::averageNodeMass(pol_t &pol) {
    using T = typename ABDSolver::T;
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    T sumNodeMass = 0;
    std::size_t numNode = 0;
    auto accumulate = [&sumNodeMass, &numNode,
                       &temp = this->temp, &pol](Handle &&bodyHandle){
                           auto&& verts = std::visit(
                               [](auto &&bodyHandle){
                                   return bodyHandle.verts();
                               }, bodyHandle);
                           auto vNum = verts.size();
                           numNode += vNum;
                           temp.resize(vNum);
                           pol(range(vNum),
                               [verts = view<space>({}, verts),
                                mTag = verts.getPropertyOffset("m"),
                                temp = view<space>(temp)] __device__ (int i) mutable {
                                   temp[i] = verts(mTag, i);
                               });
                           sumNodeMass += reduce(pol, temp);
                      };
    for (auto &&rbHandle : rigidBodies) {
        accumulate(rbHandle);
    }
    for (auto &&sbHandle : softBodies) {
        accumulate(sbHandle);
    }
    if (numNode)
        return sumNodeMass / numNode;
    else
        return 0;
}

void ABDSolver::updateWholeBoundingBox(pol_t &pol) {
    using namespace zs;
    if (pureRigidScene)
    {
        fmt::print("pure rigid scene\n");
        bvs.resize(1);
        auto merge_bv = []__host__ __device__(const bv_t &bv1, const bv_t &bv2)mutable{
            return merge(bv1, bv2);
        };
        zs::reduce(pol, std::begin(rbBvs), std::end(rbBvs), std::begin(bvs), bv_empty<bv_t>(), merge_bv);
        wholeBv = bvs.getVal(0);
    }
    else 
    {
        if (rigidSeBvhSize == 0)
        {
            fmt::print("no rigid body\n");
            wholeBv = softSeBvh.getTotalBox(pol);
        }
        else if (softSeBvhSize == 0)
        {
            fmt::print("no soft body\n");
            fmt::print("rigidSeBvhSize: {}\n", rigidSeBvh.getNumLeaves());
            wholeBv = rigidSeBvh.getTotalBox(pol);
            fmt::print("rigidSeBvhSize: {}\n", rigidSeBvh.getNumLeaves());
        }
        else
        {
            fmt::print("both rigid and soft body\n");
            wholeBv = merge(rigidSeBvh.getTotalBox(pol), softSeBvh.getTotalBox(pol));
        }
    }
    auto bv = wholeBv.value();
    boxDiagSize2 = (bv._max - bv._min).l2NormSqr();
}

void ABDSolver::initialize(pol_t &pol)
{
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    resetProfiler();

    // initialize rigid body data
    rbData.fillZeros();

    // initialize primitive indices
    stInds.resize(stNum());
    seInds.resize(seNum());
    svData.resize(svNum());

    // calculate some averages
    meanEdgeLength = averageSurfEdgeLength(pol);
    meanSurfArea = averageSurfArea(pol);
    meanNodeMass = averageNodeMass(pol);
    
    // stack primitives (surface vertices, edges, triangles, all vertices)
    auto stack_prims = [&stInds = this->stInds,
                        &seInds = this->seInds,
                        &svData = this->svData,
                        &vData = this->vData,
                        &gravity = this->gravity,
                        &dt = this->dt,
                        &pol] (Handle&& handle) {
                            auto&& [vertsFrom, svDataFrom, seIndsFrom, stIndsFrom, vOffset, svOffset, seOffset, stOffset] = std::visit(
                                [](auto&& handle) {
                                    return std::make_tuple(handle.verts(),
                                                           handle.svInds(),
                                                           handle.seInds(),
                                                           handle.stInds(),
                                                           handle.voffset(), 
                                                           handle.svoffset(), 
                                                           handle.seoffset(), 
                                                           handle.stoffset());
                                }, handle);
                            auto vNum = vertsFrom.size();
                            auto stNum = stIndsFrom.size();
                            auto seNum = seIndsFrom.size();
                            auto svNum = svDataFrom.size();
                            pol(range(stNum),
                                [stInds = view<space>(stInds),
                                stIndsFrom = view<space>(stIndsFrom),
                                stOffset = stOffset,
                                vOffset = vOffset] __device__ (int i) mutable {
                                    auto inds = stIndsFrom(i) + (int)vOffset;
                                    stInds.tuple(TIProps::inds, stOffset + i) = vec3{reinterpret_bits<T>((Ti)inds[0]), reinterpret_bits<T>((Ti)inds[1]), reinterpret_bits<T>((Ti)inds[2])};
                                    auto indsTo = stInds.pack(TIProps::inds, stOffset + i);
                                    // printf("stInds[%d]: %d %d %d\n", (int)(stOffset + i), (int)reinterpret_bits<Ti>(indsTo[0]), (int)reinterpret_bits<Ti>(indsTo[1]), (int)reinterpret_bits<Ti>(indsTo[2]));
                                });   
                            pol(range(seNum),
                                [seInds = view<space>(seInds),
                                seIndsFrom = view<space>(seIndsFrom),
                                seOffset = seOffset,
                                vOffset = vOffset] __device__ (int i) mutable {
                                    auto inds = seIndsFrom(i) + (int)vOffset;
                                    seInds.tuple(EIProps::inds, seOffset + i) = vec2{reinterpret_bits<T>((Ti)inds[0]), reinterpret_bits<T>((Ti)inds[1])};
                                    auto indsTo = seInds.pack(EIProps::inds, seOffset + i);
                                    // printf("seInds[%d]: %d %d\n", (int)(seOffset + i), (int)reinterpret_bits<Ti>(indsTo[0]), (int)reinterpret_bits<Ti>(indsTo[1]));
                                });
                            pol(range(svNum),
                                [svData = view<space>(svData),
                                svDataFrom = view<space>(svDataFrom),
                                svOffset = svOffset,
                                vOffset = vOffset] __device__ (int i) mutable {
                                    svData(SVProps::inds, svOffset + i) = reinterpret_bits<T>((Ti)(svDataFrom(i) + (int)vOffset));
                                });
                            pol(range(vNum),
                                [vData = view<space>(vData),
                                vDataFrom = view<space>({}, vertsFrom),
                                xTagFrom = vertsFrom.getPropertyOffset("x"),
                                vTagFrom = vertsFrom.getPropertyOffset("v"),
                                mTagFrom = vertsFrom.getPropertyOffset("m"),
                                volTagFrom = vertsFrom.getPropertyOffset("vol"),
                                extfTagFrom = vertsFrom.getPropertyOffset("extf"),
                                x0TagFrom = vertsFrom.getPropertyOffset("x0"),
                                isBCTagFrom = vertsFrom.getPropertyOffset("isBC"),
                                springWTagFrom = vertsFrom.getPropertyOffset("spring_w"),
                                springTargetTagFrom = vertsFrom.getPropertyOffset("spring_target"),
                                exclGrpIndTagFrom = vertsFrom.getPropertyOffset("excl_group_index"),
                                vOffset = vOffset,
                                gravity = gravity,
                                dt = dt] __device__ (int i) mutable {
                                    auto vi = i + vOffset;
                                    auto&& x = vDataFrom.pack(dim_c<3>, xTagFrom, i);
                                    auto&& v = vDataFrom.pack(dim_c<3>, vTagFrom, i);
                                    vData.tuple(VProps::xn, vi) = x;
                                    vData.tuple(VProps::vn, vi) = v;
                                    vData.tuple(VProps::xTilde, vi) = x + v * dt;
                                    vData.tuple(VProps::xHat, vi) = x;
                                    vData(VProps::m, vi) = vDataFrom(mTagFrom, i);
                                    vData.tuple(VProps::extf, vi) = vDataFrom(mTagFrom, i) * gravity;
                                    if (extfTagFrom != -1) // has external force property
                                        vData.tuple(VProps::extf, vi) = vData.pack(VProps::extf, vi) + vDataFrom.tuple(dim_c<3>, extfTagFrom, i);
                                    vData.tuple(VProps::x0, vi) = vDataFrom.pack(dim_c<3>, x0TagFrom, i);
                                    if (isBCTagFrom != -1) // has isBC property
                                        vData(VProps::isBC, vi) = reinterpret_bits<T>((Ti)vDataFrom(isBCTagFrom, i));
                                    else
                                        vData(VProps::isBC, vi) = reinterpret_bits<T>((Ti)0);
                                    vData(VProps::body, vi) = reinterpret_bits<T>((Ti)(-1)); // -1 for flex body vertices
                                    vData(VProps::vol, vi) = vDataFrom(volTagFrom, i);
                                    vData(VProps::ws, vi) = sqrt(vDataFrom(mTagFrom, i));
                                    if (springWTagFrom != -1) // has spring_w property
                                    {
                                        vData(VProps::springW, vi) = vDataFrom(springWTagFrom, i);
                                        vData.tuple(VProps::springTarget, vi) = vDataFrom.pack(dim_c<3>, springTargetTagFrom, i);
                                    } 
                                    else 
                                    {
                                        vData(VProps::springW, vi) = 0;
                                    }
                                    if (exclGrpIndTagFrom != -1) // has excl_group_index property
                                        vData(VProps::exclGrpIdx, vi, ti_c) = vDataFrom(exclGrpIndTagFrom, i, ti_c);
                                    else
                                        vData(VProps::exclGrpIdx, vi, ti_c) = -1;
                                });
                           };
    for (auto &&[bi, rbHandle] : enumerate(rigidBodies)) {
        fmt::print("Stacking rigid body ({}){}...\n", bi, rbHandle.name());
        stack_prims(rbHandle);
    }
    for (auto &&[bi, sbHandle] : enumerate(softBodies)) {
        fmt::print("Stacking soft body ({}){}...\n", bi, sbHandle.name());
        stack_prims(sbHandle);
    }

    fmt::print("Primitives stack done.\n");

    auto allocator = get_memory_source(memsrc_e::um, 0);
    auto&& rbNum = RigidBodyHandle::bodyNum;
    zs::Vector<vec3> centers_temp{allocator, rbNum};
    zs::Vector<vec12> qn_temp{allocator, rbNum};
    auto q_dot_temp = qn_temp;
    auto extf_temp = qn_temp;
    auto grav_temp = qn_temp;
    zs::Vector<mat12> M_temp{allocator, rbNum};
    zs::Vector<T> E_temp{allocator, rbNum};
    auto vol_temp = E_temp;
    auto isGhostTemp = E_temp;
    auto custom_stiffness_temp = E_temp;
    zs::Vector<int> isBC_temp{allocator, rbNum};
    zs::Vector<int> rigidSvO_temp{allocator, rbNum}; // surface vertex offset
    zs::Vector<int> rigidExclInd_temp{allocator, rbNum};
    for (auto&& [id, rbHandle] : zs::enumerate(rigidBodies)) {
        auto &verts = rbHandle.verts();
        rigidSvO_temp[id] = int(rbHandle.svoffset());
        centers_temp[id] = rbHandle.bodyTypeProperties().center;
        auto q0 = rbHandle.bodyTypeProperties().q0;
        auto &mass = rbHandle.bodyTypeProperties().m;
        auto gravF = gravity * mass;
        fmt::print("rb {} gravF: {} {} {}\n", id, gravF[0], gravF[1], gravF[2]);
        for (int k = 0; k != 3; ++k)
        {
            for (int d = 0; d != 3; ++d)
            {
                grav_temp[id][d] = gravF[d];
                grav_temp[id][3 + k * 3 + d] = gravF[k] * q0[d]; 
            }
        }
        extf_temp[id] = rbHandle.bodyTypeProperties().extForce;
        qn_temp[id] = rbHandle.bodyTypeProperties().q;
        q_dot_temp[id] = rbHandle.bodyTypeProperties().v;
        M_temp[id] = rbHandle.bodyTypeProperties().M;
        auto&& [E] = std::visit([](auto&& model) { 
                                   if constexpr (std::is_same_v<std::decay_t<decltype(model)>, std::monostate>)
                                       return std::tuple<T>{0.};
                                   else 
                                       return model.getParams(); 
                               }, rbHandle.model().affineModel);
        E_temp[id] = E;
        vol_temp[id] = rbHandle.bodyTypeProperties().vol;
        rigidExclInd_temp[id] = -1;  // TODO: bodyTypeProperties excl_group_tag
        isGhostTemp[id] = static_cast<T>(verts.size() == 0);
        isBC_temp[id] = rbHandle.bodyTypeProperties().isBC; 
        custom_stiffness_temp[id] = rbHandle.bodyTypeProperties().customBCStiffness;
        // fmt::print("isBC[{}]: {}\n", id, isBC_temp[id]);
        fmt::print("\t\t[debug-vol-mass] id = {}, vol = {}, mass = {}\n",
                   id, (float)rbHandle.bodyTypeProperties().vol, (float)rbHandle.bodyTypeProperties().m);
        pol(Collapse(verts.size()),
            [vData = view<space>(vData),
             verts = view<space>({}, verts),
             x0Tag = verts.getPropertyOffset("x0"),
             vOffset = rbHandle.voffset(),
             id = id] __device__(int i) mutable {
                 vData(VProps::body, vOffset + i) = reinterpret_bits<T>((Ti)id);
                 vData.tuple(VProps::JVec, vOffset + i) = verts.pack(dim_c<3>, x0Tag, i);
                 // TODO: remove "J" for it is just for debugging
                 auto pos0 = verts.pack(dim_c<3>, x0Tag, i);
                 zs::vec<T, 3, 12> J;
                 for (int ri = 0; ri < 3; ri++)
                 {
                     J(ri, ri) = 1.0;
                     for (int d = 0; d < 3; d++)
                         J(ri, 3 + ri * 3 + d) = pos0(d);
                 }
                 vData.tuple(VProps::J, vOffset + i) = J;
            });
    }
    int nonGhostSize = 0;
    for (int id = 0; id < RigidBodyHandle::bodyNum; id++)
    {
        if (isGhostTemp[id])
            vol_temp[id] = 0;
        else
            ++nonGhostSize;
    }
    auto avgVol = reduce(pol, vol_temp) / (T)nonGhostSize;
    fmt::print("# of rigid bodies: {}\n", RigidBodyHandle::bodyNum);
    pol(range(RigidBodyHandle::bodyNum),
        [rbData = view<space>(rbData),
         centers_temp = view<space>(centers_temp),
         extf_temp = view<space>(extf_temp),
         grav_temp = view<space>(grav_temp),
         qn_temp = view<space>(qn_temp),
         q_dot_temp = view<space>(q_dot_temp),
         M_temp = view<space>(M_temp),
         E_temp = view<space>(E_temp),
         vol_temp = view<space>(vol_temp),
         isGhostTemp = view<space>(isGhostTemp),
         isBC_temp = view<space>(isBC_temp),
         rigidSvO_temp = view<space>(rigidSvO_temp),
         rigidExclInd_temp = view<space>(rigidExclInd_temp),
         avgVol] __device__ (int id) mutable {
            rbData.tuple(RBProps::center, id) = centers_temp[id];
            rbData.tuple(RBProps::extf, id) = extf_temp[id];
            rbData.tuple(RBProps::grav, id) = grav_temp[id];
            rbData.tuple(RBProps::qn, id) = qn_temp[id];
            rbData.tuple(RBProps::qDot, id) = q_dot_temp[id];
            rbData.tuple(RBProps::M, id) = M_temp[id];
            rbData(RBProps::exclGrpIdx, id) = rigidExclInd_temp[id];
            rbData(RBProps::isGhost, id) = isGhostTemp[id];
            rbData(RBProps::vol, id) = isGhostTemp[id] > 0.5 ? avgVol : vol_temp[id];
            rbData(RBProps::E, id) = E_temp[id];
            rbData(RBProps::isBC, id) = reinterpret_bits<T>((Ti)isBC_temp[id]);
            // printf("isBC from[%d]: %d\n", id, isBC_temp[id]);
            // printf("isBC[%d]: %d\n", id, reinterpret_bits<Ti>(rbData(isBCTag, id)));
        });

    if constexpr (false)
    {
        T totalSoftVol = 0;
        for (auto &&[i, sbHandle] : enumerate(softBodies))
        {
            auto &eles = sbHandle.elems();
            temp.resize(count_warps(eles.size()));
            temp.reset(0);
            pol(range(eles.size()),
                [eles = view<space>({}, eles), 
                temp = view<space>(temp),
                n = eles.size()] __device__ (int id) mutable {
                    T vol = eles("vol", id);
                    if (vol < 0)
                        printf("!!!!!!soft elem %d has negative volume: %f\n", id, (float)vol);
                    reduce_to(id, n, vol, temp[id / 32]);
                });
            totalSoftVol += reduce(pol, temp);
        }
        fmt::print("Total soft body volume: {}\n", totalSoftVol);
    }

    fmt::print("constructing bvs...\n");
    {
        if (pureRigidScene)
        {
            zs::Vector<T> ret{temp.get_allocator(), 1};
            for (auto&& [rbi, rigidBody] : zs::enumerate(rigidBodies))
            {
                auto &svs = rigidBody.svInds();
                auto n = svs.size();
                if (n == 0) // skip ghost
                    continue;
                temp.resize(n * 3);
                pol(range(n),
                    [vData = view<space>(vData, false_c, "vData"),
                     svs = view<space>(svs, false_c, "svs"),
                     vOffset = rigidBody.voffset(),
                     temp = view<space>(temp, false_c, "temp"),
                     n] __device__ (int svi) mutable {
                        const auto vi = svs(svi) + vOffset;
                        auto x = vData.pack(VProps::xn, vi);
                        temp(svi) = x[0];
                        temp(svi + n) = x[1];
                        temp(svi + n * 2) = x[2];
                    });
                auto getPos = [&pol, &temp = temp, &n, &ret](auto initVal, const auto& op)
                {
                    pol.sync(true);
                    zs::reduce(pol, std::begin(temp), std::begin(temp) + n, std::begin(ret), (T)initVal, op);
                    auto pos_x = ret.getVal(0);
                    zs::reduce(pol, std::begin(temp) + n, std::begin(temp) + 2 * n, std::begin(ret), (T)initVal, op);
                    auto pos_y = ret.getVal(0);
                    zs::reduce(pol, std::begin(temp) + 2 * n, std::begin(temp) + 3 * n, std::begin(ret), (T)initVal, op);
                    auto pos_z = ret.getVal(0);
                    return vec3 {pos_x, pos_y, pos_z};
                };
                auto maxPos = getPos(T_min_c, zs::getmax<T>());
                auto minPos = getPos(T_max_c, zs::getmin<T>());
                // host version to skip the codes above
                rbRestBvs.setVal(bv_t(minPos, maxPos), rbi);
                rbBvs.setVal(bv_t(minPos, maxPos), rbi);
            }
        }

        // update ABD xn before constructing bvh!!!
        pol(range(RigidBodyHandle::vNum),
            [vData = view<space>(vData),
             rbData = view<space>(rbData)] __device__ (int vi) mutable {
                int ai = reinterpret_bits<Ti>(vData(VProps::body, vi));
                vData.tuple(VProps::xn, vi) = ABD_q2x(vData.pack(VProps::JVec, vi), rbData.pack(RBProps::qn, ai));
            });
        // build bvh
        if (pureRigidScene)
        {
            pure_rigid_retrieve_bounding_volumes(pol, vData, rbData, dHat, true, stInds, 0, rbBvs, culledStInds, bvs); 
            rigidStBvhSize = bvs.size(); 
            rigidStBvh.build(pol, bvs);
            fmt::print("rigidStBvhSize: {}\n", rigidStBvhSize);
            pure_rigid_retrieve_bounding_volumes(pol, vData, rbData, dHat, true, seInds, 0, rbBvs, culledSeInds, bvs); 
            rigidSeBvhSize = bvs.size(); 
            rigidSeBvh.build(pol, bvs);
        }
        else
        {
            retrieve_bounding_volumes(pol, vData, stInds, RigidBodyHandle::stNum, bvs, bvs1);
            rigidSeBvhSize = bvs.size();
            softSeBvhSize = bvs1.size();
            rigidStBvh.build(pol, bvs);
            softStBvh.build(pol, bvs1);
            retrieve_bounding_volumes(pol, vData, seInds, RigidBodyHandle::seNum, bvs, bvs1);
            rigidSeBvhSize = bvs.size();
            softSeBvhSize = bvs1.size();
            rigidSeBvh.build(pol, bvs);
            softSeBvh.build(pol, bvs1);
        }
        // ABD: remove coverts related
    }
    fmt::print("bvs construction ended.\n");

    // detect and ignore initial intersections (rigid bodies only for now)
    fmt::print("detecting initial collision...\n");
    if (RigidBodyHandle::bodyNum > 1)
    {
        // initialize rigid layers
        {
            auto allocator = get_memory_source(memsrc_e::um, 0);
            zs::Vector<int> layerTemp{allocator, RigidBodyHandle::bodyNum};
            for (auto&& [id, rbHandle] : zs::enumerate(rigidBodies)) 
                layerTemp[id] = rbHandle.layer();
            pol(range(RigidBodyHandle::bodyNum),
                [rbData = view<space>(rbData),
                layerTemp = view<space>(layerTemp)]__device__(int bi)mutable{
                    rbData(RBProps::layer, bi, ti_c) = layerTemp(bi);
                    // printf("rb %d layer: %d\n", bi, (int)rbData(RBProps::layer, bi, ti_c));
                });
        }
        int nRigid = RigidBodyHandle::bodyNum;
        zs::Vector<int> collide{temp.get_allocator(), (nRigid + 1) * nRigid / 2};
        zs::Vector<int> collideCnt{temp.get_allocator(), nRigid};
        collide.reset(0);
        collideCnt.reset(0);

        auto ij2ind = [nRigid]__host__ __device__(int i, int j) 
                      { 
                            if (i > j)
                            {
                                int tmp = i;
                                i = j;
                                j = tmp;
                            }
                            return (2 * nRigid - i + 1) * i / 2 + j - i; 
                      };
        auto ind2ij = [nRigid]__host__ __device__(int ind)
                      {
                            for (int i = 0; i < nRigid; ++i)
                            {
                                if (ind < (2 * nRigid - i) * (i + 1) / 2)
                                {
                                    return zs::make_tuple(i, ind - (2 * nRigid - i + 1) * i / 2 + i);
                                }
                            }
                      };

        // detect init collision
        int seNum;
        if (pureRigidScene)
            seNum = culledSeInds.size();
        else // soft body exist
            seNum = RigidBodyHandle::seNum;
        pol(range(seNum),
            [seInds = view<space>(seInds, false_c, "seInds"),
            stInds = view<space>(stInds, false_c, "stInds"),
            vData = view<space>(vData, false_c, "vData"),
            rbData = view<space>(rbData, false_c, "rbData"),
            bvh = view<space>(rigidStBvh),
            culledSeInds = view<space>(culledSeInds), 
            culledStInds = view<space>(culledStInds), 
            collisionMat = view<space>(collisionMat, false_c, "collisionMat"),
            collide = view<space>(collide, false_c, "collide"),
            collideCnt = view<space>(collideCnt, false_c, "collideCnt"),
            ij2ind, ind2ij,
            thickness = xi + dHat,
            pureRigidScene = pureRigidScene] __device__ (int sei) mutable {
                if (pureRigidScene)
                    sei = culledSeInds[sei];
                auto eInds = seInds.pack(EIProps::inds, sei).template reinterpret_bits<Ti>();
                auto ev0 = vData.pack(VProps::xn, eInds[0]);
                auto ev1 = vData.pack(VProps::xn, eInds[1]);
                int ebody = vData(VProps::body, eInds[0], ti_c);
                int elayer = rbData(RBProps::layer, ebody, ti_c);
                int egroup = vData(VProps::exclGrpIdx, eInds[0], ti_c);
                auto [mi, ma] = get_bounding_box(ev0, ev1);
                auto bv = bv_t{mi - thickness, ma + thickness};
                // printf("testing init collision for edge(%d-%d) from body(%d)\n", (int)eInds[0], (int)eInds[1], ebody);
                bvh.iter_neighbors(
                    bv, 
                    [&](int sti)
                    {
                        if (pureRigidScene)
                            sti = culledStInds[sti];
                        auto tInds = stInds.pack(TIProps::inds, sti).template reinterpret_bits<Ti>();
                        if (eInds[0] == tInds[0] || eInds[1] == tInds[0] ||
                            eInds[0] == tInds[1] || eInds[1] == tInds[1] ||
                            eInds[0] == tInds[2] || eInds[1] == tInds[2])
                            return;
                        int tbody = vData(VProps::body, tInds[0], ti_c);
                        if (ebody == tbody && ebody >= 0)
                            return;
                        int tlayer = rbData(RBProps::layer, tbody, ti_c);
                        if (!collisionMat(elayer, tlayer))
                            return;
                        // printf("elayer: %d, tlayer: %d\n", elayer, tlayer);
                        if (collide(ij2ind(ebody, tbody)))
                            return;
                        int tgroup = vData(VProps::exclGrpIdx, tInds[0], ti_c);
                        if (egroup == tgroup && egroup >= 0) // TODO: this is strange
                            return;
                        auto tv0 = vData.pack(VProps::xn, tInds[0]);
                        auto tv1 = vData.pack(VProps::xn, tInds[1]);
                        auto tv2 = vData.pack(VProps::xn, tInds[2]);
                        // printf("testing init collision [edge(%d-%d), triangle(%d-%d-%d)] / [body(%d), body(%d)]\n", (int)eInds[0], (int)eInds[1], (int)tInds[0], (int)tInds[1], (int)tInds[2], ebody, tbody);
                        bool intersected = et_intersected(ev0, ev1, tv0, tv1, tv2);
                        if (!intersected) // no strict intersection
                        {
                            // point-triangle
                            auto dist = dist_pt_unclassified(ev0, tv0, tv1, tv2);
                            dist = min(dist, dist_pt_unclassified(ev1, tv0, tv1, tv2));

                            // edge-edge
                            dist = min(dist, dist_ee_unclassified(ev0, ev1, tv0, tv1));
                            dist = min(dist, dist_ee_unclassified(ev0, ev1, tv0, tv2));
                            dist = min(dist, dist_ee_unclassified(ev0, ev1, tv1, tv2));

                            intersected = dist <= thickness;
                        }
                        if (intersected)
                        {
                            int old_colli = atomic_exch(exec_cuda, &collide(ij2ind(ebody, tbody)), 1);
                            if (!old_colli)
                                printf("[init collision] detected between body %d(layer %d) and %d(layer %d)\n", ebody, elayer, tbody, tlayer);
                            atomic_add(exec_cuda, &collideCnt(ebody), 1 - old_colli);
                            atomic_add(exec_cuda, &collideCnt(tbody), 1 - old_colli);
                        }
                    }
                );
            });
        
        collide = collide.clone({memsrc_e::host, -1});
        collideCnt = collideCnt.clone({memsrc_e::host, -1});
        fmt::print("init collide cnt:\n");
        for (int i = 0; i < collideCnt.size(); ++i)
        {
            fmt::print("({}) {}\t", i, collideCnt[i]);
        }
        fmt::print("\n");
        std::vector<int> collideCntHeap(collideCnt.size());
        auto ompPol = omp_exec();
        constexpr auto ompSpace = execspace_e::openmp;
        ompPol(range(collideCnt.size()),
            [&collideCntHeap](int i) mutable {
                collideCntHeap[i] = i;
            });
        
        auto cmpCnt = [&](int const &a, int const &b)
            {
                return collideCnt[a] < collideCnt[b];
            };

        std::vector<std::vector<int>> layers(layerCps);
        std::vector<int> idInLayer(nRigid);
        for (auto&& [bi, rbHandle] : enumerate(rigidBodies))
        {
            layers[rbHandle.layer()].emplace_back(bi);
            idInLayer[bi] = layers[rbHandle.layer()].size() - 1;
            // fmt::print("rigid body {:2d} in layer {:2d}\n", bi, rbHandle.layer());
            // fmt::print("already {} bodies in layer {}\n", layers[rbHandle.layer()].size(), rbHandle.layer());
        } 
        std::vector<int> emptyLayers{};
        for (auto&& [layer, bodies] : enumerate(layers))
        {
            if (bodies.empty())
            {
                emptyLayers.emplace_back(layer);
                // fmt::print("layer {} is empty\n", layer);
            }
        }

        // collect layer info
        for (auto [bi, rbHandle] : enumerate(rigidBodies))
        {
            if (collideCnt[bi] > 0)
            {
                auto dstLayer = emptyLayers.back();
                emptyLayers.pop_back();
                collisionMat.duplicateLayer(dstLayer, rbHandle.layer());
                rbHandle.setLayer(dstLayer);
            }
        }
        for (int bi = 0; bi < nRigid; ++bi)
        {
            for (int bj = bi + 1; bj < nRigid; ++bj)
            {
                if (collide[ij2ind(bi, bj)])
                {
                    auto layeri = rigidBodies[bi].layer();
                    auto layerj = rigidBodies[bj].layer(); 
                    collisionMat.setCollision(layeri, layerj, false);
                }
            }
        }
        // std::vector<std::vector<int>> layers(layerCps);
        // std::vector<int> idInLayer(nRigid);
        // for (auto&& [bi, rbHandle] : enumerate(rigidBodies))
        // {
        //     layers[rbHandle.layer()].emplace_back(bi);
        //     idInLayer[bi] = layers[rbHandle.layer()].size() - 1;
        //     // fmt::print("rigid body {:2d} in layer {:2d}\n", bi, rbHandle.layer());
        //     // fmt::print("already {} bodies in layer {}\n", layers[rbHandle.layer()].size(), rbHandle.layer());
        // } 
        // std::vector<int> emptyLayers{};
        // for (auto&& [layer, bodies] : enumerate(layers))
        // {
        //     if (bodies.empty())
        //     {
        //         emptyLayers.emplace_back(layer);
        //         fmt::print("layer {} is empty\n", layer);
        //     }
        // }

        // auto changeLayer = [&](auto bi, auto newLayer)
        // {
        //     auto oldLayer = rigidBodies[bi].layer();
        //     layers[oldLayer].erase(layers[oldLayer].begin() + idInLayer[bi]);
        //     rigidBodies[bi].setLayer(newLayer);
        //     layers[newLayer].emplace_back(bi);
        //     idInLayer[bi] = layers[newLayer].size() - 1;
        // };

        // while (true)
        // {
        //     std::make_heap(collideCntHeap.begin(), collideCntHeap.end(), cmpCnt);
        //     int i = collideCntHeap.front();
        //     std::swap(collideCntHeap.front(), collideCntHeap.back());
        //     collideCntHeap.pop_back();
        //     int cnt = collideCnt[i];
        //     if (cnt <= 0)
        //         break;
            
        //     int ilayer = emptyLayers.back(); // new layer for i
        //     emptyLayers.pop_back();
        //     // remove i from its original layer
        //     changeLayer(i, ilayer);
        //     fmt::print("rigid body {:2d} moved to layer {:2d} (as i)\n", i, ilayer);

        //     for (int j = 0; j < nRigid; ++j)
        //     {
        //         int ind = ij2ind(i, j);
        //         if (collide[ind])
        //         {
        //             int jlayer_old = rigidBodies[j].layer(); // old layer of j
        //             int jlayer = emptyLayers.back(); // new layer for j
        //             emptyLayers.pop_back();
        //             collisionMat.duplicateLayer(jlayer, jlayer_old); // copy old collision relationship
        //             collisionMat.setCollision(ilayer, jlayer, false); // ignore collision between ilayer and jlayer
        //             changeLayer(j, jlayer);
        //             fmt::print("rigid body {:2d} moved to layer {:2d} (as j)\n", j, jlayer);

        //             collide[ind] = 0;
        //             --collideCnt[j];
        //             if (--cnt)
        //             {
        //                 for (int ki = 0; ki < layers[jlayer_old].size(); ++ki) // for other bodies colliding with i in jlayer_old
        //                 {
        //                     int k = layers[jlayer_old][ki];
        //                     int ind_ik = ij2ind(i, k);
        //                     if (collide[ind_ik])
        //                     {
        //                         changeLayer(k, jlayer);
        //                         --ki;
        //                         fmt::print("rigid body {:2d} moved to layer {:2d} (as k)\n", k, jlayer);

        //                         collide[ind_ik] = 0;
        //                         --collideCnt[k];
        //                         if (!(--cnt))
        //                             break;
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }
        // till now, all initial rigid collisions shall be ignored in the collision matrix
    }

    // set collision layer
    {
        auto allocator = get_memory_source(memsrc_e::um, 0);
        // rigid
        zs::Vector<int> layerTemp{allocator, RigidBodyHandle::bodyNum};
        for (auto&& [id, rbHandle] : zs::enumerate(rigidBodies)) {
            layerTemp[id] = rbHandle.layer();
            pol(range(rbHandle.verts().size()),
                [vData = view<space>(vData),
                 layer = rbHandle.layer(),
                 vOffset = rbHandle.voffset()]__device__(int vi_of)mutable{
                    vData(VProps::layer, vi_of + vOffset, ti_c) = layer;
                    // printf("rb vertex %d layer: %d\n", vi_of, (int)vData(VProps::layer, vi_of + vOffset, ti_c));
                });
        }
        pol(range(RigidBodyHandle::bodyNum),
            [rbData = view<space>(rbData),
             layerTemp = view<space>(layerTemp)]__device__(int bi)mutable{
                rbData(RBProps::layer, bi, ti_c) = layerTemp(bi);
                printf("rb %d layer: %d\n", bi, (int)rbData(RBProps::layer, bi, ti_c));
            });
        
        // soft
        for (auto&& [id, sbHandle] : enumerate(softBodies)){
            pol(range(sbHandle.verts().size()),
                [vData = view<space>(vData),
                 layer = sbHandle.layer(),
                 vOffset = sbHandle.voffset()]__device__(int vi_of)mutable{
                    vData(VProps::layer, vi_of + vOffset, ti_c) = layer;
                    // printf("sb vertex %d layer: %d\n", vi_of, (int)vData(VProps::layer, vi_of + vOffset, ti_c));
                });
        }
    }

    // run step initialization
    stepInitialize(pol);
}

/// @brief 
/// @param pol 
void ABDSolver::stepInitialize(pol_t &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    substeps = -1;

    if (enableContact) {
        PP.reset();
        PE.reset();
        PT.reset();
        EE.reset();

        PPM.reset();
        PEM.reset();
        EEM.reset();
        
        if (enableFriction)
        {
            FPP.reset();
            FPE.reset();
            FPT.reset();
            FEE.reset();
        }

        csPT.reset();
        csEE.reset();
    }

    /// spatial accel structs
    /// frontManageRequired = true;
    /// bvh without front is initialized in ABDSystem::initialize()
    updateWholeBoundingBox(pol);

    /// update grad pn residual tolerance
    targetGRes = pnRel * std::sqrt(boxDiagSize2);
    fmt::print("box diag size: {}, targetGRes: {}\n", std::sqrt(boxDiagSize2), targetGRes);
    pol(range(RigidBodyHandle::bodyNum), 
        [rbData = view<space>(rbData)]__device__(int bi) mutable{
            rbData.tuple(RBProps::lambda, bi) = vec12::zeros(); 
    });
    pol(range(SoftBodyHandle::vNum),
        [vData = view<space>(vData), 
         sb_vOffset = RigidBodyHandle::vNum] __device__ (int id) mutable {
            int vi = sb_vOffset + id; 
            vData.tuple(VProps::lambda, vi) = vec3::zeros(); 
        }); 

    // copy extf & isBC & has_spring & spring_target
    constexpr auto tempTagsNum = 5; 
    temp12.resize(RigidBodyHandle::bodyNum * tempTagsNum); 
    zs::Vector<Ti> temp_hasSpring{temp.get_allocator(), RigidBodyHandle::bodyNum};
    // enableSelfBarrier.resize(prims.size()); 
    auto update_extf = [&vData = vData, 
                        &pol](Handle &&handle){
                                auto&& [vNum, verts, vOffset, extfTag, bodyType] = std::visit(
                                    [](auto &&handle){
                                        return std::make_tuple(handle.vNum, 
                                                               handle.verts(),
                                                               handle.voffset(),
                                                               handle.verts().getPropertyOffset("extf"),
                                                               handle.value_type);
                                    }, handle);
                                if (extfTag != -1)
                                    pol(range(vNum),
                                        [vData = view<space>(vData),
                                        verts = view<space>({}, verts),
                                        vOffset = vOffset,
                                        extfTag = extfTag] __device__ (int i) mutable {
                                            auto vi = i + vOffset;
                                            vData.tuple(VProps::extf, vi) = verts.pack(dim_c<3>, extfTag, i);
                                        });
                                pol(range(verts.size()), 
                                    [verts = proxy<space>({}, verts), 
                                    vData = view<space>(vData), 
                                    vOffset = vOffset,
                                    hasSpring = verts.hasProperty("spring_target"),
                                    springWTag = verts.getPropertyOffset("spring_w"),
                                    springTgtTag = verts.getPropertyOffset("spring_target")] __device__ (int i) mutable {
                                        auto vi = i + vOffset; 
                                        if (hasSpring)
                                        {
                                            vData(VProps::springW, vi) = verts(springWTag, i); 
                                            vData.tuple(VProps::springTarget, vi) = verts.pack(dim_c<3>, springTgtTag, i); 
                                        } else {
                                            vData(VProps::springW, vi) = 0;
                                        }
                                    }); 
                            };
    for (auto &&rbHandle : rigidBodies) {
        update_extf(rbHandle);
    }
    for (auto &&sbHandle : softBodies) {
        update_extf(sbHandle);
    }
    for (auto &&[id, rbHandle] : zs::enumerate(rigidBodies)) 
    {
        auto &verts = rbHandle.verts();

        auto &bodyTypeProperties = rbHandle.bodyTypeProperties();
        temp12.setVal(bodyTypeProperties.BCTarget, id * tempTagsNum);
        temp12.setVal(bodyTypeProperties.extForce, id * tempTagsNum + 1);
        temp_hasSpring.setVal(bodyTypeProperties.hasSpring, id);
        temp12.setVal(bodyTypeProperties.springTarget, id * tempTagsNum + 2);
        if (bodyUpToDate)
        {
            temp12.setVal(bodyTypeProperties.q, id * tempTagsNum + 3);
            temp12.setVal(bodyTypeProperties.v, id * tempTagsNum + 4);
        }
        // enableSelfBarrier[id] = true;
    }
    for (auto &&[id, sbHandle] : zs::enumerate(softBodies)) 
    {
        auto &verts = sbHandle.verts();
        pol(range(verts.size()), 
                [verts = view<space>({}, verts), 
                 vData = view<space>(vData), 
                 BCTgtTag = verts.getPropertyOffset("BCTarget"),
                 vOffset = sbHandle.voffset()] __device__ (int i) mutable {
                    auto vi = i + vOffset; 
                    int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi)); 
                    if (isBC)
                        vData.tuple(VProps::xTilde, vi) = verts.pack<3>(BCTgtTag, i); 
                 }); 
        // TODO: enableSelfBarrier
        // auto &bodyTypeProperties = sbHandle.bodyTypeProperties();
        // enableSelfBarrier[bi] = bodyTypeProperties.enableSelfBarrier;
    }
    pol(range(RigidBodyHandle::bodyNum), 
        [rbData = view<space>(rbData), 
         temp12 = view<space>(temp12),
         temp_hasSpring = view<space>(temp_hasSpring),
         bodyUpToDate = bodyUpToDate] __device__ (int id) mutable {
            int isBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, id));
            if (isBC)
                rbData.tuple(RBProps::qTilde, id) = temp12[id * tempTagsNum];
            rbData.tuple(RBProps::extf, id) = temp12[id * tempTagsNum + 1];
            rbData(RBProps::hasSpring, id) = reinterpret_bits<T>(temp_hasSpring[id]);
            rbData.tuple(RBProps::springTarget, id) = temp12[id * tempTagsNum + 2];
            if (bodyUpToDate)
            {
                rbData.tuple(RBProps::qn, id) = temp12[id * tempTagsNum + 3]; 
                rbData.tuple(RBProps::qDot, id) = temp12[id * tempTagsNum + 4];
            }
    });

    pol(range(RigidBodyHandle::vNum),
        [vData = view<space>(vData), 
        rbData = view<space>(rbData),
        tempTagsNum = tempTagsNum] __device__ (int vi) mutable {
            auto bi = reinterpret_bits<Ti>(vData(VProps::body, vi));
            auto qn = rbData.pack(RBProps::qn, bi);
            auto JVec = vData.pack(VProps::JVec, vi); 
            vData.tuple(VProps::xn, vi) = ABD_q2x(JVec, qn); 
        }); 
}

void ABDSolver::substepInitialize(pol_t &pol) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    // accumulate substeps
    ++substeps;
    // ABD: update q_tilde instead of xtilde
    pol(range(rbData.size()),
        [rbData = view<space>(rbData),
         dt = dt] __device__ (int ai) mutable {
            auto qn = rbData.pack(RBProps::qn, ai);
            rbData.tuple(RBProps::qHat, ai) = qn; // mainly for kinematic constraints 
            int isBC = reinterpret_bits<Ti>(rbData(RBProps::isBC, ai)); 
            if (isBC)
                return; 
            auto deltaQ = rbData.pack(RBProps::qDot, ai) * dt;
            auto newQ = qn + deltaQ;
            rbData.tuple(RBProps::qTilde, ai) = newQ;
    });
    pol(range(vData.size()),
        [vData = view<space>(vData), 
         sb_vOffset = RigidBodyHandle::vNum, dt = dt] __device__ (int vi) mutable {
            auto xn = vData.pack(VProps::xn, vi);
            vData.tuple(VProps::xHat, vi) = xn; 
            if (vi >= sb_vOffset)
            {
                int isBC = reinterpret_bits<Ti>(vData(VProps::isBC, vi)); 
                if (isBC)
                    return;
                auto deltaX = vData.pack(VProps::vn, vi) * dt;
                auto newX = xn + deltaX;
                vData.tuple(VProps::xTilde, vi) = newX;
            }
    });
}

ABDSolver::ABDSolver(const std::vector<BodySP<T, tvLength>> &bodies, 
                     const CollisionMatrix<> &collisionMat,
                     T dt, int frameSubsteps, bool enableCG, bool enableInversionPrevention, bool enablePureRigidOpt,
                     bool enableGround, bool enableContact, bool enableMollification,
                     bool enableContactEE, bool enableFriction, bool enableBoundaryFriction, bool enableSoftBC,
                     std::size_t layerCps,
                     std::size_t estNumCps, std::size_t dynHessCps,
                     T kinematicALCoef, //bool enableAL,
                     T pnRel, T cgRel, int fricIterCap, int PNCap, int CGCap, int CCDCap,
                     T kappa0, bool useAbsKappaDhat, T fricMu, T boundaryKappa,
                     T springStiffness, T abdSpringStiffness, 
                     T xi, T dHat, T epsv, T kinematicALTol,
                     T consTol, T armijoParam,
                     vec3 groundNormal, T gravity)
    : temp{estNumCps, zs::memsrc_e::um, 0}, 
      temp3{temp.get_allocator(), 0},
      temp3_1{temp.get_allocator(), 0},
      temp12{temp.get_allocator(), 0},
      collisionMat{collisionMat},
      PP{estNumCps}, 
      PE{estNumCps}, 
      PT{estNumCps}, 
      EE{estNumCps}, 
      // mollification
      PPM{estNumCps}, 
      PEM{estNumCps}, 
      EEM{estNumCps}, 
      // friction
      FPP{estNumCps}, 
      fricPP{temp.get_allocator(), estNumCps},
      FPE{estNumCps}, 
      fricPE{temp.get_allocator(), estNumCps},
      FPT{estNumCps},
      fricPT{temp.get_allocator(), estNumCps},
      FEE{estNumCps}, 
      fricEE{temp.get_allocator(), estNumCps},
      bvs{temp.get_allocator(), 0},
      bvs1{temp.get_allocator(), 0},
      csPT{estNumCps},
      csEE{estNumCps},
      sysHess{dynHessCps},
      rbData{temp.get_allocator(), 0},
      vData{temp.get_allocator(), 0},
      dofData{temp.get_allocator(), 0},
      stInds{temp.get_allocator(), 0},
      seInds{temp.get_allocator(), 0},
      svData{temp.get_allocator(), 0},
      rbBvs{temp.get_allocator(), 0},
      rbRestBvs{temp.get_allocator(), 0},
      culledStInds{temp.get_allocator(), 0},
      culledSeInds{temp.get_allocator(), 0},
    //   ppConsPairs{0, zs::memsrc_e::um, 0},
      dt(dt),
      frameDt(dt*frameSubsteps),
      frameSubsteps(frameSubsteps),
//      curRatio{0},
//      enableAL{enableAL},
      estNumCps(estNumCps),
      dynHessCps(dynHessCps),
      enableCG{enableCG}, enableInversionPrevention{enableInversionPrevention}, enablePureRigidOpt{enablePureRigidOpt},
      enableGround{enableGround}, enableContact{enableContact},
      enableMollification{enableMollification}, enableContactEE{enableContactEE}, enableFriction{enableFriction},
      enableBoundaryFriction{enableBoundaryFriction},
      enableSoftBC{enableSoftBC}, 
      groundNormal{groundNormal[0], groundNormal[1], groundNormal[2]},
      kinematicALCoef{kinematicALCoef}, pnRel{pnRel}, cgRel{cgRel},
      fricIterCap{fricIterCap}, PNCap{PNCap}, CGCap{CGCap}, CCDCap{CCDCap},
      kappa0{kappa0}, useAbsKappaDhat{useAbsKappaDhat}, kappa{kappa0}, fricMu{fricMu}, boundaryKappa{boundaryKappa}, springStiffness{springStiffness}, abdSpringStiffness{abdSpringStiffness},
      xi{xi}, dHat{dHat}, epsv{epsv},
      kinematicALTol{kinematicALTol}, consTol{consTol}, armijoParam{armijoParam},
      gravity{0, -gravity, 0} {

    if (collisionMat.isEmpty)
    {
        fmt::print("Collision matrix is empty, initializing with layerCps: {}\n", layerCps);
        this->collisionMat = CollisionMatrix<>(temp.get_allocator(), layerCps);
    }
    else 
    {
        this->collisionMat = this->collisionMat.clone(temp.memoryLocation());
    }

    // create body handles
    for (auto &bodyPtr : bodies) {
        std::visit([&rigidBodies = rigidBodies, &softBodies = softBodies](auto &bodyPtr) {
                       if constexpr (std::decay_t<decltype(*bodyPtr)>::bodyType == BodyType::Rigid)
                           rigidBodies.emplace_back(bodyPtr);
                       else // soft 
                           softBodies.emplace_back(bodyPtr);
                   }, bodyPtr);
    }

    // optimize for pure rigid scene
    if (enablePureRigidOpt && SoftBodyHandle::bodyNum == 0)
    {
        pureRigidScene = true; 
        rbBvs.resize(RigidBodyHandle::bodyNum);
        rbRestBvs.resize(RigidBodyHandle::bodyNum);
        softStBvhSize = 0;
        softSeBvhSize = 0;
    }

    fmt::print("#rigid bodies: {}, #soft bodies: {}, #rigid body vertices: {}, #soft body vertices: {}, #dofs: {}\n",
               RigidBodyHandle::bodyNum, SoftBodyHandle::bodyNum, RigidBodyHandle::vNum, SoftBodyHandle::vNum, dof());
    fmt::print("num total obj <verts, surfV, surfE, surfT>: {}, {}, {}, {}\n", vNum(),
               svNum(), seNum(), stNum());

    rbData.resize(RigidBodyHandle::bodyNum);
    vData.resize(vNum());
    dofData.resize(dof());


    auto cudaPol = zs::cuda_exec();
    initialize(cudaPol); // update vData, bvh, boxsize, targetGRes

    // adaptive dhat, targetGRes, kappa
    if (useAbsKappaDhat)
    {
        if (epsv == 0)
        {
            this->epsv = this->dHat;
        }
        else 
        {
            this->epsv *= this->dHat;
        }
    }
    else
    {
        // dHat (static)
        this->dHat *= std::sqrt(boxDiagSize2);
        // adaptive epsv (static)
        if (epsv == 0) {
            this->epsv = this->dHat;
        } else {
            this->epsv *= this->dHat;
        }
        // kappa (dynamic)
        suggestKappa(cudaPol);
        if (kappa0 != 0) {
            fmt::print("manual kappa: {}\n", this->kappa);
        }
    }
    fmt::print("auto dHat: {}, epsv (friction): {}\n", this->dHat, this->epsv);
    // find collision con
    findBarrierCollisions(cudaPol, xi); // for init collision check, with the help of GetABDContactNum
    // output adaptive setups
    fmt::print("construction ended\n"); 

    energies.emplace_back(std::make_unique<InertialEnergy<T>>()); 
#if s_enableAffineEnergy
    if (RigidBodyHandle::bodyNum > 0)
        energies.emplace_back(std::make_unique<AffineEnergy<T>>());
#endif // s_enableAffineEnergy
#if s_enableSoftEnergy
    if (SoftBodyHandle::bodyNum > 0)
        energies.emplace_back(std::make_unique<SoftEnergy<T>>());
#endif // s_enableSoftEnergy
#if s_enableGround
    if (enableGround)
        energies.emplace_back(std::make_unique<GroundBarrierEnergy<T>>());
#endif // s_enableGround
#if s_enableContact
    if (enableContact)
        energies.emplace_back(std::make_unique<BarrierEnergy<T>>());
#endif // s_enableContact
    // constraint energy
    // kinematic constraint
    energies.emplace_back(std::make_unique<KinematicConstraintEnergy<T>>());
    // spring constraint
    energies.emplace_back(std::make_unique<SpringConstraintEnergy<T>>());
}
} // namespace tacipc
