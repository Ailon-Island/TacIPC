#pragma once
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <optional>
#include <tacipc/macros.hpp>
#include <tacipc/dynamicBuffer.cuh>
#include <tacipc/generation.cuh>
#include <tacipc/meta.hpp>
#include <tacipc/solver/Properties.cuh>
// #include <tacipc/solver/VecWrapper.cuh>
// #include <tacipc/solver/VWrapper.cuh> 
// #include <tacipc/solver/VecWrapper.cuh>
// #include <tacipc/solver/VWrapper.cuh> 
#include <tacipc/solver/TVWrapper.cuh>
#include <tacipc/solver/CollisionMatrix.cuh>
#include <tacipc/utils.cuh>
#include <zensim/container/Bvh.hpp>
#include <zensim/container/Bvs.hpp>
#include <zensim/container/Bvtt.hpp>
#include <zensim/container/HashTable.hpp>
#include <zensim/container/Vector.hpp>
#include <zensim/cuda/Cuda.h>
#include <zensim/cuda/execution/ExecutionPolicy.cuh>
#include <zensim/execution/Atomics.hpp>
#include <zensim/geometry/Distance.hpp>
#include <zensim/math/Vec.h>
#include <zensim/math/matrix/SparseMatrix.hpp>
#include <zensim/resource/Resource.h>
namespace tacipc
{

namespace cg = ::cooperative_groups;

class ABDSolver
{
  public:
    static constexpr std::size_t tvLength = 32;

    using size_type = std::size_t;
    using T = double;
    using Ti = zs::conditional_t<zs::is_same_v<T, double>, zs::i64, zs::i32>;
    static constexpr auto ti_c = zs::wrapt<Ti>{};
    using dtiles_t = zs::TileVector<T, tvLength>;
    using ftiles_t = zs::TileVector<float, tvLength>;
    using vec2 = zs::vec<T, 2>;
    using vec3 = zs::vec<T, 3>;
    using vec6 = zs::vec<T, 6>;
    using vec12 = zs::vec<T, 12>;
    using vec3f = zs::vec<float, 3>;
    using ivec3 = zs::vec<int, 3>;
    using ivec2 = zs::vec<int, 2>;
    using mat2 = zs::vec<T, 2, 2>;
    using mat3 = zs::vec<T, 3, 3>;
    using mat4 = zs::vec<T, 4, 4>;
    using mat12 = zs::vec<T, 12, 12>;
    // TODO: remove dense matrices, only record dense elements
    using mat3x12 = zs::vec<T, 3, 12>;
    using mat6x24 = zs::vec<T, 6, 24>;
    using mat9x24 = zs::vec<T, 9, 24>;
    using mat12x24 = zs::vec<T, 12, 24>;
    using pair_t = zs::vec<int, 2>;
    using pair3_t = zs::vec<int, 3>;
    using pair4_t = zs::vec<int, 4>;
    using dpair_t = zs::vec<Ti, 2>;
    using dpair3_t = zs::vec<Ti, 3>;
    using dpair4_t = zs::vec<Ti, 4>;
    using bvh_t = zs::LBvh<3, int, T>;
    using bvs_t = zs::LBvs<3, int, T>;
    using bvfront_t = zs::BvttFront<int, int>;
    using bv_t = zs::AABBBox<3, T>;
    using RigidState = vec12;
    using vec2i = zs::vec<int, 2>;
    using vec3i = zs::vec<int, 3>;
    using pol_t = zs::CudaExecutionPolicy;
    static constexpr auto T_min_c = zs::limits<T>::lowest(); 
    static constexpr auto T_max_c = zs::limits<T>::max(); 
    // properties
    using RBProps = AffineBodyProperties<dtiles_t>;
    using VProps = VertexProperties<dtiles_t>;
    using DOFProps = DoFProperties<dtiles_t>;
    using SVProps = SurfaceVertexProperties<dtiles_t>;
    using EIProps = IndsProperties<2, dtiles_t>;
    using TIProps = IndsProperties<3, dtiles_t>;
    using FPPProps = FrictionPointPointProperties<dtiles_t>;
    using FPEProps = FrictionPointEdgeProperties<dtiles_t>;
    using FPTProps = FrictionPointTriangleProperties<dtiles_t>;
    using FEEProps = FrictionEdgeEdgeProperties<dtiles_t>;

    // BodyHandle
    template <BodyType BType> struct BodyHandle
    {
        // template parameter: triangle or tet
        static constexpr auto value_type = BType;
        using size_type = std::size_t;
        using body_ptr_t = std::conditional_t<BType == BodyType::Rigid,
                                              RigidBodySP<T, tvLength>,
                                              SoftBodySP<T, tvLength>>;
        inline static size_type vNum = 0, svNum = 0, seNum = 0, stNum = 0,
                                bodyNum = 0;
        static zs::Vector<size_type> indices;

        template <
            class BodyT,
            zs::enable_if_all<
                is_body_v<BodyT>, BodyT::bodyType == value_type, std::is_same_v<typename BodyT::value_t, T>, BodyT::tvLength == tvLength> =
                0>
        BodyHandle(std::shared_ptr<BodyT> bodyPtr) : bodyPtr{bodyPtr}
        {
            constexpr auto codim = BodyT::codim;
            vOffset = vNum;
            svOffset = svNum;
            seOffset = seNum;
            stOffset = stNum;
                        if constexpr (codim == 2)
            {
                // triangle mesh
                vNum += bodyPtr->nVerts();
                svNum += bodyPtr->nVerts();
                seNum += bodyPtr->nEdges();
                stNum += bodyPtr->nTris();
            }
            else if constexpr (codim == 3)
            {
                // tet body
                vNum += bodyPtr->nVerts();
                svNum += bodyPtr->nSurfVerts();
                seNum += bodyPtr->nSurfEdges();
                stNum += bodyPtr->nSurfTris();
            }
            else
            {
                static_assert(dependent_false<codim>::value);
            }
            bodyPtr->id = bodyNum;
            bodyNum++;
        }

        std::string &name()
        {
            return std::visit(
                [](auto &bodyPtr) -> auto&
                {
                    return bodyPtr->name;
                },
                bodyPtr
            );
        }

        const std::string &name() const
        {
            return std::visit(
                [](auto &bodyPtr)
                {
                    return bodyPtr->name;
                },
                bodyPtr
            );
        }

        auto &svInds() // surface vertice indices
        {
            zs::Vector<int> *ret;
            std::visit(
                [this, &ret](auto &bodyPtr)
                {
                    constexpr auto codim =
                        std::decay_t<decltype(*bodyPtr)>::codim;
                    if constexpr (codim == 2)
                        ret = &bodyPtr->vertInds;
                    else
                        ret = &bodyPtr->surfVertInds;
                },
                bodyPtr);
            return *ret;
        }

        auto &seInds() // surface edge indices
        {
            zs::Vector<vec2i> *ret;
            std::visit(
                [this, &ret](auto &bodyPtr)
                {
                    constexpr auto codim =
                        std::decay_t<decltype(*bodyPtr)>::codim;
                    if constexpr (codim == 2)
                        ret = &bodyPtr->edgeInds;
                    else
                        ret = &bodyPtr->surfEdgeInds;
                },
                bodyPtr);
            return *ret;
        }

        const auto &seInds() const // surface edge indices
        {
            zs::Vector<vec2i> *ret;
            std::visit(
                [this, &ret](auto &bodyPtr)
                {
                    constexpr auto codim =
                        std::decay_t<decltype(*bodyPtr)>::codim;
                    if constexpr (codim == 2)
                        ret = &bodyPtr->edgeInds;
                    else
                        ret = &bodyPtr->surfEdgeInds;
                },
                bodyPtr);
            return *ret;
        }

        auto &stInds() // surface triangle indices
        {
            zs::Vector<vec3i> *ret;
            std::visit(
                [this, &ret](auto &bodyPtr)
                {
                    constexpr auto codim =
                        std::decay_t<decltype(*bodyPtr)>::codim;
                    if constexpr (codim == 2)
                        ret = &bodyPtr->triInds;
                    else
                        ret = &bodyPtr->surfTriInds;
                },
                bodyPtr);
            return *ret;
        }

        const auto &stInds() const // surface triangle indices
        {
            zs::Vector<vec3i> *ret;
            std::visit(
                [this, &ret](auto &bodyPtr)
                {
                    constexpr auto codim =
                        std::decay_t<decltype(*bodyPtr)>::codim;
                    if constexpr (codim == 2)
                        ret = &bodyPtr->triInds;
                    else
                        ret = &bodyPtr->surfTriInds;
                },
                bodyPtr);
            return *ret;
        }

        auto &verts()
        {
            return std::visit([](auto &bodyPtr) -> auto &
                              { return bodyPtr->verts; }, bodyPtr);
        }

        auto &verts() const
        {
            return std::visit([](auto const &bodyPtr) -> auto &const
                              { return bodyPtr->verts; }, bodyPtr);
        }

        auto &elems()
        {
            dtiles_t *ret;
            std::visit(
                [this, &ret](auto &bodyPtr)
                {
                    constexpr auto codim =
                        std::decay_t<decltype(*bodyPtr)>::codim;
                    if constexpr (codim == 2)
                        ret = &bodyPtr->tris;
                    else // codim == 3
                        ret = &bodyPtr->tets;
                },
                bodyPtr);
            return *ret;
        }

        auto &elems() const
        {
            dtiles_t *ret;
            std::visit( 
                [this, &ret](auto &bodyPtr)
                {
                    constexpr auto codim =
                        std::decay_t<decltype(*bodyPtr)>::codim;
                    if (codim == 2)
                        ret = &bodyPtr->tris;
                    else
                        ret = &bodyPtr->tets;
                },
                bodyPtr);
            return *ret;
        }

        auto &meshVerts()
        {
            return std::visit([](auto &bodyPtr) -> auto &
                              { return bodyPtr->meshVerts; }, bodyPtr);
        }

        auto &meshVerts() const
        {
            return std::visit([](auto const &bodyPtr) -> auto &const
                              { return bodyPtr->meshVerts; }, bodyPtr);
        }

        auto &meshTris()
        {
            return std::visit([](auto &bodyPtr) -> auto &
                              { return bodyPtr->meshTris; }, bodyPtr);
        }

        auto &meshTris() const
        {
            return std::visit([](auto const &bodyPtr) -> auto &const
                              { return bodyPtr->meshTris; }, bodyPtr);
        }

        void updateMesh()
        {
            std::visit([](auto &bodyPtr) { bodyPtr->updateMesh(); }, bodyPtr);
        }

        auto &bodyTypeProperties()
        {
            return std::visit([](auto &bodyPtr) -> auto &
                              { return bodyPtr->bodyTypeProperties; }, bodyPtr);
        }

        auto &bodyTypeProperties() const
        {
            return std::visit([](auto &bodyPtr) -> auto &const
                              { return bodyPtr->bodyTypeProperties; }, bodyPtr);
        }

        auto &model()
        {
            return std::visit([](auto &bodyPtr) -> auto &
                              { return bodyPtr->getModel(); }, bodyPtr);
        }

        auto &model() const
        {
            return std::visit([](auto const &bodyPtr) -> auto &const
                              { return bodyPtr->getModel(); }, bodyPtr);
        }

        constexpr size_type layer() const 
        {
            size_type ret;
            std::visit([&ret](const auto &bodyPtr) { ret = bodyPtr->layer; },
                       bodyPtr);
            return ret;
        }

        constexpr void setLayer(size_type layer)
        {
            std::visit([layer](auto &bodyPtr) { bodyPtr->layer = layer; },
                       bodyPtr);

        }

        constexpr bool isBC() const 
        {
            return bodyTypeProperties().isBC;
        }

        size_type codim() const
        {
            size_type ret;
            std::visit([&ret](const auto &bodyPtr) { ret = bodyPtr->codim; },
                       bodyPtr);
            return ret;
        }

        constexpr size_type voffset() const
        {
            if constexpr (value_type == BodyType::Rigid)
                return vOffset;
            else // Soft
                return vOffset + RigidBodyHandle::vNum;
        }

        constexpr size_type svoffset() const
        {
            if constexpr (value_type == BodyType::Rigid)
                return svOffset;
            else // Soft
                return svOffset + RigidBodyHandle::svNum;
        }

        constexpr size_type seoffset() const
        {
            if constexpr (value_type == BodyType::Rigid)
                return seOffset;
            else // Soft
                return seOffset + RigidBodyHandle::seNum;
        }

        constexpr size_type stoffset() const
        {
            if constexpr (value_type == BodyType::Rigid)
                return stOffset;
            else // Soft
                return stOffset + RigidBodyHandle::stNum;
        }

        body_ptr_t bodyPtr;
        size_type vOffset, svOffset, seOffset, stOffset;
        // ZenoParticles::bv_t bv;
    };
    using RigidBodyHandle = BodyHandle<BodyType::Rigid>;
    using SoftBodyHandle = BodyHandle<BodyType::Soft>;
    using Handle = std::variant<RigidBodyHandle, SoftBodyHandle>;
    std::vector<RigidBodyHandle> rigidBodies;
    std::vector<SoftBodyHandle> softBodies;

    static std::size_t dof()
    {
        return RigidBodyHandle::bodyNum * 12 + SoftBodyHandle::vNum * 3;
    }

    static std::size_t bodyNum()
    {
        return RigidBodyHandle::bodyNum + SoftBodyHandle::bodyNum;
    }

    static std::size_t vNum()
    {
        return RigidBodyHandle::vNum + SoftBodyHandle::vNum;
    }

    static std::size_t stNum()
    {
        return RigidBodyHandle::stNum + SoftBodyHandle::stNum;
    }

    static std::size_t seNum()
    {
        return RigidBodyHandle::seNum + SoftBodyHandle::seNum;
    }

    static std::size_t svNum()
    {
        return RigidBodyHandle::svNum + SoftBodyHandle::svNum;
    }

    // Hessian
    template <typename T_> struct SystemHessian
    {
        using T = T_;
        using vec3 = zs::vec<T, 3>;
        using mat3 = zs::vec<T, 3, 3>;
        using pair_t = zs::vec<int, 2>;
        using spmat_t = zs::SparseMatrix<mat3, true>;
        using dyn_hess_t = zs::tuple<pair_t, mat3>;

        zs::Vector<mat3> softHess;
        zs::Vector<mat3> rigidHess;
        DynamicBuffer<dyn_hess_t> dynHess;
        spmat_t spmat{}; // _ptrs, _inds, _vals

        SystemHessian(std::size_t dynColCps)
            : softHess{0, zs::memsrc_e::device, 0},
              rigidHess{0, zs::memsrc_e::device, 0}, dynHess{dynColCps},
              spmatIs{0, zs::memsrc_e::um, 0}, spmatJs{0, zs::memsrc_e::um, 0}
        {
            clear();
        }

        void clear()
        {
            softHess.resize(SoftBodyHandle::vNum);
            auto pol = zs::cuda_exec().device(0);
            using namespace zs;
            constexpr auto space = zs::execspace_e::cuda;
            constexpr auto matEps = mat3::identity() * zs::limits<T>::epsilon(); 
            pol(range(softHess.size()),
                [softHess = view<space>(softHess)] __device__(int tid) mutable
                { softHess[tid] = matEps; });
            // 16 3x3s for a 12x12
            rigidHess.resize(RigidBodyHandle::bodyNum *
                             10); // 10 for upper triangle 4x4 blocks
            pol(range(rigidHess.size()),
                [rigidHess = view<space>(rigidHess)] __device__(
                    int tid) mutable { rigidHess[tid] = mat3::zeros(); });
            pol(range(RigidBodyHandle::bodyNum),
                [rigidHess = view<space>(rigidHess)] __device__(
                    int tid) mutable { 
                        rigidHess[tid * 10] = matEps;
                        rigidHess[tid * 10 + 4] = matEps;
                        rigidHess[tid * 10 + 7] = matEps;
                        rigidHess[tid * 10 + 9] = matEps;
                    });
            dynHess.reset();
        }

        static constexpr std::size_t
        toUpper4x4Ind(std::size_t i, std::size_t j) // from (row, col)
        {
            return i * (9 - i) / 2 + j - i;
        }

        static constexpr zs::tuple<std::size_t, std::size_t>
        fromUpper4x4Ind(std::size_t ind) // to (row, col)
        {
            if (ind < 4)
                return zs::make_tuple(0, ind);
            if (ind < 7)
                return zs::make_tuple(1, ind - 4 + 1);
            if (ind < 9)
                return zs::make_tuple(2, ind - 7 + 2);
            tacipc_assert(ind == 9, "[fromUpper4x4Ind]\tind should be an "
                                  "integer value within 0-9 !");
            return zs::make_tuple(3, 3);
        }

        template <bool Atomic = true, class VecT,
                  zs::enable_if_t<zs::is_vec<VecT>::value> = 0>
        static __forceinline__ __device__ void addVec(VecT &dst,
                                                      const VecT &addVal)
        {
            for (typename VecT::index_type i = 0; i != VecT::extent; ++i)
                if constexpr (Atomic)
                    atomic_add(zs::exec_cuda, &dst.val(i), addVal.val(i));
                else
                    dst.val(i) += addVal.val(i);
        }

        void buildInit()
        {
            spmatIs.resize(softHess.size() + rigidHess.size());
            spmatJs.resize(softHess.size() + rigidHess.size());
            constexpr auto space = zs::execspace_e::cuda;
            using namespace zs;
            auto pol = zs::cuda_exec().device(0);
            pol(range(rigidHess.size()),
                [spmatIs = view<space>(spmatIs),
                 spmatJs = view<space>(spmatJs)] __device__(int tid) mutable
                {
                    auto rbi = tid / 10;
                    auto [blocki, blockj] = fromUpper4x4Ind(tid % 10);
                    spmatIs[tid] = rbi * 4 + blocki;
                    spmatJs[tid] = rbi * 4 + blockj;
                });
            pol(range(softHess.size()),
                [spmatIs = view<space>(spmatIs), spmatJs = view<space>(spmatJs),
                 offset = rigidHess.size(),
                 spmatOffset =
                     rigidHess.size() / 10 * 4] __device__(int tid) mutable
                {
                    spmatIs[offset + tid] = spmatOffset + tid;
                    spmatJs[offset + tid] = spmatOffset + tid;
                });
        }

        void build()
        {
            // NOTE: make sure buildInit is called at least once before calling
            // build
            auto dof = ABDSolver::dof();
            auto dynOffset = softHess.size() + rigidHess.size();
            auto dynHessCnt = dynHess.getCount();
            auto nElem = dynHessCnt + dynOffset;
            // softHess, rigidHess, dynHess -> spmat
            spmat = spmat_t{(int)(dof / 3), (int)(dof / 3),
                            zs::memsrc_e::device, 0};
            spmatIs.resize(nElem);
            spmatJs.resize(nElem);
            auto pol = zs::cuda_exec().device(0);
            constexpr auto space = zs::execspace_e::cuda;
            using namespace zs;
            pol(range(dynHess.getCount()),
                [spmatIs = view<space>(spmatIs), spmatJs = view<space>(spmatJs),
                 dynHess = dynHess.port(),
                 offset = dynOffset] __device__(int tid) mutable
                {
                    auto &[inds, mat] = dynHess[tid];
                    spmatIs[tid + offset] = inds[0];
                    spmatJs[tid + offset] = inds[1];
                });

            // only construct the uppper part
            fmt::print("spmatIs.size: {}, spmatJs.size: {}, dof: {}\n",
                       spmatIs.size(), spmatJs.size(), dof);
            for (std::size_t i = 0; i < spmatIs.size(); i++)
            {
                if (spmatIs[i] > spmatJs[i])
                {
                    fmt::print(
                        "[not upper-triangle!!!]\tspmatIJ({}, {}), i: {}\n",
                        spmatIs[i], spmatJs[i], i);
                }
                if (spmatIs[i] >= dof / 3 | spmatJs[i] >= dof / 3)
                {
                    fmt::print("[out of upper bound index!!!]\tspmatIJ({}, "
                               "{}), i: {}\n",
                               spmatIs[i], spmatJs[i], i);
                }
            }
            spmat.build(pol, (int)(dof / 3), (int)(dof / 3), range(spmatIs),
                        range(spmatJs), zs::false_c);
            // sort without values
            spmat.localOrdering(pol, false_c);
            spmat._vals.resize(spmat.nnz());
            pol(range(spmat.nnz()),
                [spmat = view<space>(spmat)] __device__(int i) mutable
                { spmat._vals[i] = mat3::zeros(); });
            // add softHess, rigidHess, dynHess to spmat
            pol(range(rigidHess.size()),
                [rigidHess = view<space>(rigidHess),
                 spmat = view<space>(spmat)] __device__(int tid) mutable
                {
                    auto rbi = tid / 10;
                    auto [blocki, blockj] = fromUpper4x4Ind(tid % 10);
                    auto spi = rbi * 4 + blocki, spj = rbi * 4 + blockj;
                    auto loc = spmat._ptrs[spi] + blockj - blocki;
                    // tacipc_assert(loc < spmat._ptrs[spi + 1],
                    //             "rigid hessian block row index unexpected!");
                    //             // comment this for now
                    tacipc_assert_default(spmat._inds[loc] == spj);
                    auto &mat = const_cast<mat3 &>(spmat._vals[loc]);
                    addVec<false>(mat, rigidHess[tid]);
                });
            pol(range(softHess.size()),
                [softHess = view<space>(softHess), spmat = view<space>(spmat),
                 spmatOffset =
                     rigidHess.size() / 10 * 4] __device__(int tid) mutable
                {
                    auto &hess = softHess[tid];
                    auto loc = spmat._ptrs[tid + spmatOffset];
                    tacipc_assert(spmat._inds[loc] == tid + spmatOffset,
                                "failed to locate soft diag hessian block");
                    auto &mat = const_cast<mat3 &>(spmat._vals[loc]);
                    addVec<false>(mat, hess);
                });
            pol(range(dynHessCnt),
                [dynHess = dynHess.port(),
                 spmat = view<space>(spmat)] __device__(int tid) mutable
                {
                    auto &[inds, hess] = dynHess[tid];
                    auto loc = spmat.locate(inds[0], inds[1], zs::true_c);
                    auto &mat = const_cast<mat3 &>(spmat._vals[loc]);
                    addVec<true>(mat, hess);
                });
        }

      private:
        zs::Vector<int> spmatIs, spmatJs;
    };
    template <zs::execspace_e space, typename T_> struct SystemHessianView
    {
        using sys_hess_t = SystemHessian<T_>;
        using vec3 = typename sys_hess_t::vec3;
        using mat3 = typename sys_hess_t::mat3;
        using pair_t = typename sys_hess_t::pair_t;
        using spmat_t = typename sys_hess_t::spmat_t;
        using dyn_hess_t = typename sys_hess_t::dyn_hess_t;
        using dyn_buffer_t = DynamicBuffer<dyn_hess_t>;
        using soft_vector_t = zs::Vector<mat3>;
        using rigid_vector_t = zs::Vector<mat3>;

        using spmat_view_t =
            RM_CVREF_T(zs::view<space>(std::declval<spmat_t &>(), zs::true_c));
        using dyn_buffer_view_t =
            RM_CVREF_T(std::declval<dyn_buffer_t &>().port());
        using soft_vector_view_t =
            RM_CVREF_T(zs::view<space>(std::declval<soft_vector_t &>(), zs::false_c));
        using rigid_vector_view_t =
            RM_CVREF_T(zs::view<space>(std::declval<rigid_vector_t &>(), zs::false_c));

        SystemHessianView(sys_hess_t &sys)
            : spmat{zs::view<space>(sys.spmat, zs::true_c)},
              dynHess{sys.dynHess.port()},
              softHess{zs::view<space>(sys.softHess, zs::false_c, "softHess")},
              rigidHess{
                  zs::view<space>(sys.rigidHess, zs::false_c, "rigidHess")}
        {
        }

        template <unsigned int TileSize = 8, class T,
                  zs::enable_if_t<std::is_fundamental_v<T>> = 0>
        __forceinline__ __device__ T
        tile_shfl(cooperative_groups::thread_block_tile<
                      TileSize, cooperative_groups::thread_block> &tile,
                  T var, int srcLane)
        {
            return tile.shfl(var, srcLane);
        }

        template <unsigned int TileSize = 8, class VecT,
                  zs::enable_if_t<zs::is_vec<VecT>::value> = 0>
        __forceinline__ __device__ VecT
        tile_shfl(cooperative_groups::thread_block_tile<
                      TileSize, cooperative_groups::thread_block> &tile,
                  const VecT &var, int srcLane)
        {
            VecT ret{};
            for (typename VecT::index_type i = 0; i != VecT::extent; ++i)
                ret.val(i) = tile_shfl(tile, var.val(i), srcLane);
            return ret;
        }

        template <class cg_tile_t = cg::thread_block_tile<8, cg::thread_block>,
                  class VecTM, class iVecTI, class jVecTI,
                  zs::enable_if_all<
                      VecTM::dim == 2,
                      VecTM::template range_t<0>::value == iVecTI::extent * 3,
                      iVecTI::dim == 1,
                      VecTM::template range_t<1>::value == jVecTI::extent * 3,
                      jVecTI::dim == 1> = 0>
        __forceinline__ __device__ void
        addOffDiagHessTile(cg_tile_t &tile, const iVecTI &iInds,
                           const jVecTI &jInds, const VecTM &hess)
        {
            // assume active pattern 0...001111
            const int cap = __popc(tile.ballot(1));
            auto laneId = tile.thread_rank();
            for (int bi = 0; bi < iVecTI::extent; bi++)
            {
                if (iInds[bi] < 0)
                    continue;
                for (int bj = 0; bj < jVecTI::extent; bj++)
                {
                    if (jInds[bj] < 0)
                        continue;
                    if (iInds[bi] > jInds[bj])
                        continue;
                    auto no = dynHess.next_index(tile);
                    auto &[inds, mat] = dynHess[no];
                    for (int d = laneId; d < 9; d += cap)
                        mat.val(d) = hess(bi * 3 + d / 3, bj * 3 + d % 3);
                    if (laneId == 0)
                        inds = pair_t{iInds[bi], jInds[bj]};
                }
            }   
        }

        template <unsigned int TileSize = 8, class VecTM, class iVecTI,
                  class jVecTI,
                  zs::enable_if_all<
                      VecTM::dim == 2,
                      VecTM::template range_t<0>::value == iVecTI::extent * 3,
                      iVecTI::dim == 1,
                      VecTM::template range_t<1>::value == jVecTI::extent * 3,
                      jVecTI::dim == 1> = 0>
        __forceinline__ __device__ void
        addOffDiagHess(const iVecTI &iInds, const jVecTI &jInds, VecTM &hess)
        {
            auto tile = cg::tiled_partition<TileSize>(cg::this_thread_block());
            bool has_work = true;
            zs::u32 work_queue = tile.ballot(has_work);
            while (work_queue)
            {
                auto cur_rank = __ffs(work_queue) - 1;
                auto cur_work = tile_shfl(tile, hess, cur_rank); // TODO
                auto cur_iInds = tile_shfl(tile, iInds, cur_rank);
                auto cur_jInds = tile_shfl(tile, jInds, cur_rank);
                addOffDiagHessTile(tile, cur_iInds, cur_jInds, cur_work);
                if (tile.thread_rank() == cur_rank)
                    has_work = false;
                work_queue = tile.ballot(has_work);
            }
            return;
        }

        template <class VecTM, class iVecTI, class jVecTI, class RBDataT = std::false_type, class VDataT = std::false_type,
                  zs::enable_if_all<
                      VecTM::dim == 2,
                      VecTM::template range_t<0>::value == iVecTI::extent * 3,
                      iVecTI::dim == 1,
                      VecTM::template range_t<1>::value == jVecTI::extent * 3,
                      jVecTI::dim == 1> = 0>
        __forceinline__ __device__ void
        addOffDiagHessNoTile(const iVecTI &iInds,
                           const jVecTI &jInds, const VecTM &hess, 
                           bool includePre = false, RBDataT rbData = {}, VDataT vData = {}, std::size_t sb_vOffset = 0)
        {
            for (int bi = 0; bi < iVecTI::extent; bi++) 
            {
                for (int bj = 0; bj < jVecTI::extent; bj++)
                {
                    if (iInds[bi] < 0 || jInds[bj] < 0)
                        continue;
                    if (iInds[bi] > jInds[bj])
                        continue;
#if s_enableAutoPrecondition
                    if constexpr (!std::is_same_v<VDataT, std::false_type> && !std::is_same_v<RBDataT, std::false_type>)
                        if (includePre && (iInds[bi] == jInds[bj]))
                        {
                            if (iInds[bi] / 4 < rigidHess.size() / 10) // rigid hess
                            {
                                int k = iInds[bi] % 4;
                                for (int d = 0; d < 9; ++d)
                                {
                                    int di = d / 3;
                                    int dj = d % 3;
                                    
                                    atomic_add(zs::exec_cuda, &rbData(RBProps::Pre, k * 9 + d, iInds[bi] / 4),
                                                hess(bi * 3 + di, bj * 3 + dj));
                                }
                            }
                            else // soft hess
                            {
                                int vi = iInds[bi] - rigidHess.size() / 10 * 4 + sb_vOffset;
                                for (int d = 0; d < 9; ++d)
                                {
                                    int di = d / 3;
                                    int dj = d % 3;
                                    atomic_add(zs::exec_cuda, &vData(VProps::Pre, d, vi),
                                                hess(bi * 3 + di, bj * 3 + dj));
                                }
                            }
                        }
#endif
                    auto no = dynHess.next_index_no_tile();
                    auto &[inds, mat] = dynHess[no];                    
                    for (int d = 0; d < 9; d += 1)
                        mat.val(d) = hess(bi * 3 + d / 3, bj * 3 + d % 3);
                    inds = pair_t{iInds[bi], jInds[bj]};
                    // printf("Add offdiag hess[%d, %d]:\n%f %f %f\n%f %f %f\n%f %f %f\n", (int)iInds[bi], (int)jInds[bj], (float)mat(0, 0), (float)mat(0, 1), (float)mat(0, 2), (float)mat(1, 0), (float)mat(1, 1), (float)mat(1, 2), (float)mat(2, 0), (float)mat(2, 1), (float)mat(2, 2));
                }
            }
        }

        template <std::size_t TileSize = 8, bool Atomic = true, class VDataT = std::false_type>
        __forceinline__ __device__ void
        addSoftHess(int softVertInd, const mat3 &hess, bool hasWork = true, bool includePre = false, VDataT vData = {}, std::size_t sb_vOffset = 0)
        {
            auto tile = cg::tiled_partition<TileSize>(cg::this_thread_block());
            zs::u32 work_queue = tile.ballot(hasWork);
            while (work_queue)
            {
                auto cur_rank = __ffs(work_queue) - 1;
                auto cur_work = tile_shfl(tile, hess, cur_rank); // TODO
                auto cur_softvi = tile_shfl(tile, softVertInd, cur_rank);
                auto &mat = softHess[cur_softvi];
                const int cap = __popc(tile.ballot(1));
                auto laneId = tile.thread_rank();
                for (int d = laneId; d < 9; d += cap)
                {
                    if constexpr (Atomic)
                        atomic_add(zs::exec_cuda, &mat.val(d),
                                   cur_work(d / 3, d % 3));
                    else
                        mat.val(d) += cur_work(d / 3, d % 3);
                    
#if s_enableAutoPrecondition
                    if constexpr (!std::is_same_v<VDataT, std::false_type>)
                        if (includePre)
                        {
                            if constexpr (Atomic)
                                atomic_add(zs::exec_cuda, &vData(VProps::Pre, d, cur_softvi + sb_vOffset),
                                            cur_work(d / 3, d % 3));
                            else
                                vData(VProps::Pre, d, cur_softvi + sb_vOffset) += cur_work(d / 3, d % 3);
                        }
#endif
                }
                if (tile.thread_rank() == cur_rank)
                    hasWork = false;
                work_queue = tile.ballot(hasWork);
            }
            return;
        }

        template <bool Atomic = true, class VDataT = std::false_type>
        __forceinline__ __device__ void
        addSoftHessNoTile(int softVertInd, const mat3 &hess, bool includePre = false, VDataT vData = {}, std::size_t sb_vOffset = 0)
        {
            auto &mat = softHess[softVertInd];
            for (int d = 0; d < 9; ++d)
            {
                int di = d / 3;
                int dj = d % 3;
                if constexpr (Atomic)
                    atomic_add(zs::exec_cuda, &mat.val(d),
                                hess(di, dj));
                else
                    mat.val(d) += hess(di, dj);
#if s_enableAutoPrecondition
                if constexpr (!std::is_same_v<VDataT, std::false_type>)
                    if (includePre)
                    {
                        if constexpr (Atomic)
                            atomic_add(zs::exec_cuda, &vData(VProps::Pre, d, softVertInd + sb_vOffset),
                                        hess(di, dj));
                        else
                            vData(VProps::Pre, d, softVertInd + sb_vOffset) += hess(di, dj);
                    }
#endif
            }
            return;
        }

        template <std::size_t TileSize = 8, bool Atomic = true, class RBDataT = std::false_type>
        __forceinline__ __device__ void addRigidHess(int rigidBodyHandleInd,
                                                     const mat12 &hess,
                                                     bool hasWork = true, 
                                                     bool includePre = false, 
                                                     RBDataT rbData = {})
        {
            auto tile = cg::tiled_partition<TileSize>(cg::this_thread_block());
            zs::u32 work_queue = tile.ballot(hasWork);
            while (work_queue)
            {
                auto cur_rank = __ffs(work_queue) - 1;
                auto cur_work = tile_shfl(tile, hess, cur_rank); // TODO
                auto cur_rbi = tile_shfl(tile, rigidBodyHandleInd, cur_rank);
                const int cap = __popc(tile.ballot(1));
                auto laneId = tile.thread_rank();
                auto blockCnt = 0;
                for (int i = 0; i < 4; i++)
                    for (int j = i; j < 4; j++)
                    {
                        auto &mat = rigidHess[10 * cur_rbi + blockCnt++];
                        int iOffset = i * 3;
                        int jOffset = j * 3;
                        for (int d = laneId; d < 9; d += cap)
                        {
                            int di = d / 3, dj = d % 3;
                            if constexpr (Atomic)
                                atomic_add(
                                    zs::exec_cuda, &mat(di, dj),
                                    cur_work(iOffset + di, jOffset + dj));
                            else
                                mat(di, dj) +=
                                    cur_work(iOffset + di, jOffset + dj);
                        }
                    }
#if s_enableAutoPrecondition
                // preconditioner, use diag blocks
                if constexpr (!std::is_same_v<RBDataT, std::false_type>)
                    if (includePre)
                    {
                        for (int k = 0; k < 4; ++k)
                        {
                            int iOffset = k * 3;
                            for (int d = laneId; d < 9; d += cap)
                            {
                                int di = d / 3;
                                int dj = d % 3;
                                if constexpr (Atomic)
                                    atomic_add(
                                        zs::exec_cuda, &rbData(RBProps::Pre, k * 9 + d, cur_rbi),
                                        cur_work(iOffset + di, iOffset + dj));
                                else
                                    rbData(k * 9 + d, cur_rbi) +=
                                        cur_work(iOffset + di, iOffset + dj);
                            }
                        }
                    }
#endif
                if (tile.thread_rank() == cur_rank)
                    hasWork = false;
                work_queue = tile.ballot(hasWork);
            }
            return;
        }

        template <bool Atomic = true, class RBDataT = std::false_type>
        __forceinline__ __device__ void addRigidHessNoTile(int rigidBodyHandleInd,
                                                     const mat12 &hess, bool includePre = false, RBDataT rbData = {})
        {
            auto blockCnt = 0;
            for (int i = 0; i < 4; i++)
                for (int j = i; j < 4; j++)
                {
                    auto &mat = rigidHess[10 * rigidBodyHandleInd + blockCnt++];
                    int iOffset = i * 3;
                    int jOffset = j * 3;
                    for (int d = 0; d < 9; ++d)
                    {
                        int di = d / 3, dj = d % 3;
                        if constexpr (Atomic)
                            atomic_add(
                                zs::exec_cuda, &mat(di, dj),
                                hess(iOffset + di, jOffset + dj));
                        else
                            mat(di, dj) +=
                                hess(iOffset + di, jOffset + dj);
                    }
                }
#if s_enableAutoPrecondition
            // preconditioner, use diag blocks
            if constexpr (!std::is_same_v<RBDataT, std::false_type>)
                if (includePre)
                {
                    for (int k = 0; k < 4; ++k)
                    {
                        int iOffset = k * 3;
                        for (int d = 0; d < 9; ++d)
                        {
                            int di = d / 3;
                            int dj = d % 3;
                            if constexpr (Atomic)
                                atomic_add(
                                    zs::exec_cuda, &rbData(RBProps::Pre, k * 9 + d, rigidBodyHandleInd),
                                    hess(iOffset + di, iOffset + dj));
                            else
                                rbData(k * 9 + d, rigidBodyHandleInd) +=
                                    hess(iOffset + di, iOffset + dj);
                        }
                    }
                }
#endif
            return;
        }

        spmat_view_t spmat;
        dyn_buffer_view_t dynHess;
        soft_vector_view_t softHess;
        rigid_vector_view_t rigidHess;
    };
    template <zs::execspace_e space, typename T_>
    static auto port(SystemHessian<T_> &hess)
    {
        return SystemHessianView<space, T_>{hess};
    }

    zs::Vector<T> temp;         // temporary buffer
    zs::Vector<vec3> temp3;     // temporary buffer
    zs::Vector<vec3> temp3_1;   // yet another temporary buffer
    zs::Vector<vec12> temp12;   // temporary buffer

    // collision matrix
    CollisionMatrix<> collisionMat;
    // contacts
    DynamicBuffer<pair_t> PP;  ///< Point-point contacts
    DynamicBuffer<pair3_t> PE; ///< Point-edge contacts
    DynamicBuffer<pair4_t> PT; ///< Point-triangle contacts
    DynamicBuffer<pair4_t> EE; ///< Edge-edge contacts
    // mollifier
    DynamicBuffer<pair4_t> PPM; ///< Point-point mollifier
    DynamicBuffer<pair4_t> PEM; ///< Point-edge mollifier
    DynamicBuffer<pair4_t> EEM; ///< Edge-edge mollifier
    // friction
    DynamicBuffer<pair_t> FPP;  ///< Point-point friction
    TVWrapper<FPPProps> fricPP; ///< Point-point friction data
    DynamicBuffer<pair3_t> FPE; ///< Point-edge friction
    TVWrapper<FPEProps> fricPE; ///< Point-edge friction data
    DynamicBuffer<pair4_t> FPT; ///< Point-triangle friction
    TVWrapper<FPTProps> fricPT; ///< Point-triangle friction data
    DynamicBuffer<pair4_t> FEE; ///< Edge-edge friction
    TVWrapper<FEEProps> fricEE; ///< Edge-edge friction data

    zs::Vector<bv_t> bvs, bvs1; // as temporary buffer, bvs1 for cases in which
                                // two buffers are needed
    DynamicBuffer<pair4_t> csPT, csEE; // collision pairs for CCD

    // Energy.cu
    /// @brief Base class for energy terms
    template <class T> struct BaseEnergy
    {
        using value_type = T;
        BaseEnergy() = default;
        virtual void update(ABDSolver &solver, pol_t &pol, bool forGrad = false)
        {
        }
        virtual T energy(ABDSolver &solver, pol_t &pol) = 0;
        virtual void addGradientAndHessian(ABDSolver &solver, pol_t &pol) = 0;
    };
    /// @brief Inertial energy term
    template <class T> struct InertialEnergy : BaseEnergy<T>
    {
        using value_type = T;
        InertialEnergy() = default;
        virtual T energy(ABDSolver &solver, pol_t &pol) override;
        virtual void addGradientAndHessian(ABDSolver &solver,
                                           pol_t &pol) override;
    };
    /// @brief Affine energy term
    template <class T> struct AffineEnergy : BaseEnergy<T>
    {
        using value_type = T;
        AffineEnergy() = default;
        virtual T energy(ABDSolver &solver, pol_t &pol) override;
        virtual void addGradientAndHessian(ABDSolver &solver,
                                           pol_t &pol) override;
    };
    /// @brief Elastic energy term
    template <class T> struct SoftEnergy : BaseEnergy<T>
    {
        using value_type = T;
        SoftEnergy() = default;
        virtual T energy(ABDSolver &solver, pol_t &pol) override;
        virtual void addGradientAndHessian(ABDSolver &solver,
                                           pol_t &pol) override;
        template <class Model>
        T elasticEnergy(ABDSolver &solver, pol_t &pol, SoftBodyHandle &softBody, const Model &model);
        template <class Model>
        void addElasticGradientAndHessian(ABDSolver &solver, pol_t &pol, SoftBodyHandle &softBody, const Model &model);
    };
    /// @brief Ground Barrier energy term
    template <class T> struct GroundBarrierEnergy : BaseEnergy<T>
    {
        using value_type = T;
        GroundBarrierEnergy() = default;
        virtual T energy(ABDSolver &solver, pol_t &pol) override;
        virtual void addGradientAndHessian(ABDSolver &solver,
                                           pol_t &pol) override;
    };
    /// @brief Barrier energy term
    template <class T> struct BarrierEnergy : BaseEnergy<T>
    {
        using value_type = T;
        BarrierEnergy() = default;
        virtual void
        update(ABDSolver &solver, pol_t &pol,
               bool forGrad = false) override; // find barrier collision pairs
        virtual T energy(ABDSolver &solver, pol_t &pol) override;
        virtual void addGradientAndHessian(ABDSolver &solver,
                                           pol_t &pol) override;
        void addBarrierGradientAndHessian(ABDSolver &solver, pol_t &pol);
        void addFrictionGradientAndHessian(ABDSolver &solver, pol_t &pol);
    };
    /// @brief Kinematic Constraint energy term

    template <class T> struct KinematicConstraintEnergy : BaseEnergy<T>
    {
        using value_type = T;
        KinematicConstraintEnergy() = default;
        virtual void
        update(ABDSolver &solver, pol_t &pol,
               bool forGrad = false) override; // compute kinematic constraints
        virtual T energy(ABDSolver &solver, pol_t &pol) override;
        virtual void addGradientAndHessian(ABDSolver &solver,
                                           pol_t &pol) override;
    };

    template <class T> struct SpringConstraintEnergy : BaseEnergy<T>
    {
        using value_type = T;
        SpringConstraintEnergy() = default;
        virtual T energy(ABDSolver &solver, pol_t &pol) override;
        virtual void addGradientAndHessian(ABDSolver &solver, pol_t &pol) override;
    };

    std::vector<std::unique_ptr<BaseEnergy<T>>> energies; ///< Energy terms
    SystemHessian<T> sysHess;                             ///< System Hessian
    // dtiles_t newtonInfo; // (dof), for gradient and optimization direction
    TVWrapper<RBProps> rbData;    ///< Rigid body data on device
    TVWrapper<VProps> vData;      ///< Vertex data on device
    TVWrapper<DOFProps> dofData;  ///< Dof data on device

    // auxiliary data (spatial acceleration)
    TVWrapper<TIProps> stInds; ///< Surface triangle indices
    TVWrapper<EIProps> seInds; ///< Surface edge indices
    TVWrapper<SVProps> svData; ///< Surface vertex indices

    // for simulated objects
    bvh_t rigidStBvh, ///< Rigid body surface triangle bvh
        rigidSeBvh;   ///< Rigid body surface edge bvh
    bvh_t softStBvh,  ///< Soft body surface triangle bvh
        softSeBvh;    ///< Soft body surface edge bvh
    size_type rigidStBvhSize = 0, rigidSeBvhSize = 0, softStBvhSize = 0,
              softSeBvhSize = 0;

    // for collision objects
    bvh_t bouStBvh,              ///< Boundary surface triangle bvh
        bouSeBvh;                ///< Boundary surface edge bvh
    std::optional<bv_t> wholeBv; ///< The whole bounding volume

    T dt,                          ///< Simulation time step
        frameDt;                   ///< Rendering frame time step
    int frameSubsteps = 1;         ///< # of substeps per frame
    bool enableCG = true;          ///< Whether to enable conjugate gradient solver
    int substeps = 0;              ///< Current # of substeps
    int frames = 0;                ///< Current frame
    T totalSimulationElapsed = 0;  ///< Simulation time elapsed
    T totalDCDElapsed = 0;         ///< DCD time elapsed
    T totalCCDElapsed = 0;         ///< CCD time elapsed
    T totalGradHessElapsed = 0;    ///< Gradient and Hessian computation time elapsed
    T totalLinearSolveElapsed = 0; ///< Newton precomputation time elapsed
    T totalEnergyElapsed = 0;      ///< Energy computation time elapsed
    T totalCPUUpdateElapsed = 0;   ///< CPU update time elapsed
    size_type layerCps = 64;       ///< Max collision layer number
    size_type estNumCps = 1000000; ///< Max collision pairs number
    size_type dynHessCps = 10000;  ///< Max dynamic hessian blocks number
    bool enableInversionPrevention = false; ///< Whether to enable inversion prevention (J barrier + CCD)
    bool enablePureRigidOpt = false; ///< Whether to enable pure rigid scene optimization
    bool enableGround = false;     ///< Whether ground is enabled
    bool enableContact = true;     ///< Whether contact is enabled
    bool enableMollification =
        true; ///< Whether mollification is enabled // NOTE: ignore for now
    bool enableContactEE = true; ///< Whether contact between edges is enabled
    bool enableFriction = true;  ///< Whether friction is enabled
    bool enableBoundaryFriction = true; ///< Whether boundary friction is enabled
    bool enableSoftBC = false;   ///< Whether softened BC is enabled
    vec3 groundNormal;           ///< Ground normal
    T kinematicALCoef = 1e4;     ///< Kinematic augmented Lagrangian coefficient
    T pnRel = 1e-2;              ///< Proximal Newton tolerance
    T cgRel = 1e-3;              ///< Conjugate gradient tolerance
    int fricIterCap = 2;         ///< Max friction iteration number
    int PNCap = 1000;            ///< Max proximal Newton iteration number
    int CGCap = 500;             ///< Max conjugate gradient iteration number
    int CCDCap = 20000;          ///< Max CCD iteration number
    T kappa0 = 1e4;              ///< Initial barrier stiffness
    bool useAbsKappaDhat = false;///< Whether to use absolute kappa, dhat
    T kappa = 1e4;               ///< Barrier stiffness
    T kappaMin = 0;              ///< Min barrier stiffness (not specified, for automatic tuning)
    T kappaMax = 0;              ///< Max barrier stiffness (not specified, for automatic tuning)
    T fricMu = 0;                ///< Friction coefficient
    T boundaryKappa = 1;         ///< Kinematic constraint stiffness
    T springStiffness = 1e3;     ///< Spring stiffness
    T abdSpringStiffness = 10.;  ///< ABD spring stiffness
    T xi = 0;
    T dHat = 2.5e-3;
    T epsv = 0.0;
    T kinematicALTol = 1e-1; // Kinematic augmented Lagrangian tolerance
    T consTol = 1e-2;
    T armijoParam = 1e-4;
    vec3 gravity = vec3{0, -9.8, 0}; ///< Gravity
    T boxDiagSize2 = 0;
    T targetGRes = 0;
    T meanEdgeLength = 0, meanSurfArea = 0, meanNodeMass = 0;
    bool updateBasis = false;
    bool kinematicSatisfied = false;
    bool bodyUpToDate = true;

    // pure ABD collision optimization
    bool pureRigidScene = false;
    zs::Vector<bv_t> rbBvs, rbRestBvs;
    zs::Vector<int> culledStInds, culledSeInds;

    // Initialize.cu
    ABDSolver(const std::vector<BodySP<T, tvLength>> &bodies, 
              const CollisionMatrix<> &collisionMat,
              T dt,
              int frameSubsteps, bool enableCG, bool enableInversionPrevention, bool enablePureRigidOpt, bool enableGround, bool enableContact,
              bool enableMollification, bool enableContactEE,
              bool enableFriction, bool enableBoundaryFriction, bool enableSoftBC,
              std::size_t layerCps,
              std::size_t estNumCps, std::size_t dynHessCps,
              T kinematicALCoef, // bool enableAL,
              T pnRel, T cgRel, int fricIterCap, int PNCap, int CGCap,
              int CCDCap, T kappa0, bool useAbsKappaDhat, T fricMu, T boundaryKappa, T springStiffness, T abdSpringStiffness, T xi, T dHat,
              T epsv, T kinematicALTol, T consTol, T armijoParam,
              vec3 groundNormal, T gravity);
    // ignore for now
    void suggestKappa(pol_t &pol); // TODO: true implementation of dynamic kappa
    void initKappa(pol_t &pol);
    T largestMu() const {
        return 1e6; // TODO: which model is specified here? cloth only?
        // T mu = 0;
        // for (auto &&primHandle : prims) {
        //     auto [m, l] = primHandle.getModelLameParams();
        //     if (m > mu)
        //         mu = m;
        // }
        // return mu;
    }
    void initialize(pol_t &pol);
    void resetProfiler();
    T averageSurfEdgeLength(pol_t &pol);
    T averageSurfArea(pol_t &pol);
    T averageNodeMass(pol_t &pol);
    void updateWholeBoundingBox(pol_t &pol);
    void stepInitialize(pol_t &pol);
    void substepInitialize(pol_t &pol);
    // Pipeline.cu
    void step(pol_t &pol, bool updateBody = false);
    void step(pol_t &pol, T frameDt, bool updateBody = false);
    void step(pol_t &pol, int frameSubsteps, bool updateBody = false);
    void step(pol_t &pol, T frameDt, int frameSubsteps, 
              bool updateBody = false);
    void substep(pol_t &pol);
    bool newtonKrylov(pol_t &pol);
    T kinematicConstraintResidual(pol_t &pol);
    void updateKinematicConstraintsLambda(pol_t &pol);
    void computeKinematicConstraints(pol_t &pol);
    void updateEnergy(pol_t &pol, bool forGrad = false);
    T energy(pol_t &pol);
    void addGradientAndHessian(pol_t &pol);
    T prepareLineSearch(pol_t &pol); /// prepare initial step size, alpha
    T lineSearch(pol_t &pol, T alpha = 1.);
    void updateVelocities(
        pol_t &pol);         /// @brief Update velocities with new positions
    void update(pol_t &pol); /// @brief Update data to bodies
    // LinearSolve.cu
    void cgSolve(pol_t &pol);
    void precondition(pol_t &pol);
    void multiplyHessian(pol_t &pol);
    void directSolve(pol_t &pol);

    // Collision.cu
    auto getCnts() const
    {
        return zs::make_tuple(PP.getCount(), PE.getCount(), PT.getCount(),
                              EE.getCount(), PPM.getCount(), PEM.getCount(),
                              EEM.getCount(), csPT.getCount(), csEE.getCount());
    }
    auto getCollisionCnts() const
    {
        return zs::make_tuple(csPT.getCount(), csEE.getCount());
    }
    void findBarrierCollisions(pol_t &pol, T xi = 0);
    void findBarrierCollisionsImpl(pol_t &pol, T xi = 0,
                                   bool withBoundary = false);
    void findCCDCollisions(pol_t &pol, T alpha, T xi = 0);
    void findCCDCollisionsImpl(pol_t &pol, T alpha, T xi = 0,
                               bool withBoundary = false);
    // void findInversionCCDCollisionsImpl(pol_t &pol, T alpha, T xi = 0);
    void precomputeFrictions(pol_t &pol, T xi = 0);
    void precomputeBoundaryFrictions(pol_t &pol, T activeGap2 = 0);
    T groundCCD(pol_t &pol, T alpha);
    T CCD(pol_t &pol, T xi, T alpha);
    T ACCD(pol_t &pol, T xi, T alpha);
    T inversionPreventCCD(pol_t &pol, T alpha);
};
} // namespace tacipc