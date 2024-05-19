#pragma once
#include <variant>
#include <tacipc/meta.hpp>
#include <tacipc/utils.cuh>
#include <tacipc/meshio/mesh.hpp>
#include <tacipc/meshio/writeMesh.hpp>
#include <tacipc/constitutiveModel/constitutiveModel.cuh>
// #include <tacipc/solver/VecWrapper.cuh>
// #include <tacipc/solver/VWrapper.cuh>
#include <tacipc/solver/TVWrapper.cuh>
#include <tacipc/serialization.cuh>
#include <zensim/container/TileVector.hpp>
#include <zensim/omp/execution/ExecutionPolicy.hpp>

namespace tacipc
{
/// @brief body classes declaration
template <enum BodyType BodyType_v, class T, std::size_t TvLen = 32, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> struct TriangleBody;
template <enum BodyType BodyType_v, class T, std::size_t TvLen = 32, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> struct TetBody;

template <class T, std::size_t TvLen = 32>
using RigidTriangleBody = TriangleBody<BodyType::Rigid, T, TvLen>;
template <class T, std::size_t TvLen = 32>
using RigidTetBody = TetBody<BodyType::Rigid, T, TvLen>;
template <class T, std::size_t TvLen = 32>
using SoftTriangleBody = TriangleBody<BodyType::Soft, T, TvLen>;
template <class T, std::size_t TvLen = 32>
using SoftTetBody = TetBody<BodyType::Soft, T, TvLen>;

template <class T, std::size_t TvLen = 32>
using RigidBody = std::variant<RigidTriangleBody<T, TvLen>, RigidTetBody<T, TvLen>>;
template <class T, std::size_t TvLen = 32>
using SoftBody = std::variant<SoftTriangleBody<T, TvLen>, SoftTetBody<T, TvLen>>;
template <class T, std::size_t TvLen = 32>
using RigidBodySP = typename variant_wrap<RigidBody<T, TvLen>, std::shared_ptr>::type;
template <class T, std::size_t TvLen = 32>
using SoftBodySP = typename variant_wrap<SoftBody<T, TvLen>, std::shared_ptr>::type;
template <class T, std::size_t TvLen = 32>
using Body = typename variant_cat<RigidBody<T, TvLen>, SoftBody<T, TvLen>>::type;
template <class T, std::size_t TvLen = 32>
using BodySP = typename variant_cat<RigidBodySP<T, TvLen>, SoftBodySP<T, TvLen>>::type;

/// @brief check if a type is a body
/// @tparam T 
template <class T>
struct is_triangle_body {
    static constexpr bool value = false;
};
template <enum BodyType BodyType_v, class T, std::size_t TvLen>
struct is_triangle_body<TriangleBody<BodyType_v, T, TvLen>> {
    static constexpr bool value = true;
};
template <class T>
inline constexpr bool is_triangle_body_v = is_triangle_body<T>::value;
template <class T>
struct is_tet_body {
    static constexpr bool value = false;
};
template <enum BodyType BodyType_v, class T, std::size_t TvLen>
struct is_tet_body<TetBody<BodyType_v, T, TvLen>> {
    static constexpr bool value = true;
};
template <class T>
inline constexpr bool is_tet_body_v = is_tet_body<T>::value;
template <typename T>
inline constexpr bool is_body_v = is_triangle_body_v<T> || is_tet_body_v<T>;

template <class T>
struct is_rigid_body {
    static constexpr bool value = false;
};
template <class t, std::size_t TvLen>
struct is_rigid_body<RigidTriangleBody<t, TvLen>> {
    static constexpr bool value = true;
};
template <class t, std::size_t TvLen>
struct is_rigid_body<RigidTetBody<t, TvLen>> {
    static constexpr bool value = true;
};
template <class T>
inline constexpr bool is_rigid_body_v = is_rigid_body<T>::value;
template <class T>
inline constexpr bool is_rigid_body_sp_v = false;
template <class T>
inline constexpr bool is_rigid_body_sp_v<std::shared_ptr<T>> = is_rigid_body_v<T>;
template <class T>
struct is_soft_body {
    static constexpr bool value = false;
};
template <class t, std::size_t TvLen>
struct is_soft_body<SoftTriangleBody<t, TvLen>> {
    static constexpr bool value = true;
};
template <class t, std::size_t TvLen>
struct is_soft_body<SoftTetBody<t, TvLen>> {
    static constexpr bool value = true;
};
template <class T>
inline constexpr bool is_soft_body_v = is_soft_body<T>::value;
template <class T>
inline constexpr bool is_soft_body_sp_v = false;
template <class T>
inline constexpr bool is_soft_body_sp_v<std::shared_ptr<T>> = is_soft_body_v<T>;


/// @brief body properties classes declaration
/// @tparam T
/// @tparam TvLen
template <class T, std::size_t TvLen = 32, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> struct RigidBodyProperties;
template <class T, std::size_t TvLen = 32, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> struct SoftBodyProperties;

template <class T>
struct is_rigid_body_properties {
    static constexpr bool value = false;
};
template <class T, std::size_t TvLen>
struct is_rigid_body_properties<RigidBodyProperties<T, TvLen>> {
    static constexpr bool value = true;
};
template <class T>  
constexpr bool is_rigid_body_properties_v = is_rigid_body_properties<T>::value;

template <class T>
struct is_soft_body_properties {
    static constexpr bool value = false;
};
template <class T, std::size_t TvLen>
struct is_soft_body_properties<SoftBodyProperties<T, TvLen>> {
    static constexpr bool value = true;
};
template <class T>
constexpr bool is_soft_body_properties_v = is_soft_body_properties<T>::value;

template <class T>
constexpr bool is_body_type_properties_v = is_rigid_body_properties<T>::value || is_soft_body_properties<T>::value;

using BodyTypeProperties = variant_wrap_floating_point<RigidBodyProperties, SoftBodyProperties>::type;

template <class TriBodyT, std::enable_if_t<is_triangle_body_v<TriBodyT>, int> = 0>
void to_json(json &j, TriBodyT const &b);
template <class TetBodyT, std::enable_if_t<is_tet_body_v<TetBodyT>, int> = 0>
void to_json(json &j, TetBodyT const &b);
template <class RBPropsT, std::enable_if_t<is_rigid_body_properties_v<RBPropsT>, int> = 0>
void to_json(json &j, RBPropsT const &p);
template <class SBPropsT, std::enable_if_t<is_soft_body_properties_v<SBPropsT>, int> = 0>
void to_json(json &j, SBPropsT const &p);

/// @brief 
/// @tparam bodyType
/// @tparam T 
/// @tparam TvLen 
template <enum BodyType BodyType_v, class T, std::size_t TvLen, std::enable_if_t<std::is_floating_point_v<T>, int>> struct TriangleBody
{
    static constexpr std::size_t codim = 2;
    static constexpr BodyType bodyType = BodyType_v;
    static constexpr std::size_t tvLength = TvLen;
    using size_type = std::size_t;
    using value_t = T;
    using ind_t = zs::conditional_t<zs::is_same_v<T, double>, zs::i64, zs::i32>;
    using tiles_t = zs::TileVector<T, TvLen>;
    using vec3 = zs::vec<T, 3>;
    using mat2 = zs::vec<T, 2, 2>;
    using vec2i = zs::vec<int, 2>;
    using vec3i = zs::vec<int, 3>;
    using body_t = TriangleBody<BodyType_v, T, TvLen>;
    using bodyTypeProperties_t = std::conditional_t<bodyType == BodyType::Rigid, RigidBodyProperties<T, TvLen>, SoftBodyProperties<T, TvLen>>;

    std::string name = "";
    std::size_t id = -1; // id in simulation
    std::size_t layer = 0; // (collision) layer in simulation
    tiles_t verts;
    zs::Vector<int> vertInds;
    zs::Vector<vec2i> edgeInds;
    zs::Vector<vec3i> triInds;
    tiles_t tris;
    eigen::matX3d meshVerts;
    eigen::matX3i meshTris;
    bool solid;
    T thickness;
    bodyTypeProperties_t bodyTypeProperties;

    TriangleBody(std::string name, tiles_t const &verts, zs::Vector<int> const&vertInds, zs::Vector<vec2i> const &edgeInds, zs::Vector<vec3i> const &triInds, tiles_t const &tris, eigen::matX3d const  &meshVerts, eigen::matX3i const &meshTris, bool solid = false, T thickness = 1)
        : name{name}, verts{verts}, vertInds{vertInds}, edgeInds{edgeInds}, triInds{triInds}, tris{tris}, meshVerts{meshVerts}, meshTris{meshTris}, solid{solid}, thickness{thickness}, bodyTypeProperties{*this} 
    {}
    TriangleBody(std::string name, tiles_t &&verts, zs::Vector<int> &&vertInds, zs::Vector<vec2i> &&edgeInds, zs::Vector<vec3i> &&triInds, tiles_t &&tris, eigen::matX3d &&meshVerts, eigen::matX3i &&meshTris, bool solid = false, T thickness = 1)
        : name{name}, verts{verts}, vertInds{vertInds}, edgeInds{edgeInds}, triInds{triInds}, tris{tris}, meshVerts{meshVerts}, meshTris{meshTris}, solid{solid}, thickness{thickness}, bodyTypeProperties{*this}
    {}
    void setDensity(T density)
    {
        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();
        auto memloc = verts.memoryLocation();
        moveTo({memsrc_e::host, -1});
        pol(range(nTris()),
            [tris = view<space>(tris),
             mTag = tris.getPropertyOffset("m"),
             volTag = tris.getPropertyOffset("vol"),
             density](int ti) mutable
            {
                tris(mTag, ti) = tris(volTag, ti) * density;
            });
        pol(range(nVerts()),
            [verts = view<space>(verts),
             mTag = verts.getPropertyOffset("m"),
             volTag = verts.getPropertyOffset("vol"),
             density](int vi) mutable
            {
                verts(mTag, vi) = verts(volTag, vi) * density;
            });
        bodyTypeProperties.setDensity(density);
        moveTo(memloc);
    }
    void moveTo(zs::MemoryLocation const &mloc);
    size_type nVerts() const { return verts.size(); }
    size_type nEdges() const { return edgeInds.size(); }
    size_type nTris() const { return tris.size(); }
    void updateMesh()
    {
        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

        auto h_verts = verts.clone({memsrc_e::host, -1});
        auto h_tris = triInds.clone({memsrc_e::host, -1});

        meshVerts.resize(nVerts(), 3);
        meshTris.resize(nTris(), 3);
        pol(range(nVerts()),
            [h_verts = view<space>({}, h_verts),
             xTag = h_verts.getPropertyOffset("x"),
             &meshVerts = this->meshVerts](int vi) mutable
            {
                auto const &x = h_verts.pack(dim_c<3>, xTag, vi);
                meshVerts.row(vi) = eigen::vec3d{(double)x[0], (double)x[1], (double)x[2]};
            });
        pol(range(nTris()),
            [h_tris = view<space>(h_tris),
             &meshTris = this->meshTris](int ti) mutable
            {
                auto const &t = h_tris(ti);
                meshTris.row(ti) = eigen::vec3i{t[0], t[1], t[2]};
            });
    }
    auto& getBodyTypeProperties();
    auto& getBodyTypeProperties() const;
    auto& getModel() { return getBodyTypeProperties().getModel(); }
    auto& getModel() const { return getBodyTypeProperties().getModel(); }
    void save(std::string filename, Eigen::Matrix4d const &transform=Eigen::Matrix4d::Identity()) const 
    {
        auto basename = filename.substr(0, filename.find_last_of('.'));
        save_mesh(basename + ".obj", transform);
        save_json(basename + ".json");
    }
    void save_mesh(std::string const &filename, Eigen::Matrix4d const &transform=Eigen::Matrix4d::Identity()) const
    {
        auto meshVerts_ = meshVerts;
        if (transform != Eigen::Matrix4d::Identity())
        {
            meshVerts_ = (meshVerts_ * transform.block<3, 3>(0, 0).transpose()).rowwise() + transform.block<3, 1>(0, 3).transpose();
        }
        meshio::TriMesh mesh{meshVerts_, meshTris};
        meshio::writeObjFile(filename, mesh);
    }
    void save_json(std::string const &filename) const
    {
        json j;
        to_json(j, *this);
        std::ofstream o(filename);
        o << j.dump(4) << std::endl;
        o.close();
    }
};

template <enum BodyType BodyType_v, class T, std::size_t TvLen, std::enable_if_t<std::is_floating_point_v<T>, int>> struct TetBody
{
    static constexpr std::size_t codim = 3;
    static constexpr BodyType bodyType = BodyType_v;
    static constexpr std::size_t tvLength = TvLen;
    using size_type = std::size_t;
    using value_t = T;
    using ind_t = zs::conditional_t<zs::is_same_v<T, double>, zs::i64, zs::i32>;
    using tiles_t = zs::TileVector<T, TvLen>;
    using vec3 = zs::vec<T, 3>;
    using vec4 = zs::vec<T, 4>;
    using mat3 = zs::vec<T, 3, 3>;
    using vec2i = zs::vec<int, 2>;
    using vec3i = zs::vec<int, 3>;
    using vec4i = zs::vec<int, 4>;
    using body_t = TetBody<BodyType_v, T, TvLen>;
    using bodyTypeProperties_t = std::conditional_t<bodyType == BodyType::Rigid, RigidBodyProperties<T, TvLen>, SoftBodyProperties<T, TvLen>>;

    std::string name = "";
    std::size_t id = -1; // id in simulation
    std::size_t layer = 0; // (collision) layer in simulation
    tiles_t verts;
    zs::Vector<int> surfVertInds;
    zs::Vector<vec2i> surfEdgeInds;
    zs::Vector<vec3i> surfTriInds;
    tiles_t tets;
    eigen::matX3d meshVerts;    /// make sure to call updateMesh() before using
    eigen::matX3i meshTris;
    eigen::matX4i meshTets;
    bodyTypeProperties_t bodyTypeProperties;

    TetBody(std::string name, tiles_t const &verts, zs::Vector<int> const &surfVertInds, zs::Vector<vec2i> const &surfEdgeInds, zs::Vector<vec3i> const &surfTriInds, tiles_t const &tets, eigen::matX3d const &meshVerts, eigen::matX3i const &meshTris, eigen::matX4i const &meshTets)
        : name{name}, verts{verts}, surfVertInds{surfVertInds}, surfEdgeInds{surfEdgeInds}, surfTriInds{surfTriInds}, tets{tets}, meshVerts{meshVerts}, meshTris{meshTris}, meshTets{meshTets}, bodyTypeProperties{*this}
    {}
    TetBody(std::string name, tiles_t &&verts, zs::Vector<int> &&surfVertInds, zs::Vector<vec2i> &&surfEdgeInds, zs::Vector<vec3i> &&surfTriInds, tiles_t &&tets, eigen::matX3d &&meshVerts, eigen::matX3i &&meshTris, eigen::matX4i &&meshTets)
        : name{name}, verts{verts}, surfVertInds{surfVertInds}, surfEdgeInds{surfEdgeInds}, surfTriInds{surfTriInds}, tets{tets}, meshVerts{meshVerts}, meshTris{meshTris}, meshTets{meshTets}, bodyTypeProperties{*this}
    {}
    void setDensity(T density)
    {
        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();
        auto memloc = verts.memoryLocation();
        moveTo({memsrc_e::host, -1});
        pol(range(nTets()),
            [tets = view<space>(tets),
             mTag = tets.getPropertyOffset("m"),
             volTag = tets.getPropertyOffset("vol"),
             density](int ti) mutable
            {
                tets(mTag, ti) = tets(volTag, ti) * density;
            });
        pol(range(nVerts()),
            [verts = view<space>(verts),
             mTag = verts.getPropertyOffset("m"),
             volTag = verts.getPropertyOffset("vol"),
             density](int vi) mutable
            {
                verts(mTag, vi) = verts(volTag, vi) * density;
            });
        bodyTypeProperties.setDensity(density);
        moveTo(memloc);
    }
    void moveTo(zs::MemoryLocation const &mloc);
    size_type nVerts() const { return verts.size(); }
    size_type nSurfVerts() const { return surfVertInds.size(); }
    size_type nSurfEdges() const { return surfEdgeInds.size(); }
    size_type nSurfTris() const { return surfTriInds.size(); }
    size_type nTets() const { return tets.size(); }
    void updateMesh() /// update mesh data from body data, shall be called MANUALLY
    {
        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

        auto h_verts = verts.clone({memsrc_e::host, -1});
        auto h_tris = surfTriInds.clone({memsrc_e::host, -1});
        auto h_tets = tets.clone({memsrc_e::host, -1});

        meshVerts.resize(nVerts(), 3);
        meshTris.resize(nSurfTris(), 3);
        meshTets.resize(nTets(), 4);
        pol(range(nVerts()),
            [h_verts = view<space>({}, h_verts),
             xTag = h_verts.getPropertyOffset("x"),
             &meshVerts = this->meshVerts](int vi) mutable
            {
                auto const &x = h_verts.pack(dim_c<3>, xTag, vi);
                meshVerts.row(vi) = eigen::vec3d{x[0], x[1], x[2]};
            });
        pol(range(nSurfTris()),
            [h_tris = view<space>(h_tris),
             &meshTris = this->meshTris](int ti) mutable
            {
                auto const &t = h_tris(ti);
                meshTris.row(ti) = eigen::vec3i{t[0], t[1], t[2]};
            });
        pol(range(nTets()),
            [h_tets = view<space>(h_tets),
             &meshTets = this->meshTets,
             indsTag = h_tets.getPropertyOffset("inds")](int ti) mutable
            {
                auto const &t = h_tets.pack(dim_c<4>, indsTag, ti).template reinterpret_bits<ind_t>();
                meshTets.row(ti) = eigen::vec4i{t[0], t[1], t[2], t[3]};
            });
    }
    auto& getBodyTypeProperties();
    auto& getBodyTypeProperties() const;
    auto& getModel() { return getBodyTypeProperties().getModel(); }
    auto& getModel() const { return getBodyTypeProperties().getModel(); }
    template <bool MSH=true>
    void save(std::string filename, Eigen::Matrix4d const &transform=Eigen::Matrix4d::Identity()) const 
    {
        auto basename = filename.substr(0, filename.find_last_of('.'));
        if constexpr (MSH)
            save_mesh<MSH>(basename + ".msh", transform);
        else
            save_mesh<MSH>(basename + ".obj", transform);
        save_json(basename + ".json");
    }
    template <bool MSH=true>
    void save_mesh(std::string filename, Eigen::Matrix4d const &transform=Eigen::Matrix4d::Identity()) const 
    {
        auto meshVerts_ = meshVerts;
        fmt::print("preparing mesh to save...\n");
        if (transform != Eigen::Matrix4d::Identity())
        {
            meshVerts_ = (meshVerts_ * transform.block<3, 3>(0, 0).transpose()).rowwise() + transform.block<3, 1>(0, 3).transpose();
        }

        fmt::print("saving to {}...\n", filename);
        if constexpr (MSH)
        {
            meshio::TetMesh mesh{meshVerts_, meshTets};
            meshio::writeMshFile(filename, mesh);
        }
        else
        {
            meshio::TriMesh mesh{meshVerts_, meshTris};
            meshio::writeObjFile(filename, mesh);   
        }
    }
    void save_json(std::string const &filename) const
    {
        json j;
        to_json(j, *this);
        std::ofstream o(filename);
        o << j.dump(4) << std::endl;
        o.close();
    }
};

template <class ...PtrTs>
constexpr int bodyId(const std::variant<PtrTs...> &ptr) { 
    return std::visit([](auto &ptr) { 
        static_assert(is_body_v<RM_CVREF_T(*ptr)>, "getting body id for non-body pointer");
        return ptr->id; 
    }, ptr); 
}

template <class TriBodyT, bool solid = false, std::enable_if_t<is_triangle_body_v<TriBodyT>, int> = 0>
std::shared_ptr<TriBodyT> genTriBodyFromTriMesh(std::string name, const meshio::TriMesh &mesh, typename TriBodyT::value_t density, typename TriBodyT::value_t thickness = 1., Eigen::Matrix4d const &transform = Eigen::Matrix4d::Identity())
{
    tacipc_assert((!solid || is_rigid_body_v<TriBodyT>), "solid triangle body for non-rigid body is not supported.");
    using value_t = typename TriBodyT::value_t;
    using ind_t = typename TriBodyT::ind_t;
    using tiles_t = typename TriBodyT::tiles_t;
    // using tiles_t = TriangleBody<float>::tiles_t; // for IDE
    using value_t = typename TriBodyT::value_t;
    using vec3 = typename TriBodyT::vec3;
    using vec2i = typename TriBodyT::vec2i;
    using vec3i = typename TriBodyT::vec3i;
    using mat2 = typename TriBodyT::mat2;
    using namespace zs;

    std::vector<PropertyTag> vertTags{{"m", 1}, // to be set by other APIs // mass
                                      {"vol", 1},
                                      {"x", 3}, // position
                                      {"x0", 3}, // initial position
                                      {"v", 3},
                                      {"contact", 3}}; // to be set by other APIs // velocity
    std::vector<PropertyTag> triTags{{"vol", 1}, // volume
                                     {"area", 1}, // area
                                     {"IB", 4}, // the Inverse of its Basis
                                     {"inds", 3}, // indices of vertices
                                     {"m", 1}}; // to be set by other APIs // mass

    tiles_t verts{vertTags, mesh.nVerts(), memsrc_e::host};
    zs::Vector<int> vertInds{mesh.nVerts(), memsrc_e::host};
    zs::Vector<vec2i> edgeInds{mesh.nEdges(), memsrc_e::host};
    zs::Vector<vec3i> triInds{mesh.nTris(), memsrc_e::host};
    tiles_t tris{triTags, mesh.nTris(), memsrc_e::host};

    constexpr auto space = execspace_e::openmp;
    auto ompExec = omp_exec();

    auto meshVerts = applyTransform(mesh.getVerts(), transform);
    ompExec(range(verts.size()),
            [verts = view<space>(verts, false_c, "verts"), 
             vertInds = view<space>(vertInds, false_c, "vertInds"),
             xTag = verts.getPropertyOffset("x"),
             x0Tag = verts.getPropertyOffset("x0"),
             vTag = verts.getPropertyOffset("v"),
             mTag = verts.getPropertyOffset("m"), 
             volTag = verts.getPropertyOffset("vol"),
             contactTag = verts.getPropertyOffset("contact"),
             &meshVerts](int tid) mutable
            {
                vertInds(tid) = tid;
                auto const &x = meshVerts.row(tid);
                vec3 xVec{x[0], x[1], x[2]};
                verts.tuple(dim_c<3>, xTag, tid) = xVec;
                verts.tuple(dim_c<3>, x0Tag, tid) = xVec;
                verts.tuple(dim_c<3>, vTag, tid) = vec3::zeros();
                verts(mTag, tid) = 0;
                verts(volTag, tid) = 0;
                verts.tuple(dim_c<3>, contactTag, tid) = vec3::zeros();
            });

    auto const &meshEdges = mesh.getEdges();
    ompExec(range(edgeInds.size()),
            [edgeInds = view<space>(edgeInds, false_c, "edgeInds"), &meshEdges](int tid) mutable
            {
                auto const &e = meshEdges.row(tid);
                edgeInds[tid] = vec2i{e[0], e[1]};
            });

    if (mesh.nTris() == 0) // experimental
    {
        if (mesh.nEdges() == 0) // particles
        {
            // fmt::print("creating experimental particle object...\n");
            ompExec(range(verts.size()),
                    [verts = view<space>(verts, false_c, "verts"), 
                    mTag = verts.getPropertyOffset("m"), 
                    volTag = verts.getPropertyOffset("vol"),
                    density,
                    &meshVerts](int tid) mutable
                    {
                        verts(mTag, tid) = density;
                        verts(volTag, tid) = density;
                    });
        }
        else
        {
            tacipc_assert(false, "line objects not supported now.");   
        }
    }
    else 
    {
        if (solid) // without IB, seems unnecessary
        {
            auto const &meshTris = mesh.getTris();
            ompExec(range(tris.size()),
                    [tris = view<space>({}, tris, false_c, "tris"), verts = view<space>({}, verts, false_c, "verts"),
                    triInds = view<space>(triInds, false_c, "triInds"),
                    volTag = tris.getPropertyOffset("vol"),
                    mTag = tris.getPropertyOffset("m"),
                    areaTag = tris.getPropertyOffset("area"),
                    indsTag = tris.getPropertyOffset("inds"),
                    ibTag = tris.getPropertyOffset("IB"),
                    x0Tag = verts.getPropertyOffset("x0"),
                    vmTag = verts.getPropertyOffset("m"), 
                    vvolTag = verts.getPropertyOffset("vol"),
                    &meshTris,
                    density](int tid) mutable
                    {
                        auto const &t = meshTris.row(tid);
                        triInds(tid) = vec3i{t[0], t[1], t[2]};
                        tris.tuple(dim_c<3>, indsTag, tid) = vec3{reinterpret_bits<value_t>((ind_t)t[0]), reinterpret_bits<value_t>((ind_t)t[1]), reinterpret_bits<value_t>((ind_t)t[2])};
                        // NOTE: currently, use the non-SVD way to compute IB
                        vec3 xs[3];
                        for (int d = 0; d != 3; ++d)
                            xs[d] = verts.pack(dim_c<3>, x0Tag, t[d]);
                        vec3 ds[2] = {xs[1] - xs[0], xs[2] - xs[0]};
                        auto cross = ds[0].cross(ds[1]);
                        auto area = cross.norm() * static_cast<value_t>(0.5);
                        auto vol = xs[0].dot(cross) / static_cast<value_t>(6); // vol of (xs0, xs1, xs2, (0,0,0))
                        tris(areaTag, tid) = area;
                        tris(volTag, tid) = vol;
                        tris(mTag, tid) = vol * density;
                        for (int d = 0; d != 3; ++d)
                        {
                            atomic_add(exec_omp, &verts(vmTag, t[d]),
                                    vol * density / static_cast<value_t>(4)); // maybe 4, but 1/4 of the mass gets lost (at (0,0,0))
                            atomic_add(exec_omp, &verts(vvolTag, t[d]),
                                    vol / static_cast<value_t>(4)); // maybe 4, but 1/4 of the mass gets lost (at (0,0,0))
                        }
                    });

        }
        else // non-solid, i.e., shell
        {
            auto const &meshTris = mesh.getTris();
            ompExec(range(tris.size()),
                    [tris = view<space>(tris, false_c, "tris"), verts = view<space>(verts, false_c, "verts"),
                    triInds = view<space>(triInds, false_c, "triInds"),
                    volTag = tris.getPropertyOffset("vol"),
                    mTag = tris.getPropertyOffset("m"),
                    areaTag = tris.getPropertyOffset("area"),
                    indsTag = tris.getPropertyOffset("inds"),
                    ibTag = tris.getPropertyOffset("IB"),
                    x0Tag = verts.getPropertyOffset("x0"),
                    vmTag = verts.getPropertyOffset("m"), 
                    vvolTag = verts.getPropertyOffset("vol"),
                    &meshTris,
                    density, thickness](int tid) mutable
                    {
                        auto const &t = meshTris.row(tid);
                        triInds(tid) = vec3i{t[0], t[1], t[2]};
                        tris.tuple(dim_c<3>, indsTag, tid) = vec3{reinterpret_bits<value_t>((ind_t)t[0]), reinterpret_bits<value_t>((ind_t)t[1]), reinterpret_bits<value_t>((ind_t)t[2])};
                        // NOTE: currently, use the non-SVD way to compute IB
                        vec3 xs[3];
                        for (int d = 0; d != 3; ++d)
                            xs[d] = verts.pack(dim_c<3>, x0Tag, t[d]);
                        vec3 ds[2] = {xs[1] - xs[0], xs[2] - xs[0]};
                        mat2 B{};
                        B(0, 0) = ds[0].norm();
                        B(1, 0) = 0;
                        B(0, 1) = ds[0].dot(ds[1]) / B(0, 0);
                        B(1, 1) = ds[0].cross(ds[1]).norm() / B(0, 0);
                        auto IB = inverse(B);
                        if (std::isnan(IB(0, 0)) || std::isnan(IB(0, 1)) ||
                            std::isnan(IB(1, 0)) || std::isnan(IB(1, 1)))
                            IB = mat2::zeros();
                        tris.tuple(dim_c<4>, ibTag, tid) = IB;

                        auto area = ds[0].cross(ds[1]).norm() * static_cast<value_t>(0.5);
                        tris(areaTag, tid)  = area;
                        tris(volTag, tid)   = area * thickness;
                        tris(mTag, tid)     = area * thickness * density; 
                        for (int d = 0; d != 3; ++d)
                        {
                            atomic_add(exec_omp, &verts(vmTag, t[d]),
                                    area * thickness * density / static_cast<value_t>(3));
                            atomic_add(exec_omp, &verts(vvolTag, t[d]),
                                    area * thickness / static_cast<value_t>(3));
                        }
                    });
        }
    }

    auto bodyPtr = std::make_shared<TriBodyT>(name, verts, vertInds, edgeInds, triInds, tris, meshVerts, mesh.getTris(), solid);
    // fmt::print("body generated.\n");
    bodyPtr->moveTo({memsrc_e::device, 0});
    return bodyPtr;
}
template <class TriBodyT, bool solid = false, std::enable_if_t<is_triangle_body_v<TriBodyT>, int> = 0>
std::shared_ptr<TriBodyT> genTriBodyFromTriMesh(const meshio::TriMesh &mesh, typename TriBodyT::value_t density, typename TriBodyT::value_t thickness = 1., Eigen::Matrix4d const &transform = Eigen::Matrix4d::Identity())
{
    return genTriBodyFromTriMesh<TriBodyT, solid>("", mesh, density, thickness, transform);
}

template <class TriBodyT, bool solid = false, std::enable_if_t<is_triangle_body_v<TriBodyT>, int> = 0>
std::shared_ptr<TriBodyT> genTriBodyFromTriMesh(const meshio::TriMesh &mesh, typename TriBodyT::value_t density, typename TriBodyT::value_t thickness, zs::vec<typename TriBodyT::value_t, 4, 4> const &transform)
{
    return genTriBodyFromTriMesh<TriBodyT, solid>(mesh, density, thickness, vec2Eigen(transform));
}

template <class TriBodyT, bool solid = false, std::enable_if_t<is_triangle_body_v<TriBodyT>, int> = 0>
std::shared_ptr<TriBodyT> genTriBodyFromTriMesh(std::string name, const meshio::TriMesh &mesh, typename TriBodyT::value_t density, typename TriBodyT::value_t thickness, zs::vec<typename TriBodyT::value_t, 4, 4> const &transform)
{
    return genTriBodyFromTriMesh<TriBodyT, solid>(name, mesh, density, thickness, vec2Eigen(transform));
}
template <class TriBodyT, bool solid = false, std::enable_if_t<is_triangle_body_v<TriBodyT>, int> = 0>
std::shared_ptr<TriBodyT> genTriBodyFromTriMesh(std::string name, const meshio::TriMesh &mesh, typename TriBodyT::value_t density, Eigen::Matrix4d const &transform = Eigen::Matrix4d::Identity())
{
    return genTriBodyFromTriMesh<TriBodyT, solid>(name, mesh, density, 1., transform);
}
template <class TriBodyT, bool solid = false, std::enable_if_t<is_triangle_body_v<TriBodyT>, int> = 0>
std::shared_ptr<TriBodyT> genTriBodyFromTriMesh(const meshio::TriMesh &mesh, typename TriBodyT::value_t density, Eigen::Matrix4d const &transform = Eigen::Matrix4d::Identity())
{
    return genTriBodyFromTriMesh<TriBodyT, solid>("", mesh, density, transform);
}

template <class TriBodyT, bool solid = false, std::enable_if_t<is_triangle_body_v<TriBodyT>, int> = 0>
std::shared_ptr<TriBodyT> genTriBodyFromTriMesh(const meshio::TriMesh &mesh, typename TriBodyT::value_t density, zs::vec<typename TriBodyT::value_t, 4, 4> const &transform)
{
    return genTriBodyFromTriMesh<TriBodyT, solid>(mesh, density, vec2Eigen(transform));
}

template <class TriBodyT, bool solid = false, std::enable_if_t<is_triangle_body_v<TriBodyT>, int> = 0>
std::shared_ptr<TriBodyT> genTriBodyFromTriMesh(std::string name, const meshio::TriMesh &mesh, typename TriBodyT::value_t density, zs::vec<typename TriBodyT::value_t, 4, 4> const &transform)
{
    return genTriBodyFromTriMesh<TriBodyT, solid>(name, mesh, density, vec2Eigen(transform));
}

template <class TetBodyT, std::enable_if_t<is_tet_body_v<TetBodyT>, int> = 0>
std::shared_ptr<TetBodyT> genTetBodyFromTetMesh(std::string name, const meshio::TetMesh &mesh, typename TetBodyT::value_t density, Eigen::Matrix4d const &transform = Eigen::Matrix4d::Identity())
{
    using tiles_t = typename TetBodyT::tiles_t;
    using value_t = typename TetBodyT::value_t;
    using ind_t = typename TetBodyT::ind_t;
    using vec3 = typename TetBodyT::vec3;
    using vec4 = typename TetBodyT::vec4;
    using vec2i = typename TetBodyT::vec2i;
    using vec3i = typename TetBodyT::vec3i;
    using vec4i = typename TetBodyT::vec4i;
    using mat3 = typename TetBodyT::mat3;
    using namespace zs;

    std::vector<PropertyTag> vertTags{{"m", 1}, // to be set by other APIs
                                      {"vol", 1},
                                      {"x", 3},
                                      {"x0", 3},
                                      {"v", 3},
                                      {"contact", 3}}; // to be set by other APIs
    std::vector<PropertyTag> tetTags{{"vol", 1},
                                     {"IB", 9}, // the Inverse of its Basis
                                     {"inds", 4},
                                     {"m", 1}}; // to be set by other APIs

    tiles_t verts{vertTags, mesh.nVerts(), memsrc_e::host};
    zs::Vector<int> surfVertInds{mesh.nSurfVerts(), memsrc_e::host};
    zs::Vector<vec2i> surfEdgeInds{mesh.nSurfEdges(), memsrc_e::host};
    zs::Vector<vec3i> surfTriInds{mesh.nSurfTris(), memsrc_e::host};
    tiles_t tets{tetTags, mesh.nTets(), memsrc_e::host};

    constexpr auto space = execspace_e::openmp;
    auto ompExec = omp_exec();
    auto meshVerts = applyTransform(mesh.getVerts(), transform);

    ompExec(range(verts.size()),
            [verts = view<space>(verts),
                xTag = verts.getPropertyOffset("x"),
                x0Tag = verts.getPropertyOffset("x0"),
                vTag = verts.getPropertyOffset("v"),
                mTag = verts.getPropertyOffset("m"),
                volTag = verts.getPropertyOffset("vol"),
                contactTag = verts.getPropertyOffset("contact"),
                &meshVerts](int tid) mutable
            {
                auto const &x = meshVerts.row(tid);
                vec3 xVec{x[0], x[1], x[2]};
                verts.tuple(dim_c<3>, xTag, tid) = xVec;
                verts.tuple(dim_c<3>, x0Tag, tid) = xVec;
                verts.tuple(dim_c<3>, vTag, tid) = vec3::zeros();
                verts(mTag, tid) = 0;
                verts(volTag, tid) = 0;
                verts.tuple(dim_c<3>, contactTag, tid) = vec3::zeros();
            });

    // surface vertices
    auto const &meshSurfVertInds = mesh.getSurfVertInds();
    ompExec(range(surfVertInds.size()),
            [surfVertInds = view<space>(surfVertInds),
                &meshSurfVertInds](int tid) mutable
            {
                surfVertInds[tid] = meshSurfVertInds[tid];
            });

    // surface edges
    auto const &meshSurfEdges = mesh.getSurfEdges();
    ompExec(range(surfEdgeInds.size()),
            [surfEdgeInds = view<space>(surfEdgeInds),
                &meshSurfEdges](int tid) mutable
            {
                auto const &e = meshSurfEdges.row(tid);
                surfEdgeInds(tid) = vec2i{e[0], e[1]};
            });

    // surface triangles
    auto const &meshSurfTris = mesh.getSurfTris();
    ompExec(range(surfTriInds.size()),
            [surfTriInds = view<space>(surfTriInds),
                &meshSurfTris](int tid) mutable
            {
                auto const &t = meshSurfTris.row(tid);
                surfTriInds[tid] = vec3i{t[0], t[1], t[2]};
            });

    // tets / volume-related
    auto const &meshTets = mesh.getTets();
    ompExec(range(tets.size()),
            [tets = view<space>({}, tets),
                verts = view<space>({}, verts),
                volTag = tets.getPropertyOffset("vol"),
                mTag = tets.getPropertyOffset("m"),
                indsTag = tets.getPropertyOffset("inds"),
                ibTag = tets.getPropertyOffset("IB"),
                x0Tag = verts.getPropertyOffset("x0"),
                vmTag = verts.getPropertyOffset("m"),
                vvolTag = verts.getPropertyOffset("vol"),
                &meshTets,
                density](int tid) mutable
            {
                auto const &t = meshTets.row(tid);
                tets.tuple(dim_c<4>, indsTag, tid) = vec4{reinterpret_bits<value_t>((ind_t)t[0]), reinterpret_bits<value_t>((ind_t)t[1]), reinterpret_bits<value_t>((ind_t)t[2]), reinterpret_bits<value_t>((ind_t)t[3])};
                auto&& tt = tets.pack(dim_c<4>, indsTag, tid).template reinterpret_bits<ind_t>();
                // NOTE: currently, use the non-SVD way to compute IB
                vec3 xs[4];
                for (int d = 0; d != 4; ++d)
                    xs[d] = verts.pack(dim_c<3>, x0Tag, t[d]);
                vec3 ds[3] = {xs[1] - xs[0], xs[2] - xs[0], xs[3] - xs[0]};
                mat3 B{};
                for (int d = 0; d != 3; ++d)
                    for (int i = 0; i != 3; ++i)
                        B(d, i) = ds[i][d];
                auto IB = inverse(B);
                if (std::isnan(IB(0, 0)) || std::isnan(IB(0, 1)) || std::isnan(IB(0, 2)) ||
                    std::isnan(IB(1, 0)) || std::isnan(IB(1, 1)) || std::isnan(IB(1, 2)) ||
                    std::isnan(IB(2, 0)) || std::isnan(IB(2, 1)) || std::isnan(IB(2
                    , 2)))
                {
                    printf("[ERROR] IB is nan!!!\n");
                    IB = mat3::zeros();
                }
                tets.tuple(dim_c<9>, ibTag, tid) = IB;

                auto vol = ds[1].dot(ds[2].cross(ds[0])) / static_cast<value_t>(6);
                if (vol < 0)
                {
                    printf("[ERROR] negative volume!!!\n");
                }
                tets(volTag, tid) = vol;
                tets(mTag, tid) = vol * density;
                for (int d = 0; d != 4; ++d)
                {
                    atomic_add(exec_omp, &verts(vmTag, t[d]),
                           vol * density / static_cast<value_t>(4));
                    atomic_add(exec_omp, &verts(vvolTag, t[d]),
                            vol / static_cast<value_t>(4)); 
                }
            });
    auto bodyPtr = std::make_shared<TetBodyT>(name, verts, surfVertInds, surfEdgeInds, surfTriInds, tets, meshVerts, mesh.getSurfTris(), mesh.getTets());
    bodyPtr->moveTo({memsrc_e::device, 0});
    return bodyPtr;
}

template <class TetBodyT, std::enable_if_t<is_tet_body_v<TetBodyT>, int> = 0>
std::shared_ptr<TetBodyT> genTetBodyFromTetMesh(const meshio::TetMesh &mesh, typename TetBodyT::value_t density, Eigen::Matrix4d const &transform = Eigen::Matrix4d::Identity())
{
    return genTetBodyFromTetMesh<TetBodyT>("", mesh, density, transform);
}

template <class TetBodyT, std::enable_if_t<is_tet_body_v<TetBodyT>, int> = 0>
std::shared_ptr<TetBodyT> genTetBodyFromTetMesh(const meshio::TetMesh &mesh, typename TetBodyT::value_t density, zs::vec<typename TetBodyT::value_t, 4, 4> const &transform)
{
    return genTetBodyFromTetMesh<TetBodyT>(mesh, density, vec2Eigen(transform));
}

template <class TetBodyT, std::enable_if_t<is_tet_body_v<TetBodyT>, int> = 0>
std::shared_ptr<TetBodyT> genTetBodyFromTetMesh(std::string name, const meshio::TetMesh &mesh, typename TetBodyT::value_t density, zs::vec<typename TetBodyT::value_t, 4, 4> const &transform)
{
    return genTetBodyFromTetMesh<TetBodyT>(mesh, density, vec2Eigen(transform));
}

/// @brief Rigid Body Properties
/// @tparam BodyT 
// template <class BodyT, std::enable_if_t<is_body_v<BodyT>, int>> 
template <class T, std::size_t TvLen, std::enable_if_t<std::is_floating_point_v<T>, int>>
struct RigidBodyProperties
{
    // static_assert(is_body_v<BodyT>, "Invalid body type.");
    using value_t = T;
    using ind_t = zs::conditional_t<zs::is_same_v<T, double>, zs::i64, zs::i32>;
    using tiles_t = zs::TileVector<value_t, TvLen>;
    using vec3 = zs::vec<value_t, 3>;
    using vec12 = zs::vec<value_t, 12>;
    using mat4 = zs::vec<value_t, 4, 4>;
    using mat4x3 = zs::vec<value_t, 4, 3>;
    using mat12 = zs::vec<value_t, 12, 12>;
    using mat3x12 = zs::vec<value_t, 3, 12>;
    using constitutiveModel_t = ConstitutiveModel<T, BodyType::Rigid>;

    BodySP<T, TvLen> body;
    tiles_t &verts;

    vec3 center;   
    value_t m;     
    value_t vol;
    vec12 q0;
    vec12 q;
    vec12 v;
    mat12 M;
    vec12 contactForce;
    vec12 extForce;
    bool isBC;
    vec12 BCTarget;
    value_t customBCStiffness; 
    bool hasSpring;
    vec12 springTarget;

    constitutiveModel_t constitutiveModel;

    template <class BodyT, zs::enable_if_all<
        is_rigid_body_v<BodyT>,
        std::is_same_v<typename BodyT::value_t, T>,
        BodyT::tvLength == TvLen 
    > = 0> 
    RigidBodyProperties(BodyT &body)
        : body{std::shared_ptr<BodyT>{&body}},
          verts(body.verts),
          center{vec3::zeros()},
          m{0},
          vol{0},
          q0{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}, 
          q{q0},
          v{vec12::zeros()}, 
          M{mat12::zeros()},
          contactForce{vec12::zeros()}, 
          extForce{vec12::zeros()},
          isBC{false},
          BCTarget{vec12::zeros()},
          customBCStiffness{-1},
          hasSpring{false},
          springTarget{vec12::zeros()} 
    {
        using namespace zs;

        constexpr auto space = execspace_e::openmp;
        auto ompExec = omp_exec();

        zs::Vector<value_t> tempM{0, memsrc_e::host};
        zs::Vector<value_t> tempV{0, memsrc_e::host};
        zs::Vector<vec3> tempX(0, memsrc_e::host);

        if constexpr (is_triangle_body_v<BodyT>)
        {       
            auto &tris = body.tris;
            auto &verts = body.verts;
            tempM.resize(verts.size());
            tempV.resize(verts.size());
            tempX.resize(verts.size());
            // tempM.resize(tris.size());
            // tempV.resize(tris.size());
            // tempX.resize(tris.size());
            // ompExec(range(tris.size()),
            ompExec(range(verts.size()),
                    [verts = view<space>({}, verts),
                    // tris = view<space>({}, tris), 
                    tempM = view<space>(tempM),
                    tempV = view<space>(tempV),
                    tempX = view<space>(tempX),
                    // indsTag = tris.getPropertyOffset("inds"),
                    // volTag = tris.getPropertyOffset("vol"),
                    // mTag = tris.getPropertyOffset("m"),
                    volTag = verts.getPropertyOffset("vol"),
                    mTag = verts.getPropertyOffset("m"),
                    x0Tag = verts.getPropertyOffset("x0"),
                    solid = body.solid](int tid) mutable
                    {
                        // auto&& t = tris.pack(dim_c<3>, indsTag, tid).template reinterpret_bits<ind_t>();
                        // auto mass = tempM(tid) = tris(mTag, tid); 
                        // tempV(tid) = tris(volTag, tid);
                        // tempX(tid) = vec3::zeros();
                        // for (int d = 0; d != 3; ++d)
                        //     tempX(tid) += verts.pack(dim_c<3>, x0Tag, t[d]) * mass;
                        // tempX[tid] /= solid?(static_cast<value_t>(4)):(static_cast<value_t>(3));
                        auto mass = verts(mTag, tid);
                        tempM(tid) = mass;
                        tempV(tid) = verts(volTag, tid);
                        if (solid)
                        {
                            tempM(tid) *= static_cast<value_t>(4 / 3);
                            tempV(tid) *= static_cast<value_t>(4 / 3);
                        }
                        tempX(tid) = verts.pack(dim_c<3>, x0Tag, tid) * mass;
                    });
        }
        else // TetBody
        {
            auto &tets = body.tets;
            auto &verts = body.verts;
            tempM.resize(tets.size());
            tempV.resize(tets.size());
            tempX.resize(tets.size());
            ompExec(range(tets.size()),
                    [tets = view<space>({}, tets), 
                    verts = view<space>({}, verts),
                    tempM = view<space>(tempM),
                    tempV = view<space>(tempV),
                    tempX = view<space>(tempX),
                    indsTag = tets.getPropertyOffset("inds"),
                    volTag = tets.getPropertyOffset("vol"),
                    mTag = tets.getPropertyOffset("m"),
                    x0Tag = verts.getPropertyOffset("x0")](int tid) mutable
                    {
                        auto&& t = tets.pack(dim_c<4>, indsTag, tid).template reinterpret_bits<ind_t>();
                        auto mass = tempM(tid) = tets(mTag, tid); 
                        tempV(tid) = tets(volTag, tid);
                        tempX(tid) = vec3::zeros();
                        for (int d = 0; d != 4; ++d)
                        {
                            tempX(tid) += verts.pack(dim_c<3>, x0Tag, t[d]) * mass;
                        }
                        tempX[tid] /= static_cast<value_t>(4);
                    });
        }

        m = reduce(ompExec, tempM);
        vol = reduce(ompExec, tempV);
        center = (m > 0)?(reduce(ompExec, tempX) / m):vec3::zeros();

        // de-offset the vertices
        ompExec(range(body.verts.size()), 
            [verts = view<space>({}, body.verts), 
            x0Tag = body.verts.getPropertyOffset("x0"),
            &center = this->center] (int vi) mutable {
                verts.tuple(dim_c<3>, x0Tag, vi) = verts.pack(dim_c<3>, x0Tag, vi) - center; 
            }); 

        // compute q0, q
        q = vec12{center[0], center[1], center[2], 
                1, 0, 0, 0, 1, 0, 0, 0, 1}; 
        if (body.verts.size()) // compute M
        {
            tempM.resize(144);
            tempM.reset(0);
            ompExec(range(body.verts.size()),
                    [verts = view<space>({}, body.verts),
                    x0Tag = body.verts.getPropertyOffset("x0"),
                    mTag = body.verts.getPropertyOffset("m"),
                    temp = view<space>(tempM)] (int vi) mutable
                    {
                        auto &&x0 = verts.pack(dim_c<3>, x0Tag, vi);
                        auto J = mat3x12::zeros();
                        for (int d = 0; d < 3; ++d)
                        {
                            J(d, d) = static_cast<value_t>(1);
                            for (int i = 0; i < 3; ++i)
                                J(i, 3 + i * 3 + d) = x0[d];
                        }

                        auto localM = verts(mTag, vi) * J.transpose() * J;
                        for (int di = 0; di < 12; ++di)
                            for (int dj = 0; dj < 12; ++dj)
                                atomic_add(exec_omp, &temp(di * 12 + dj), localM(di, dj));
                    });
            if constexpr (is_triangle_body_v<BodyT>) 
                if (body.solid) { // add mass at center if solid triangle body
                    auto J = mat3x12::zeros();
                    for (int d = 0; d < 3; ++d)
                    {
                        J(d, d) = static_cast<value_t>(1);
                        // for (int i = 0; i < 3; ++i)
                        //     J(i, 3 + i * 3 + d) = center[d];
                    }
                    auto centerM = m / static_cast<value_t>(4) * J.transpose() * J;
                    for (int di = 0; di < 12; ++di)
                        for (int dj = 0; dj < 12; ++dj)
                            tempM.setVal(centerM(di, dj) + tempM.getVal(di * 12 + dj), di * 12 + dj);
                }
            for (int di = 0; di < 12; ++di)
                for (int dj = 0; dj < 12; ++dj)
                {
                    M(di, dj) = tempM.getVal(di * 12 + dj);
                    if ((di == dj) && (std::abs(M(di, dj)) < zs::limits<T>::epsilon()))
                        M(di, dj) = 10;//zs::limits<T>::epsilon();
                }
        }
        else
        {
            // M = mat12::zeros();
            for (int d = 0; d < 12; ++d)
                M(d, d) = zs::limits<T>::epsilon();
        }
    }
    void setDensity(value_t density)
    {
        auto m_old = m;
        m = vol * density;
        M *= m / m_old;
    }
    void updateMesh()
    {
        std::visit([&](auto &body) { body->updateMesh(); }, body);
    }
    void setQ(vec12 const &q)
    {
        using namespace zs;

        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

        this->q = q;
        auto memloc = verts.memoryLocation();
        verts = verts.clone({memsrc_e::host, -1});
        pol(range(verts.size()),
            [verts = view<space>(verts),
             xTag = verts.getPropertyOffset("x"),
             x0Tag = verts.getPropertyOffset("x0"),
             q](int vi) mutable
            {
                auto x0 = verts.pack(dim_c<3>, x0Tag, vi);
                verts.tuple(dim_c<3>, xTag, vi) = ABD_q2x(x0, q);
            });
        verts = verts.clone(memloc);
        updateMesh();
    }
    void setBC(bool isBC = true)
    {
        this->isBC = isBC;
        BCTarget = q;
    }
    template <class ElemT>
    void setBCTarget(eigen::vec12<ElemT> const &q)
    {
        setBCTarget(eigen2Vec(q));
    }
    template <class ElemT>
    void setBCTarget(Eigen::Matrix4<ElemT> const &transform)
    {
        setBCTarget(eigen2Vec(transform));
    }
    void setBCTarget(vec12 const &q)
    {
        using namespace zs; 

        BCTarget = q;
        for (int d = 0; d < 3; d++)
            for (int j = 0; j < 3; j++)
                BCTarget[d] += BCTarget[3 + d * 3 + j] * center[j]; 
    }
    void setBCTarget(mat4 const &transform)
    {
        using namespace zs; 

        for (int di = 0; di < 3; di++){
            BCTarget[di] = transform[di][3]; 
            for (int dj = 0; dj < 3; dj++)
                BCTarget[3 + di * 3 + dj] = transform[di][dj]; 
        }
        for (int d = 0; d < 3; d++)
            for (int j = 0; j < 3; j++)
                BCTarget[d] += BCTarget[3 + d * 3 + j] * center[j]; 
    }
    void setCustomeBCStiffness(value_t stiffness)
    {
        customBCStiffness = stiffness;
    }
    void setSpringTarget(vec12 const &q)
    {
        hasSpring = true;
        springTarget = q;
    }
    void removeSpring()
    {
        hasSpring = false;
    }
    void clearForce()
    {
        extForce = vec12::zeros();
    }
    void applyUniformForce(vec3 const &force)
    {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto pol = cuda_exec();

        Vector<vec12> extf{1, memsrc_e::device, 0}; 
        extf.setVal(vec12::zeros()); 
        pol(range(verts.size()), 
            [verts = proxy<space>({}, verts), 
            extf = proxy<space>(extf), 
            mTag = verts.getPropertyOffset("m"),
            x0Tag = verts.getPropertyOffset("x0"),
            force = force] __device__ (int vi) mutable {
                force *= verts(mTag, vi); 
                auto p0 = verts.pack(dim_c<3>, x0Tag, vi); 
                for (int d = 0; d != 3; d++)
                {
                    atomic_add(exec_cuda, &extf[0][d], (T)force[d]); 
                    for (int k = 0; k != 3; k++)
                        atomic_add(exec_cuda, &extf[0][3 + k * 3 + d], p0(d) * force[k]); 
                }
            }); 
        extForce += extf.getVal();
    }
    template <class ElemT>
    void applyUniformForce(eigen::vec3<ElemT> const &force)
    {
        applyUniformForce(to_vec3<T>(force));
    }
    void applyPointForce(vec3 const &force, vec3 const &point)
    {
        using namespace zs;

        for (int di = 0; di < 3; di++)
        {
            extForce[di] += force[di];  // test torque 
            for (int dj = 0; dj < 3; dj++)
                extForce[3 + di * 3 + dj] += force[di] * point[dj]; 
        }
    }
    template <class ElemT>
    void applyPointForce(eigen::vec3<ElemT> const &force, eigen::vec3<ElemT> const &point)
    {
        applyPointForce(to_vec3<T>(force), to_vec3<T>(point));
    }
    template <template<class, int, int> class MatT, class ElemT>
    void applyQForce(MatT<ElemT, 4, 3> const &force)
    {
        for (int di = 0; di < 4; di++)
            for (int dj = 0; dj < 3; dj++)
                extForce[di * 3 + dj] += force[di, dj];
    }
    void applyQForce(vec12 const &force)
    {
        extForce += force;
    }
    template <class ElemT>
    void applyQForce(eigen::vec12<ElemT> const &force)
    {
        applyQForce(eigen2Vector<vec12>(force));
    }
    void applyTorque(vec3 const &axis)
    {
        if (axis.norm() < zs::limits<float>::epsilon())
            return; 
        
        auto point = getOrthoVec(axis);
        auto force = axis.cross(point).normalized() * axis.norm(); 
        auto restPoint = getRestPoint(point, q); 
        for (int di = 0; di < 3; di++)
        {
            for (int dj = 0; dj < 3; dj++)
                extForce[3 + di * 3 + dj] += force[di] * restPoint[dj]; 
        }
    }
    void moveTo(zs::MemoryLocation const &mloc){}
    auto& getModel() 
    {
        return constitutiveModel;
    }
    auto& getModel() const 
    {
        return constitutiveModel;
    }
};

template<class T, std::size_t TvLen, std::enable_if_t<std::is_floating_point_v<T>, int>> 
struct SoftBodyProperties
{
    // static_assert(is_body_v<BodyT>, "Invalid body type.");
    // static constexpr BodyType bodyType = BodyType::Soft;
    using value_t = T;
    using ind_t = zs::conditional_t<zs::is_same_v<T, double>, zs::i64, zs::i32>;
    using tiles_t = zs::TileVector<value_t, TvLen>;
    using vec3 = zs::vec<value_t, 3>;
    using vec4 = zs::vec<value_t, 4>;
    using mat4 = zs::vec<value_t, 4, 4>;
    using constitutiveModel_t = ConstitutiveModel<T, BodyType::Soft>;

    BodySP<T, TvLen> body;
    tiles_t &verts;

    vec3 contactForce;
    vec3 extForce;
    bool isBC;
    value_t customBCStiffness;

    constitutiveModel_t constitutiveModel;

    template <class BodyT, zs::enable_if_all<
        is_soft_body_v<BodyT>,
        std::is_same_v<typename BodyT::value_t, T>,
        BodyT::tvLength == TvLen 
    > = 0> 
    SoftBodyProperties(BodyT &body)
        : body{std::shared_ptr<BodyT>{&body}}, verts(body.verts), contactForce{vec3::zeros()}, extForce{vec3::zeros()}, isBC{false}, customBCStiffness{-1}
    {
        using namespace zs;

        constexpr auto space = execspace_e::openmp;
        auto ompExec = omp_exec();

        zs::Vector<value_t> tempM{0, memsrc_e::host};
        zs::Vector<vec3> tempX(0, memsrc_e::host);

        if constexpr (is_triangle_body_v<BodyT>)
        {       
            auto &tris = body.tris;
            auto &verts = body.verts;
            tempM.resize(tris.size());
            tempX.resize(tris.size());
            ompExec(range(tris.size()),
                    [tris = view<space>({}, tris), 
                    verts = view<space>({}, verts),
                    tempM = view<space>(tempM),
                    tempX = view<space>(tempX),
                    indsTag = tris.getPropertyOffset("inds"),
                    mTag = tris.getPropertyOffset("m"),
                    x0Tag = verts.getPropertyOffset("x0"),
                    solid = body.solid](int tid) mutable
                    {
                        auto&& t = tris.pack(dim_c<3>, indsTag, tid).template reinterpret_bits<ind_t>();
                        auto mass = tempM(tid) = tris(mTag, tid); 
                        tempX(tid) = vec3::zeros();
                        for (int d = 0; d != 3; ++d)
                            tempX(tid) += verts.pack(dim_c<3>, x0Tag, t[d]) * mass;
                        tempX[tid] /= solid?(static_cast<value_t>(4)):(static_cast<value_t>(3));
                    });
        }
        else // TetBody
        {
            auto &tets = body.tets;
            auto &verts = body.verts;
            tempM.resize(tets.size());
            tempX.resize(tets.size());
            ompExec(range(tets.size()),
                    [tets = view<space>({}, tets), 
                    verts = view<space>({}, verts),
                    tempM = view<space>(tempM),
                    tempX = view<space>(tempX),
                    indsTag = tets.getPropertyOffset("inds"),
                    mTag = tets.getPropertyOffset("m"),
                    x0Tag = verts.getPropertyOffset("x0")](int tid) mutable
                    {
                        auto&& t = tets.pack(dim_c<4>, indsTag, tid).template reinterpret_bits<ind_t>();
                        auto mass = tempM(tid) = tets(mTag, tid); 
                        tempX(tid) = vec3::zeros();
                        for (int d = 0; d != 4; ++d)
                        {
                            tempX(tid) += verts.pack(dim_c<3>, x0Tag, t[d]) * mass;
                        }
                        tempX[tid] /= static_cast<value_t>(4);
                    });
        }

        auto m = reduce(ompExec, tempM);
        auto center = (m > 0)?(reduce(ompExec, tempX) / m):vec3::zeros();

        // de-offset the vertices
        ompExec(range(body.verts.size()), 
            [verts = view<space>({}, body.verts), 
            x0Tag = body.verts.getPropertyOffset("x0"),
            &center = center] (int vi) mutable {
                verts.tuple(dim_c<3>, x0Tag, vi) = verts.pack(dim_c<3>, x0Tag, vi) - center; 
            }); 
    }
    void setDensity(value_t density)
    {
        // nothing to do
    }
    template <int Rows, class ElemT>
    void setBC(Eigen::Matrix<ElemT, Rows, 1> const &isBC)
    {
        setBC(eigen2Vector(isBC));
    }
    void setBC(zs::Vector<bool> const &isBC)
    {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto pol = cuda_exec();

        auto d_isBC = isBC.clone(verts.get_allocator());
        if (!verts.hasProperty("isBC"))
            verts.append_channels(pol, {{"isBC", 1}, {"BCTarget", 3}});

        pol(range(verts.size()), 
            [verts = view<space>({}, verts), 
            isBC = view<space>(d_isBC),
            isBCTag = verts.getPropertyOffset("isBC"), 
            BCTgtTag = verts.getPropertyOffset("BCTarget"),
            xTag = verts.getPropertyOffset("x")
            ]__device__ (int vi) mutable {
                verts(isBCTag, vi) = isBC(vi);
                if (isBC(vi))
                    verts.tuple(dim_c<3>, BCTgtTag, vi) = verts.pack(dim_c<3>, xTag, vi);
                // auto xn = verts.pack(dim_c<3>, xTag, vi);
                // auto xTilde = verts.pack(dim_c<3>, BCTgtTag, vi);
                // printf("vertex %d, x: %.4f %.4f %.4f, x_tilde: %.4f %.4f %.4f\n", vi, (float)xn(0), (float)xn(1), (float)xn(2), (float)xTilde(0), (float)xTilde(1), (float)xTilde(2));
            }); 
    }

    void setBC(bool isBC = true)
    {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto pol = cuda_exec();

        if (verts.getPropertyOffset("isBC") == -1)
            verts.append_channels(pol, {{"isBC", 1}, {"BCTarget", 3}});

        pol(range(verts.size()), 
            [verts = view<space>({}, verts), 
            isBCTag = verts.getPropertyOffset("isBC"), 
            BCTgtTag = verts.getPropertyOffset("BCTarget"),
            xTag = verts.getPropertyOffset("x"),
            isBC
            ]__device__ (int vi) mutable {
                verts(isBCTag, vi) = isBC;
                if (isBC)
                    verts.tuple(dim_c<3>, BCTgtTag, vi) = verts.pack<3>(xTag, vi);
            }); 
    }

    template <class ElemT>
    void setBCTarget(eigen::matX3<ElemT> const &targetVerts)
    {
        using namespace zs; 
        constexpr auto space = execspace_e::cuda; 
        auto pol = zs::cuda_exec();
        
        auto targetPos = eigen2Vector(targetVerts);
        targetPos = targetPos.clone(verts.get_allocator());
        pol(range(verts.size()), 
            [verts = proxy<space>({}, verts), 
            targetPos = proxy<space>(targetPos),
            BCTgtTag = verts.getPropertyOffset("BCTarget")] __device__ (int vi) mutable {
                verts.tuple(dim_c<3>, BCTgtTag, vi) = targetPos[vi]; 
            }); 
    }

    template <class ElemT>
    void setBCTarget(Eigen::Matrix4<ElemT> const &transform)
    {
        setBCTarget(eigen2Vec(transform));
    }
    void setBCTarget(mat4 const &transform)
    {
        using namespace zs; 
        constexpr auto space = execspace_e::cuda; 
        auto pol = zs::cuda_exec();

        pol(range(verts.size()), 
            [verts = proxy<space>({}, verts), 
            transform = transform,
            BCTgtTag = verts.getPropertyOffset("BCTarget")] __device__ (int vi) mutable {
                auto p = to_point4<T>(verts.pack<3>(BCTgtTag, vi));
                p = transform * p;
                                verts.tuple(dim_c<3>, BCTgtTag, vi) = to_vec3<T>(p); 
            }); 
    }

    void setCustomBCStiffness(T stiffness)
    {
        customBCStiffness = stiffness;
    }

    template <class ElemT>
    void setSpringConstraint(eigen::matX3<ElemT> const &targetVerts, T weight)
    {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto pol = zs::cuda_exec();         

        if (targetVerts.rows() != verts.size())
            throw std::runtime_error("The sizes of vertices and spring targets should be the same!");

        auto targetPos = eigen2Vector(targetVerts);
        targetPos = targetPos.clone(verts.get_allocator());
        if (!verts.hasProperty("spring_target"))
        {
            verts.append_channels(pol, {{"spring_target", 3}, {"spring_ws", 3}, {"spring_w", 1}});
            pol(range(verts.size()),
                [verts = proxy<space>({}, verts), 
                targetPos = proxy<space>(targetPos),
                springTgtTag = verts.getPropertyOffset("spring_target"),
                springWsTag = verts.getPropertyOffset("spring_ws"),
                springWTag = verts.getPropertyOffset("spring_w"),
                weight] __device__ (int vi) mutable {
                    auto& target = targetPos[vi];
                    verts.tuple(dim_c<3>, springTgtTag, vi) = target; 
                    verts.tuple(dim_c<3>, springWsTag, vi) = weight * target; 
                    verts(springWTag, vi) = weight; 
                });             
        } else {
            pol(range(verts.size()), 
                [verts = proxy<space>({}, verts), 
                targetPos = proxy<space>(targetPos),
                springTgtTag = verts.getPropertyOffset("spring_target"),
                springWsTag = verts.getPropertyOffset("spring_ws"),
                springWTag = verts.getPropertyOffset("spring_w"),
                weight] __device__ (int vi) mutable {
                    auto& target = targetPos[vi];
                    verts(springWTag, vi) += weight; 
                    auto ws = verts.pack<3>(springWsTag, vi) + weight * target;
                    verts.tuple(dim_c<3>, springWsTag, vi) = ws;
                    verts.tuple(dim_c<3>, springTgtTag, vi) = ws / verts(springWTag, vi);
                });
        }
    }

    void moveTo(zs::MemoryLocation const &mloc){}
    auto& getModel() 
    {
        return constitutiveModel;
    }
    auto& getModel() const 
    {
        return constitutiveModel;
    }
};


template <template <typename> class Model, class BodyT, class... Args, 
    zs::enable_if_all<is_body_v<BodyT>, 
        std::is_constructible_v<Model<typename BodyT::value_t>, Args...>
    > = 0
>
void setConstitutiveModel(BodyT& body, Args &&... args) // TODO: support multiple constitutive models (e.g. for soft bodies, elastic + plastic)
{
    using value_t = typename BodyT::value_t;
    using constitutiveModel_t = typename BodyT::bodyTypeProperties_t::constitutiveModel_t;

    body.getModel().setModel(Model<value_t>{args...});
}


template <enum BodyType BodyType_v, class T, std::size_t TvLen, std::enable_if_t<std::is_floating_point_v<T>, int> I>
auto& TriangleBody<BodyType_v, T, TvLen, I>::getBodyTypeProperties()
{
    return bodyTypeProperties;
}
template <enum BodyType BodyType_v, class T, std::size_t TvLen, std::enable_if_t<std::is_floating_point_v<T>, int> I>
auto& TriangleBody<BodyType_v, T, TvLen, I>::getBodyTypeProperties() const
{
    return bodyTypeProperties;
}

template <enum BodyType BodyType_v, class T, std::size_t TvLen, std::enable_if_t<std::is_floating_point_v<T>, int> I>
auto& TetBody<BodyType_v, T, TvLen, I>::getBodyTypeProperties()
{
    return bodyTypeProperties;
}
template <enum BodyType BodyType_v, class T, std::size_t TvLen, std::enable_if_t<std::is_floating_point_v<T>, int> I>
auto& TetBody<BodyType_v, T, TvLen, I>::getBodyTypeProperties() const
{
    return bodyTypeProperties;
}

template <enum BodyType BodyType_v, class T, std::size_t TvLen, std::enable_if_t<std::is_floating_point_v<T>, int> I>
void TriangleBody<BodyType_v, T, TvLen, I>::moveTo(zs::MemoryLocation const &mloc)
{
    verts = verts.clone(mloc);
    vertInds = vertInds.clone(mloc);
    edgeInds = edgeInds.clone(mloc);
    triInds = triInds.clone(mloc);
    tris = tris.clone(mloc);
    bodyTypeProperties.moveTo(mloc);
}

template <enum BodyType BodyType_v, class T, std::size_t TvLen, std::enable_if_t<std::is_floating_point_v<T>, int> I>
void TetBody<BodyType_v, T, TvLen, I>::moveTo(zs::MemoryLocation const &mloc)    
{
    verts = verts.clone(mloc);
    surfVertInds = surfVertInds.clone(mloc);
    surfEdgeInds = surfEdgeInds.clone(mloc);
    surfTriInds = surfTriInds.clone(mloc);
    tets = tets.clone(mloc);
    bodyTypeProperties.moveTo(mloc);
}

} // namespace tacipc