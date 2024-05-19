#include <tacipc/generation.cuh>

namespace tacipc
{
template struct TriangleBody<BodyType::Rigid, float>;
template struct TriangleBody<BodyType::Rigid, double>;
template struct TriangleBody<BodyType::Soft, float>;
template struct TriangleBody<BodyType::Soft, double>;
template struct TetBody<BodyType::Rigid, float>;
template struct TetBody<BodyType::Rigid, double>;
template struct TetBody<BodyType::Soft, float>;
template struct TetBody<BodyType::Soft, double>;

template struct RigidBodyProperties<float>;
template struct RigidBodyProperties<double>;
template struct SoftBodyProperties<float>;
template struct SoftBodyProperties<double>;

template <class TriBodyT, std::enable_if_t<is_triangle_body_v<TriBodyT>, int>>
void to_json(json &j, TriBodyT const &b)
{
    j["name"] = b.name;
    j["id"] = b.id;
    j["layer"] = b.layer;
    j["verts"] = b.verts;
    j["vertInds"] = b.vertInds;
    j["edgeInds"] = b.edgeInds;
    j["triInds"] = b.triInds;
    j["tris"] = b.tris;
    j["solid"] = b.solid;
    j["bodyTypeProperties"] = b.bodyTypeProperties;
}

template <class TetBodyT, std::enable_if_t<is_tet_body_v<TetBodyT>, int>>
void to_json(json &j, TetBodyT const &b)
{
    j["name"] = b.name;
    j["id"] = b.id;
    j["layer"] = b.layer;
    j["verts"] = b.verts;
    j["surfVertInds"] = b.surfVertInds;
    j["surfEdgeInds"] = b.surfEdgeInds;
    j["surfTriInds"] = b.surfTriInds;
    j["tets"] = b.tets;
    j["bodyTypeProperties"] = b.bodyTypeProperties;
}

template <class RBPropsT, std::enable_if_t<is_rigid_body_properties_v<RBPropsT>, int>>
void to_json(json &j, RBPropsT const &p)
{
    j["center"] = p.center;
    j["m"] = p.m;
    j["vol"] = p.vol;
    j["q0"] = p.q0;
    j["q"] = p.q;
    j["v"] = p.v;
    j["M"] = p.M;

    j["contactForce"] = p.contactForce;
    j["extForce"] = p.extForce;

    j["isBC"] = p.isBC;
    j["BCTarget"] = p.BCTarget;
    j["customBCStiffness"] = p.customBCStiffness;

    j["hasSpring"] = p.hasSpring;
    j["springTarget"] = p.springTarget;

    // j["constitutiveModel"] = p.constitutiveModel;
}

template <class SBPropsT, std::enable_if_t<is_soft_body_properties_v<SBPropsT>, int>>
void to_json(json &j, SBPropsT const &p)
{
    j["contactForce"] = p.contactForce;
    j["extForce"] = p.extForce;

    j["isBC"] = p.isBC;
    j["customBCStiffness"] = p.customBCStiffness;

    // j["constitutiveModel"] = p.constitutiveModel;
}

template std::shared_ptr<RigidTriangleBody<float>> genTriBodyFromTriMesh<RigidTriangleBody<float>, true>(std::string name, meshio::TriMesh const&, float density, Eigen::Matrix4d const&);
template std::shared_ptr<RigidTriangleBody<double>> genTriBodyFromTriMesh<RigidTriangleBody<double>, true>(std::string name, meshio::TriMesh const&, double density, Eigen::Matrix4d const&);
template std::shared_ptr<SoftTriangleBody<float>> genTriBodyFromTriMesh<SoftTriangleBody<float>, true>(std::string name, meshio::TriMesh const&, float density, Eigen::Matrix4d const&);
template std::shared_ptr<SoftTriangleBody<double>> genTriBodyFromTriMesh<SoftTriangleBody<double>, true>(std::string name, meshio::TriMesh const&, double density, Eigen::Matrix4d const&);
template std::shared_ptr<RigidTriangleBody<float>> genTriBodyFromTriMesh<RigidTriangleBody<float>, false>(std::string name, meshio::TriMesh const&, float density, Eigen::Matrix4d const&);
template std::shared_ptr<RigidTriangleBody<double>> genTriBodyFromTriMesh<RigidTriangleBody<double>, false>(std::string name, meshio::TriMesh const&, double density, Eigen::Matrix4d const&);
template std::shared_ptr<SoftTriangleBody<float>> genTriBodyFromTriMesh<SoftTriangleBody<float>, false>(std::string name, meshio::TriMesh const&, float density, Eigen::Matrix4d const&);
template std::shared_ptr<SoftTriangleBody<double>> genTriBodyFromTriMesh<SoftTriangleBody<double>, false>(std::string name, meshio::TriMesh const&, double density, Eigen::Matrix4d const&);
template std::shared_ptr<RigidTriangleBody<float>> genTriBodyFromTriMesh<RigidTriangleBody<float>, true>(meshio::TriMesh const&, float density, Eigen::Matrix4d const&);
template std::shared_ptr<RigidTriangleBody<double>> genTriBodyFromTriMesh<RigidTriangleBody<double>, true>(meshio::TriMesh const&, double density, Eigen::Matrix4d const&);
template std::shared_ptr<SoftTriangleBody<float>> genTriBodyFromTriMesh<SoftTriangleBody<float>, true>(meshio::TriMesh const&, float density, Eigen::Matrix4d const&);
template std::shared_ptr<SoftTriangleBody<double>> genTriBodyFromTriMesh<SoftTriangleBody<double>, true>(meshio::TriMesh const&, double density, Eigen::Matrix4d const&);
template std::shared_ptr<RigidTriangleBody<float>> genTriBodyFromTriMesh<RigidTriangleBody<float>, false>(meshio::TriMesh const&, float density, Eigen::Matrix4d const&);
template std::shared_ptr<RigidTriangleBody<double>> genTriBodyFromTriMesh<RigidTriangleBody<double>, false>(meshio::TriMesh const&, double density, Eigen::Matrix4d const&);
template std::shared_ptr<SoftTriangleBody<float>> genTriBodyFromTriMesh<SoftTriangleBody<float>, false>(meshio::TriMesh const&, float density, Eigen::Matrix4d const&);
template std::shared_ptr<SoftTriangleBody<double>> genTriBodyFromTriMesh<SoftTriangleBody<double>, false>(meshio::TriMesh const&, double density, Eigen::Matrix4d const&);

template std::shared_ptr<RigidTetBody<float>> genTetBodyFromTetMesh<RigidTetBody<float>>(std::string name, meshio::TetMesh const&, float density, Eigen::Matrix4d const&);
template std::shared_ptr<RigidTetBody<double>> genTetBodyFromTetMesh<RigidTetBody<double>>(std::string name, meshio::TetMesh const&, double density, Eigen::Matrix4d const&);
template std::shared_ptr<SoftTetBody<float>> genTetBodyFromTetMesh<SoftTetBody<float>>(std::string name, meshio::TetMesh const&, float density, Eigen::Matrix4d const&);
template std::shared_ptr<SoftTetBody<double>> genTetBodyFromTetMesh<SoftTetBody<double>>(std::string name, meshio::TetMesh const&, double density, Eigen::Matrix4d const&);
template std::shared_ptr<RigidTetBody<float>> genTetBodyFromTetMesh<RigidTetBody<float>>(meshio::TetMesh const&, float density, Eigen::Matrix4d const&);
template std::shared_ptr<RigidTetBody<double>> genTetBodyFromTetMesh<RigidTetBody<double>>(meshio::TetMesh const&, double density, Eigen::Matrix4d const&);
template std::shared_ptr<SoftTetBody<float>> genTetBodyFromTetMesh<SoftTetBody<float>>(meshio::TetMesh const&, float density, Eigen::Matrix4d const&);
template std::shared_ptr<SoftTetBody<double>> genTetBodyFromTetMesh<SoftTetBody<double>>(meshio::TetMesh const&, double density, Eigen::Matrix4d const&);

} // namespace tacipc