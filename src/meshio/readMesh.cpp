#include <cstdint>
#include <filesystem>
#include <igl/pathinfo.h>
#include <igl/read_triangle_mesh.h>
#include <igl/readMSH.h>
#include <igl/readOBJ.h>
#include <tacipc/meshio/mesh.hpp>
#include <tacipc/meshio/readMesh.hpp>

namespace meshio
{
TriMesh readTriMesh(std::string meshPath)
{
    eigen::matX3d verts;
    eigen::matX3i tris;
    std::string dirname, basename, extension, filename;
    igl::pathinfo(meshPath, dirname, basename, extension, filename);
    if (extension == "dae")
    {
        meshPath = dirname + "/" + filename + ".obj";
    }
    igl::read_triangle_mesh(meshPath, verts, tris);
    return TriMesh{std::move(verts), std::move(tris)};
}

TriMesh readObjFile(std::string objPath)
{
    eigen::matX3d verts;
    eigen::matX3i tris;
    igl::readOBJ(objPath, verts, tris);
    return TriMesh{std::move(verts), std::move(tris)};
}

TetMesh readMshFile(std::string mshPath)
{
    eigen::matXXd verts;
    eigen::matXXi tris;
    eigen::matXXi tets;
    eigen::matX3i surfTris;
    eigen::vecXi triTag, tetTag;
    igl::readMSH(mshPath, verts, tris, tets, triTag, tetTag);

    return TetMesh{std::move(verts), std::move(tets)};
}
} // namespace meshio
