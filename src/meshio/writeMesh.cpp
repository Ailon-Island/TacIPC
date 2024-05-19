#include <cstdint>
#include <filesystem>
#include <igl/writeMSH.h>
#include <igl/writeOBJ.h>
#include <tacipc/meshio/mesh.hpp>
#include <tacipc/meshio/writeMesh.hpp>

namespace meshio
{
void writeObjFile(std::string objPath, const TriMesh &mesh)
{
    igl::writeOBJ(objPath, mesh.getVerts(), mesh.getTris());
}

void writeObjFile(std::string objPath, const TetMesh &mesh)
{
    igl::writeOBJ(objPath, mesh.getVerts(), mesh.getSurfTris());
}

void writeMshFile(std::string mshPath, const TetMesh &mesh)
{
    igl::writeMSH(mshPath, mesh.getVerts(), mesh.getSurfTris(), mesh.getTets(), {}, {}, {}, {}, {}, {}, {});
} 
} // namespace meshio
