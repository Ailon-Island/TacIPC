#pragma once
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string_view>
#include <utility>
#include <vector>
#include <tacipc/meshio/mesh.hpp>
#include <zensim/math/Vec.h>

namespace meshio
{
TriMesh readTriMesh(std::string meshPath);
TriMesh readObjFile(std::string objPath);
// NOTE: currently, msh file is only for tetrahedral mesh
TetMesh readMshFile(std::string mshPath);
} // namespace meshio
