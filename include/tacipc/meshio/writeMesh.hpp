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
#include <zensim/math/Vec.h>

namespace meshio
{
void writeObjFile(std::string objPath, const TriMesh &mesh);

void writeObjFile(std::string objPath, const TetMesh &mesh);

void writeMshFile(std::string mshPath, const TetMesh &mesh);
} // namespace meshio