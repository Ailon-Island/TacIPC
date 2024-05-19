#pragma once
#include <cassert>
#include <tuple>
#include <unordered_set>
#include <vector>
#include <tacipc/types.hpp>
#include <tacipc/meta.hpp>
#include <zensim/container/TileVector.hpp>
#include <zensim/execution/Atomics.hpp>

namespace meshio
{
// reference:
// https://stackoverflow.com/questions/7110301/generic-hash-for-tuples-in-unordered-map-unordered-set
template <class T> inline void hash_combine(std::size_t &seed, T const &v)
{
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl
{
    static void apply(size_t &seed, Tuple const &tuple)
    {
	HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
	hash_combine(seed, std::get<Index>(tuple));
    }
};

template <class Tuple> struct HashValueImpl<Tuple, 0>
{
    static void apply(size_t &seed, Tuple const &tuple)
    {
	hash_combine(seed, std::get<0>(tuple));
    }
};

template <typename TupleT> struct HashTuple
{
    size_t operator()(TupleT const &tt) const
    {
	size_t seed = 0;
	HashValueImpl<TupleT>::apply(seed, tt);
	return seed;
    }
};

template <typename TupleT> struct HashTupleUnordered
{
    size_t operator()(TupleT const &tt) const
    {
	size_t seed = 0;
	HashValueImpl<TupleT>::apply(seed, tt);
	return seed;
    }
};

template <class Ta, class Tb> static uint64_t concatInts(Ta a, Tb b)
{
    return (static_cast<uint64_t>(a) << 32) + static_cast<uint64_t>(b);
}

template <class Ta, class Tb> static uint64_t unorderedConcatInts(Ta a, Tb b)
{
    return a < b ? concatInts(a, b) : concatInts(b, a);
}

template <class TupleT> static TupleT sortTriplet(TupleT const &triplet)
{
    auto t0 = std::get<0>(triplet);
    auto t1 = std::get<1>(triplet);
    auto t2 = std::get<2>(triplet);
    if (t2 < t1)
	std::swap(t1, t2);
    if (t1 < t0)
	std::swap(t0, t1);
    if (t2 < t1)
	std::swap(t1, t2);
    return std::make_tuple(t0, t1, t2);
}

class TriMesh
{
    eigen::matX3d verts;
    eigen::vecXi vertInds;
    eigen::matX2i edges;
    eigen::matX3i tris;

    void genVertInds()
    {
        vertInds.resize(verts.rows(), Eigen::NoChange);
        for (int i = 0; i < verts.rows(); ++i)
            vertInds[i] = i;
    }

    void genEdges()
    {
	std::unordered_map<uint64_t, eigen::vec2i> edgeMap;
	for (std::size_t ti = 0; ti < tris.rows(); ++ti)
	{
	    auto tri = tris.row(ti);
	    edgeMap[unorderedConcatInts(tri[0], tri[1])] =
		eigen::vec2i{tri[0], tri[1]};
	    edgeMap[unorderedConcatInts(tri[1], tri[2])] =
		eigen::vec2i{tri[1], tri[2]};
	    edgeMap[unorderedConcatInts(tri[2], tri[0])] =
		eigen::vec2i{tri[2], tri[0]};
	}
	edges.resize(edgeMap.size(), Eigen::NoChange);
	int ei = 0;
	for (const auto &it : edgeMap)
	    edges.row(ei++) << it.second[0], it.second[1];
    }

  public:
    TriMesh(){};

    TriMesh(const eigen::matX3d &verts, const eigen::matX3i &tris)
	: verts{verts}, tris{tris}
    {
    genVertInds();
	genEdges();
    }

    TriMesh(eigen::matX3d &&verts, eigen::matX3i &&tris)
	: verts{verts}, tris{tris}
    {
    genVertInds();
	genEdges();
    }

    TriMesh(eigen::matX3d &&verts, eigen::matX3i &&tris, eigen::vecXi &&vertInds)
    : verts{verts}, tris{tris}, vertInds{vertInds}
    {
    genEdges();
    }

    const eigen::matX3d &getVerts() const { return verts; }

    void setVerts(const eigen::matX3d &verts) { this->verts = verts; }

    const eigen::vecXi &getVertInds() const { return vertInds; }

    const eigen::matX3i &getTris() const { return tris; }

    const eigen::matX2i &getEdges() const { return edges; }

    std::size_t nVerts() const { return verts.rows(); }

    std::size_t nEdges() const { return edges.rows(); }

    std::size_t nTris() const { return tris.rows(); }
};

struct TetMesh
{
    eigen::matX3d verts;
    eigen::vecXi surfVertInds;
    eigen::matX3i surfTris;
    eigen::matX4i tets;
    TriMesh surfMesh;

    void genSurfTris()
    {
	using triplet_t = std::tuple<int, int, int>;
	std::vector<triplet_t> orderedTris;
	std::vector<int> trisCnts;
	int trisSize = 0;
	int surfTrisSize = 0;
	std::unordered_map<triplet_t, std::size_t, HashTuple<triplet_t>>
	    unorderedTrisIndMap;
    std::unordered_set<int> surfVertIndSet;
	auto addTri =
	    [&](const auto &tet, ::size_t ia, std::size_t ib, std::size_t ic)
	{
	    auto orderedTri = std::make_tuple(tet[ia], tet[ib], tet[ic]);
	    auto unorderedTri = sortTriplet(orderedTri);
	    if (auto it = unorderedTrisIndMap.find(unorderedTri);
		it != unorderedTrisIndMap.end())
	    {
		auto ind = it->second;
		auto cnt = trisCnts[ind];
		assert(((void)"[TetMesh]:\tEach triangle should be shared by 2 "
			      "tetrahedra at most!",
			cnt == 1));
		++trisCnts[ind];
		--surfTrisSize;
	    }
	    else
	    {
		trisCnts.push_back(1);
		orderedTris.push_back(std::move(orderedTri));
		unorderedTrisIndMap.insert(
		    {std::move(unorderedTri), trisSize++});
		++surfTrisSize;
	    }
	};

	for (std::size_t teti = 0; teti < tets.rows(); ++teti)
	{
	    auto tet = tets.row(teti);
	    addTri(tet, 0, 2, 1);
	    addTri(tet, 0, 1, 3);
	    addTri(tet, 0, 3, 2);
	    addTri(tet, 1, 2, 3);
	}

	surfTris.resize(surfTrisSize, Eigen::NoChange);
	surfTrisSize = 0;
	for (std::size_t ti = 0; ti < trisSize; ti++)
	{
	    if (trisCnts[ti] == 1)
	    {
		auto const &orderedTri = orderedTris[ti];
		surfTris.row(surfTrisSize++) << std::get<0>(orderedTri),
		    std::get<1>(orderedTri), std::get<2>(orderedTri);
        surfVertIndSet.insert(std::get<0>(orderedTri));
        surfVertIndSet.insert(std::get<1>(orderedTri));
        surfVertIndSet.insert(std::get<2>(orderedTri));
	    }
	}

    surfVertInds.resize(surfVertIndSet.size(), Eigen::NoChange);
    std::vector<int> surfVertIndsVec{surfVertIndSet.begin(), surfVertIndSet.end()};
    std::sort(surfVertIndsVec.begin(), surfVertIndsVec.end());
    surfVertInds = eigen::vecXi::Map(surfVertIndsVec.data(), surfVertIndsVec.size());

	surfMesh = TriMesh{verts, surfTris};
    }

  public:
    TetMesh(){};

    TetMesh(const eigen::matX3d &verts, const eigen::matX4i &tets)
	: verts{verts}, tets{tets}
    {
	genSurfTris();
    }

    TetMesh(eigen::matX3d &&verts, eigen::matX4i &&tets)
	: verts{verts}, tets{tets}
    {
	genSurfTris();
    }

    const eigen::matX3d &getVerts() const { return verts; }

    void setVerts(const eigen::matX3d &verts) { this->verts = verts; surfMesh = TriMesh{verts, surfTris}; }

    const eigen::matX4i &getTets() const { return tets; }

    const eigen::vecXi &getSurfVertInds() const { return surfVertInds; }

    const eigen::matX3i &getSurfTris() const { return surfTris; }

    const eigen::matX2i &getSurfEdges() const { return surfMesh.getEdges(); }

    TriMesh extractSurfMesh() const { return surfMesh; }

    std::size_t nVerts() const { return verts.rows(); }

    std::size_t nSurfVerts() const { return surfVertInds.rows(); }

    std::size_t nSurfTris() const { return surfTris.rows(); }

    std::size_t nSurfEdges() const { return getSurfEdges().rows(); }

    std::size_t nTets() const { return tets.rows(); }
};

using Mesh = std::variant<TriMesh, TetMesh>;
template <typename T>
constexpr bool is_mesh_v = tacipc::is_variant_member_v<T, Mesh>;
} // namespace meshio
