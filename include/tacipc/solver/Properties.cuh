#pragma once

#include <zensim/types/SmallVector.hpp>
#include <zensim/container/TileVector.hpp>

namespace tacipc
{
template <class PrevT, std::size_t... Sizes>
struct StaticPropertyTag
{
    static constexpr bool isPropertyTag = true;

    static constexpr bool has_prev = !std::is_same_v<PrevT, std::false_type>;
    const PrevT prev;

    static constexpr int numChannels = 1 * (Sizes * ...);
    static constexpr std::size_t dimensions = sizeof...(Sizes);

    static constexpr zs::value_seq<Sizes...> dimensionSizes{};

    const char *name;
    std::size_t offset;

    constexpr StaticPropertyTag(const char *name, std::size_t offset, const PrevT prev)
        : name(name), offset(offset), prev(prev)
    {}
};

template <std::size_t... Sizes>
constexpr auto createStaticPropertyTag(const char *name)
{
    return StaticPropertyTag<std::false_type, Sizes...>{name, 0, {}};
}

template <std::size_t... Sizes, class PrevT>
constexpr auto createStaticPropertyTag(const char *name, const PrevT &prev)
{
    return StaticPropertyTag<PrevT, Sizes...>{name, prev.offset + prev.numChannels, prev};
}

template <typename TagT>
std::vector<zs::PropertyTag> createPropertyTagsVector(TagT tag)
{
    if constexpr (TagT::has_prev)
    {
        auto tags = createPropertyTagsVector(tag.prev);
        // fmt::print("tag: name={}, numChannels={}, offset={}\n", tag.name, tag.numChannels, tag.offset);
        tags.push_back({tag.name, tag.numChannels});
        return tags;
    }
    else
    {
        // fmt::print("tag: name={}, numChannels={}, offset={}\n", tag.name, tag.numChannels, tag.offset);
        return {{tag.name, tag.numChannels}};
    }
}

template <typename T> class Properties
{
  public:
    using tv_t = T;
    using value_t = typename tv_t::value_type;
    virtual ~Properties() = default;
};

template <typename T = zs::TileVector<double, 32>>
class AffineBodyProperties : public Properties<T>
{
  public:
    using tv_t = T;
    static inline constexpr auto contact = createStaticPropertyTag<12>("contact");
    static inline constexpr auto center = createStaticPropertyTag<3>("center", contact);
    static inline constexpr auto isGhost = createStaticPropertyTag<1>("isGhost", center);
    static inline constexpr auto layer = createStaticPropertyTag<1>("layer", isGhost);
    static inline constexpr auto exclGrpIdx = createStaticPropertyTag<1>("excl_group_index", layer);
    static inline constexpr auto q = createStaticPropertyTag<12>("q", exclGrpIdx);
    static inline constexpr auto M = createStaticPropertyTag<12, 12>("M", q);
    static inline constexpr auto vol = createStaticPropertyTag<1>("vol", M);
    static inline constexpr auto grav = createStaticPropertyTag<12>("grav", vol);
    static inline constexpr auto extf = createStaticPropertyTag<12>("extf", grav);
    static inline constexpr auto qn = createStaticPropertyTag<12>("qn", extf);
    static inline constexpr auto qn0 = createStaticPropertyTag<12>("qn0", qn);
    static inline constexpr auto qDir = createStaticPropertyTag<12>("q_dir", qn0);
    static inline constexpr auto qDot = createStaticPropertyTag<12>("q_dot", qDir);
    static inline constexpr auto qTilde = createStaticPropertyTag<12>("q_tilde", qDot);
    static inline constexpr auto temp = createStaticPropertyTag<12>("temp", qTilde);
    static inline constexpr auto r = createStaticPropertyTag<12>("r", temp);
    static inline constexpr auto Pre = createStaticPropertyTag<3, 3, 4>("Pre", r);
    static inline constexpr auto E = createStaticPropertyTag<1>("E", Pre);
    static inline constexpr auto qHat = createStaticPropertyTag<12>("q_hat", E);
    static inline constexpr auto cons = createStaticPropertyTag<12>("cons", qHat);
    static inline constexpr auto lambda = createStaticPropertyTag<12>("lambda", cons);
    static inline constexpr auto isBC = createStaticPropertyTag<1>("isBC", lambda);
    static inline constexpr auto BCStiffness = createStaticPropertyTag<1>("custom_bc_stiffness", isBC);
    static inline constexpr auto hasSpring = createStaticPropertyTag<1>("has_spring", BCStiffness);
    static inline constexpr auto springTarget = createStaticPropertyTag<12>("spring_target", hasSpring);

    static inline auto propertyTags = createPropertyTagsVector(springTarget);

    AffineBodyProperties() = default;
};

template <typename T = zs::TileVector<double, 32>>
class VertexProperties : public Properties<T>
{
  public:
    using tv_t = T;
    static inline constexpr auto grad = createStaticPropertyTag<3>("grad");
    static inline constexpr auto m = createStaticPropertyTag<1>("m", grad);
    static inline constexpr auto ws = createStaticPropertyTag<1>("ws", m); // also as contraint jacobian
    static inline constexpr auto vol = createStaticPropertyTag<1>("vol", ws);
    static inline constexpr auto dir = createStaticPropertyTag<3>("dir", vol);
    static inline constexpr auto xn = createStaticPropertyTag<3>("xn", dir);
    static inline constexpr auto xHat = createStaticPropertyTag<3>("x_hat", xn);
    static inline constexpr auto xTilde = createStaticPropertyTag<3>("x_tilde", xHat);
    static inline constexpr auto grav = createStaticPropertyTag<3>("grav", xTilde);
    static inline constexpr auto extf = createStaticPropertyTag<3>("extf", grav);
    static inline constexpr auto isBC = createStaticPropertyTag<1>("isBC", extf);
    static inline constexpr auto cons = createStaticPropertyTag<3>("cons", isBC);
    static inline constexpr auto lambda = createStaticPropertyTag<3>("lambda", cons);
    static inline constexpr auto vn = createStaticPropertyTag<3>("vn", lambda);
    static inline constexpr auto xn0 = createStaticPropertyTag<3>("xn0", vn);
    static inline constexpr auto x0 = createStaticPropertyTag<3>("x0", xn0);
    static inline constexpr auto temp = createStaticPropertyTag<3>("temp", x0);
    static inline constexpr auto JVec = createStaticPropertyTag<3>("J_vec", temp);
    static inline constexpr auto J = createStaticPropertyTag<3, 12>("J", JVec);
    static inline constexpr auto body = createStaticPropertyTag<1>("body", J);
    static inline constexpr auto layer = createStaticPropertyTag<1>("layer", body);
    static inline constexpr auto springTarget = createStaticPropertyTag<3>("spring_target", layer);
    static inline constexpr auto exclGrpIdx = createStaticPropertyTag<1>("excl_group_index", springTarget);
    static inline constexpr auto springW = createStaticPropertyTag<1>("spring_w", exclGrpIdx);
    static inline constexpr auto contact = createStaticPropertyTag<3>("contact", springW);
    static inline constexpr auto Pre = createStaticPropertyTag<3, 3>("Pre", contact);
    static inline constexpr auto friction_force = createStaticPropertyTag<3>("friction_force", Pre);
    static inline constexpr auto collision_force = createStaticPropertyTag<3>("collision_force", friction_force);
    static inline constexpr auto elastic_force = createStaticPropertyTag<3>("elastic_force", collision_force);


    static inline auto propertyTags = createPropertyTagsVector(elastic_force);

    VertexProperties() = default;
};

template <typename T = zs::TileVector<double, 32>>
class DoFProperties : public Properties<T>
{
  public:
    using tv_t = T;
    static inline constexpr auto grad = createStaticPropertyTag<1>("grad");
    static inline constexpr auto dir = createStaticPropertyTag<1>("dir", grad);
    static inline constexpr auto temp = createStaticPropertyTag<1>("temp", dir);
    static inline constexpr auto p = createStaticPropertyTag<1>("p", temp);
    static inline constexpr auto q = createStaticPropertyTag<1>("q", p);
    static inline constexpr auto r = createStaticPropertyTag<1>("r", q);

    static inline auto propertyTags = createPropertyTagsVector(r);

    DoFProperties() = default;
};

template <typename T = zs::TileVector<double, 32>>
class SurfaceVertexProperties : public Properties<T>
{
  public:
    using tv_t = T;
    static inline constexpr auto inds = createStaticPropertyTag<1>("inds");
    static inline constexpr auto fn = createStaticPropertyTag<1>("fn", inds);

    static inline auto propertyTags = createPropertyTagsVector(fn);

    SurfaceVertexProperties() = default;
};


template <std::size_t Dim = 1, typename T = zs::TileVector<double, 32>>
class IndsProperties : public Properties<T>
{
  public:
    using tv_t = T;
    constexpr static std::size_t dim = Dim;
    static inline constexpr auto inds = createStaticPropertyTag<Dim>("inds");

    static inline auto propertyTags = createPropertyTagsVector(inds);

    IndsProperties() = default;
};

template <typename T = zs::TileVector<double, 32>>
class FrictionPointPointProperties : public Properties<T>
{
  public:
    using tv_t = T;
    static inline constexpr auto basis = createStaticPropertyTag<3, 2>("basis");
    static inline constexpr auto fn = createStaticPropertyTag<1>("fn", basis);

    static inline auto propertyTags = createPropertyTagsVector(fn);

    FrictionPointPointProperties() = default;
};

template <typename T = zs::TileVector<double, 32>>
class FrictionPointEdgeProperties : public Properties<T>
{
  public:
    using tv_t = T;
    static inline constexpr auto basis = createStaticPropertyTag<3, 2>("basis");
    static inline constexpr auto fn = createStaticPropertyTag<1>("fn", basis);
    static inline constexpr auto yita = createStaticPropertyTag<1>("yita", fn);

    static inline auto propertyTags = createPropertyTagsVector(yita);

    FrictionPointEdgeProperties() = default;
};

template <typename T = zs::TileVector<double, 32>>
class FrictionPointTriangleProperties : public Properties<T>
{
  public:
    using tv_t = T;
    static inline constexpr auto basis = createStaticPropertyTag<3, 2>("basis");
    static inline constexpr auto fn = createStaticPropertyTag<1>("fn", basis);
    static inline constexpr auto beta = createStaticPropertyTag<2>("beta", fn);

    static inline auto propertyTags = createPropertyTagsVector(beta);

    FrictionPointTriangleProperties() = default;
};

template <typename T = zs::TileVector<double, 32>>
class FrictionEdgeEdgeProperties : public Properties<T>
{
  public:
    using tv_t = T;
    static inline constexpr auto basis = createStaticPropertyTag<3, 2>("basis");
    static inline constexpr auto fn = createStaticPropertyTag<1>("fn", basis);
    static inline constexpr auto gamma = createStaticPropertyTag<2>("gamma", fn);

    static inline auto propertyTags = createPropertyTagsVector(gamma);

    FrictionEdgeEdgeProperties() = default;
};

} // namespace tacipc