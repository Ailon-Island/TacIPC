#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#include <tacipc/meta.hpp>
#include <tacipc/types.hpp>
#include <tacipc/solver/Properties.cuh>
#include <zensim/container/TileVector.hpp>
#include <zensim/omp/execution/ExecutionPolicy.hpp>
#include <zensim/cuda/execution/ExecutionPolicy.cuh>

namespace tacipc
{
template <typename PropertiesT> struct TVWrapper
{
    using tiles_t = typename PropertiesT::tv_t;
    using value_t = typename PropertiesT::value_t;
    using props_t = PropertiesT;
    using policy_t =
        std::variant<zs::CudaExecutionPolicy, zs::OmpExecutionPolicy>;
    // zs::CudaExecutionPolicy cudaPol;
    // zs::OmpExecutionPolicy ompPol;
    tiles_t tileVector;
    policy_t cudaPol, ompPol;
    
    // PropertiesT properties;
    TVWrapper(typename tiles_t::allocator_type const &allocator, size_t size)
        // : properties(),
        : tileVector(allocator, PropertiesT::propertyTags, size),
          cudaPol(zs::cuda_exec()), ompPol(zs::omp_exec())
    {
    }

    TVWrapper(tiles_t tileVector)
        : tileVector{tileVector}, 
          cudaPol(zs::cuda_exec()), ompPol(zs::omp_exec())
    {
    }

    TVWrapper<PropertiesT> clone(zs::MemoryLocation const &mloc)
    {
        return {tileVector.clone(mloc)};
    }
    
    constexpr zs::memsrc_e memspace() const noexcept
    {
        return tileVector.memspace(); 
    }
    decltype(auto) get_allocator() const noexcept
    {
        return tileVector.get_allocator();
    }
    auto size() const -> size_t { return tileVector.size(); }
    void resize(size_t size) { tileVector.resize(size); }

    policy_t &getExecutionPolicy()
    {
        if (memspace() == zs::memsrc_e::host)
            return ompPol;
        else
            return cudaPol;
    }

    template <typename T = double> auto fillZeros() -> void
    {
        std::visit([&](auto &pol) { tileVector.reset(pol, (T)0); },
                   getExecutionPolicy());
        // tileVector.reset(getExecutionPolicy(), T(0));
    }
};

template <typename TVWrapperT, zs::execspace_e ExecSpace, bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
struct View
{
    constexpr static auto execSpace = ExecSpace;
    using value_t = typename TVWrapperT::value_t;
    constexpr static auto is_const_structure = std::is_const_v<TVWrapperT>;
    using decay_tv_t = typename TVWrapperT::tiles_t;
    using tv_t = std::conditional_t<is_const_structure, const decay_tv_t, decay_tv_t>;
    using view_t = zs::TileVectorUnnamedView<ExecSpace, tv_t, false, Base>;
    using props_t = typename TVWrapperT::props_t;

    view_t view;

    View(TVWrapperT &wrapper, const zs::SmallString &tagName = {})
        : view(zs::view<ExecSpace>(wrapper.tileVector, zs::wrapv<Base>{}, tagName))
    {
    }

    template <typename TagT,
                typename SeqT = std::decay_t<decltype(TagT::dimensionSizes)>>
    constexpr auto pack(TagT tag, std::size_t index, SeqT seq = {}) const
    {
        if constexpr (seq.count == 0)
            return view.pack(tag.dimensionSizes, tag.offset, index);
        else
            return view.pack(seq, tag.offset, index);
    }

    template <auto N0, auto... Ns, typename TagT, zs::enable_if_t<TagT::isPropertyTag> = 0>
    constexpr auto pack(TagT tag, std::size_t index) const
    {
        if constexpr (sizeof...(Ns) == 0)
            static_assert(TagT::numChannels == N0, "pack: numChannels does not match the given shape");
        else
            static_assert(TagT::numChannels == N0 * (Ns * ...), "pack: numChannels does not match the given shape");

        return view.pack(zs::dim_c<N0, Ns...>, tag.offset, index);
    }

    template <typename TagT,
                typename SeqT = std::decay_t<decltype(TagT::dimensionSizes)>>
    constexpr auto tuple(TagT tag, std::size_t index, SeqT seq = std::decay_t<TagT>::dimensionSizes) const
    {
        // printf("tuple: %s, %d, %d\n", tag.name, (int)tag.offset, (int)index);
        if constexpr (seq.count == 0)
            return view.tuple(tag.dimensionSizes, tag.offset, index);
        else
            return view.tuple(seq, tag.offset, index);
    }
    
    template <auto... Ns, typename TagT, zs::enable_if_t<TagT::isPropertyTag> = 0>
    constexpr auto tuple(TagT tag, std::size_t index) const
    {
        static_assert(TagT::numChannels == (Ns * ...), "tuple: numChannels does not match the given shape");

        return view.tuple(zs::dim_c<Ns...>, tag.offset, index);
    }

    // non const
    template <typename TagT, bool V = is_const_structure, typename TT = value_t,
            zs::enable_if_all<!V, sizeof(TT) == sizeof(value_t), 
                              zs::is_same_v<TT, zs::remove_cvref_t<TT>>, TagT::numChannels == 1,
                              (std::alignment_of_v<TT> == std::alignment_of_v<value_t>)>
            = 0>
    constexpr std::add_lvalue_reference_t<TT> operator()(TagT tag, std::size_t index, zs::wrapt<TT> = {})
    {
        return view(tag.offset, index, zs::wrapt<TT>{});
    }

    template <typename TagT, bool V = is_const_structure, typename TT = value_t,
            zs::enable_if_all<!V, sizeof(TT) == sizeof(value_t), zs::is_same_v<TT, zs::remove_cvref_t<TT>>,
                        (std::alignment_of_v<TT> == std::alignment_of_v<value_t>)>
            = 0>
    constexpr std::add_lvalue_reference_t<TT> operator()(TagT tag, std::size_t dimension, std::size_t index, zs::wrapt<TT> = {})
    {
        return view(tag.offset + dimension, index, zs::wrapt<TT>{});
    }

    // const
    template <typename TagT, typename TT = value_t,
            zs::enable_if_all<sizeof(TT) == sizeof(value_t), 
                              zs::is_same_v<TT, zs::remove_cvref_t<TT>>,  TagT::numChannels == 1,
                              (std::alignment_of_v<TT> == std::alignment_of_v<value_t>)>
            = 0>
    constexpr auto operator()(TagT tag, std::size_t index, zs::wrapt<TT> = {}) const
    {
        return view(tag.offset, index, zs::wrapt<TT>{});
    }

    template <typename TagT, typename TT = value_t,
            zs::enable_if_all<sizeof(TT) == sizeof(value_t), zs::is_same_v<TT, zs::remove_cvref_t<TT>>,
                        (std::alignment_of_v<TT> == std::alignment_of_v<value_t>)>
            = 0>
    constexpr auto operator()(TagT tag, std::size_t dimension, std::size_t index, zs::wrapt<TT> = {}) const
    {
        return view(tag.offset + dimension, index, zs::wrapt<TT>{});
    }

    auto size() const -> std::size_t { return view.size(); }
};

template <zs::execspace_e ExecSpace, typename T, bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
constexpr decltype(auto) view(TVWrapper<T> &wrapper, zs::wrapv<Base> = {}, const zs::SmallString &tagName = {}) {
    // return wrapper.template view<ExecSpace, Base>(tagName);
    return View<TVWrapper<T>, ExecSpace, Base>(wrapper, tagName);
}
template <zs::execspace_e ExecSpace, typename T, bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
constexpr decltype(auto) view(const TVWrapper<T> &wrapper, zs::wrapv<Base> = {}, const zs::SmallString &tagName = {}) {
    // return wrapper.template view<ExecSpace, Base>(tagName);
    return View<const TVWrapper<T>, ExecSpace, Base>(wrapper, tagName);
}

} // namespace tacipc
