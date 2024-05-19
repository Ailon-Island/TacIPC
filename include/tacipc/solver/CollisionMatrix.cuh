#pragma once
#include <tacipc/meta.hpp>
#include <tacipc/utils.cuh>
// #include <tacipc/solver/VWrapper.cuh>
#include <zensim/math/Vec.h>
#include <zensim/execution/Atomics.hpp>
#include <zensim/cuda/execution/ExecutionPolicy.cuh>

namespace tacipc
{

template <class MatViewT, class size_type>
struct CollisionMatrixCopyCol
{
    CollisionMatrixCopyCol(MatViewT mat, size_type dst, size_type src)
        : mat{mat}, dst{dst}, src{src}
    {}

    MatViewT mat;
    size_type dst;
    size_type src;

    constexpr void operator()(size_type other)
    {
        mat.setCollision(dst, other, 
                         mat(src, other));
    }
};

///< collision matrix class
///  implemented by a list of 64-bit flags
///  only a triangular portion (row <= col) is legal
template <class T = zs::u64, typename AllocatorT = zs::ZSPmrAllocator<>>
struct CollisionMatrix : zs::Vector<T, AllocatorT>
{
    using allocator_type = AllocatorT;
    using line_type = T;
    constexpr static line_type one = 1;
    using mat_type = zs::Vector<T>;
    using size_type = std::size_t;
    using policy_type =
        std::variant<zs::CudaExecutionPolicy, zs::OmpExecutionPolicy>;

    mat_type mat;
    policy_type cudaPol {zs::cuda_exec()};
    policy_type ompPol {zs::omp_exec()};
    bool isEmpty = false;

    CollisionMatrix() 
        : isEmpty{true}
    {}

    CollisionMatrix(AllocatorT const &allocator, size_type size)
        : mat{allocator, size}
    {
        init();
    }

    explicit CollisionMatrix(size_type size, zs::memsrc_e mre = zs::memsrc_e::host, zs::ProcID devid = -1)
        : mat{size, mre, devid}
    {
        init();
    }

    CollisionMatrix(mat_type const &mat)
        : mat{mat}
    {
    }

    CollisionMatrix(mat_type &&mat)
        : mat{std::move(mat)}
    {
    }

    CollisionMatrix(CollisionMatrix const &other)
        : mat{other.mat}
    {
    }

    constexpr size_type size() const { return mat.size(); }
    constexpr void resize(size_type size) { mat.resize(size); }

    constexpr zs::memsrc_e memspace() const noexcept
    {
        return mat.memspace(); 
    }

    policy_type &getExecutionPolicy()
    {
        if (memspace() == zs::memsrc_e::host)
            return ompPol;
        else
            return cudaPol;
    }

    auto reset(int ch)
    {
        mat.reset(ch);
    }

    void fillZeros()
    {
        reset(0);
    }

    void init()
    {
        reset(-1);
    }

    CollisionMatrix<T, AllocatorT> clone(zs::MemoryLocation const &mloc)
    {
        return {mat.clone(mloc)};
    }

    constexpr bool getVal(size_type i, size_type j) const
    {
        if (i >= mat.size() || j >= mat.size())
            throw std::runtime_error("Collision map out of boundary");
        if (i > j)
        {
            size_type tmp = i;
            i = j;
            j = tmp;
        }
        return mat.getVal(i) & (one << j);
    }

    void duplicateLayer(size_type dst, size_type src)
    {
        using namespace zs;

        mat.setVal(mat.getVal(src), dst); // first copy the line

        // then copy the rest
        constexpr auto cudaSpace = execspace_e::cuda;
        constexpr auto ompSpace = execspace_e::openmp;
        std::visit([&](auto &pol) { 
                        if constexpr (std::is_same_v<decltype(pol), zs::CudaExecutionPolicy>)
                            pol(range(max(dst, src)), 
                                CollisionMatrixCopyCol(view<cudaSpace>(*this), dst, src));
                        else 
                            pol(range(max(dst, src)), 
                                CollisionMatrixCopyCol(view<ompSpace>(*this), dst, src));
                   },
                   getExecutionPolicy());
        setCollision(dst, dst, getVal(src, src));
    }

    void setCollision(size_type i, bool collide = true)
    {
        if (i >= mat.size())
            throw std::runtime_error("Collision map out of boundary");
        if (collide)
        {
            mat.setVal(-1, i);
            for (int j = 0; j < i; ++j)
            {
                mat.setVal(mat.getVal(j) | (one << i), j);
            }
        }
        else // intersect
        {
            mat.setVal(0, i);
            for (int j = 0; j < i; ++j)
            {
                mat.setVal(mat.getVal(j) & ~(one << i), j);
            }
        }
    }

    void setCollision(size_type i, size_type j, bool collide = true)
    {
        if (i >= mat.size() || j >= mat.size())
            throw std::runtime_error("Collision map out of boundary");
        if (i > j)
        {
            size_type tmp = i;
            i = j;
            j = tmp;
        }
        if (collide)
        {
            mat.setVal(mat.getVal(i) | (one << j), i);
        }
        else // intersect
        {
            mat.setVal(mat.getVal(i) & ~(one << j), i);
        }
    }
};

template <zs::execspace_e ExecSpace, typename CollisionMatrixT, bool Base = false> 
struct CollisionMatrixView 
{
    using line_type = typename CollisionMatrixT::line_type;
    constexpr static line_type one = 1;
    using size_type = typename CollisionMatrixT::size_type;
    constexpr static auto is_const_structure = std::is_const_v<CollisionMatrixT>;
    using decay_mat_type = typename CollisionMatrixT::mat_type;
    using mat_type = std::conditional_t<is_const_structure, const decay_mat_type, decay_mat_type>;
    using view_type = zs::VectorView<ExecSpace, mat_type, Base>;

    view_type view;

    CollisionMatrixView() noexcept = default;
    explicit constexpr CollisionMatrixView(CollisionMatrixT &collisionMatrix, const zs::SmallString &tagName = {})
        : view(zs::view<ExecSpace>(collisionMatrix.mat, zs::wrapv<Base>{}, tagName))
    {}

    constexpr bool operator()(size_type i, size_type j) const
    {
        if (i >= view.size() || j >= view.size())
            throw std::runtime_error("Collision map out of boundary");
        if (i > j)
        {
            size_type tmp = i;
            i = j;
            j = tmp;
        }
        return view(i) & (one << j);
    }

    constexpr void setCollision(size_type i, bool collide = true)
    {
        if (i >= view.size())
            throw std::runtime_error("Collision map out of boundary");
        if (collide)
        {
            view(i) = -1;
            for (int j = 0; j < i; ++j)
            {
                view(j) |= one << i;
            }
        }
        else // intersect
        {
            view(i) = 0;
            for (int j = 0; j < i; ++j)
            {
                view(j) &= ~(one << i);
            }
        }
    }

    constexpr void setCollision(size_type i, size_type j, bool collide = true)
    {
        if (i >= view.size() || j >= view.size())
            throw std::runtime_error("Collision map out of boundary");
        if (i > j)
        {
            size_type tmp = i;
            i = j;
            j = tmp;
        }
        if (collide)
        {
            view(i) |= one << j;
        }
        else // intersect
        {
            view(i) &= ~(one << j);
        }
    }

    constexpr void intersectWith(size_type i)
    {
        setCollision(i, false);
    }

    constexpr void intersectWith(size_type i, size_type j)
    {
        setCollision(i, j, false);
    }

    constexpr void collideWith(size_type i)
    {
        setCollision(i, true);
        
    }

    constexpr void collideWith(size_type i, size_type j)
    {
        setCollision(i, j, true);
    }
};

template <zs::execspace_e ExecSpace, typename T, typename AllocatorT, bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
constexpr decltype(auto) view(CollisionMatrix<T, AllocatorT> &collisionMat, zs::wrapv<Base> = {}, const zs::SmallString &tagName = {}) {
    return CollisionMatrixView<ExecSpace, CollisionMatrix<T, AllocatorT>, Base>(collisionMat, tagName);
}
template <zs::execspace_e ExecSpace, typename T, typename AllocatorT, bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
constexpr decltype(auto) view(const CollisionMatrix<T, AllocatorT> &collisionMat, zs::wrapv<Base> = {}, const zs::SmallString &tagName = {}) {
    return CollisionMatrixView<ExecSpace, const CollisionMatrix<T, AllocatorT>, Base>(collisionMat, tagName);
}

} // namespace tacipc

