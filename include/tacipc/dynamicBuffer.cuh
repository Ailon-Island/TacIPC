#pragma once
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <optional>
#include <zensim/container/Vector.hpp>
#include <zensim/cuda/Cuda.h>
#include <zensim/cuda/execution/ExecutionPolicy.cuh>
#include <zensim/execution/Atomics.hpp>
#include <zensim/math/Vec.h>
#include <zensim/resource/Resource.h>

namespace tacipc
{
// DynamicBuffer
template <typename ValT> struct DynamicBuffer
{
    DynamicBuffer(std::size_t n = 0)
        : buf{n, zs::memsrc_e::device, 0}, cnt{1, zs::memsrc_e::device, 0},
            prevCount{0}
    {
        reset();
    }
    ~DynamicBuffer() = default;

    std::size_t getCount() const { return cnt.getVal(); }
    std::size_t getBufferCapacity() const { return buf.capacity(); }
    ///
    void snapshot() { prevCount = cnt.size(); }
    void resizeToCounter()
    {
        auto c = getCount();
        if (c <= getBufferCapacity())
            return;
        buf.resize(c);
    }
    void restartCounter() { cnt.setVal(prevCount); }
    void reset() { cnt.setVal(0); }
    ///
    void assignCounterFrom(const DynamicBuffer &o) { cnt = o.cnt; }
    ///
    int reserveFor(int inc)
    {
        int v = cnt.getVal();
        buf.resize((std::size_t)(v + inc));
        return v;
    }

    zs::Vector<ValT> &_get_buf()
    {
        // for testing
        return buf;
    }

    struct Port
    {
        constexpr static auto space = zs::execspace_e::cuda; 
        using vec_t = zs::Vector<ValT>;
        using vec_view_t = decltype(zs::view<space>(std::declval<vec_t&>(), zs::false_c)); 
        vec_view_t buf;
        int *cnt;
        std::size_t cap;
        __forceinline__ __device__ void try_push(ValT &&val)
        {
            auto no = zs::atomic_add(zs::exec_cuda, cnt, 1);
            if (no < cap)
                buf[no] = std::move(val);
            else 
                printf("[Error] DynamicBuffer::Port::try_push: buffer overflow! capacity=%d\n", (int)cap);
        }
        __forceinline__ __device__ int
        next_index(zs::cg::thread_block_tile<8, zs::cg::thread_block> &tile)
        {
            int no = -1;
            if (tile.thread_rank() == 0)
                no = zs::atomic_add(zs::exec_cuda, cnt, 1);
            no = tile.shfl(no, 0);
            return (no < cap ? no : -1);
        }
        __forceinline__ __device__ int
        next_index_no_tile()
        {
            int no = zs::atomic_add(zs::exec_cuda, cnt, 1); 
            return (no < cap ? no : -1);
        }
        __forceinline__ __device__ ValT &operator[](int i)
        {
            return buf[i];
        }
        __forceinline__ __device__ const ValT &operator[](int i) const
        {
            return buf[i];
        }
    };
    Port port()
    {
        constexpr auto space = zs::execspace_e::cuda; 
        return Port{zs::view<space>(buf, zs::false_c, "dynamic_buffer_vector"), cnt.data(),
                    buf.capacity()}; // numDofs, tag;
    }

    protected:
    zs::Vector<ValT> buf;
    zs::Vector<int> cnt;
    std::size_t prevCount;
};
template <typename... DynBufs> void snapshot(DynBufs &...dynBufs)
{
    (void)((void)dynBufs.snapshot(), ...);
}
template <typename... DynBufs> bool allFit(DynBufs &...dynBufs)
{
    return ((dynBufs.getCount() <= dynBufs.getBufferCapacity()) && ...);
}
template <typename... DynBufs> void resizeAndRewind(DynBufs &...dynBufs)
{
    (void)((void)dynBufs.resizeToCounter(), ...);
    (void)((void)dynBufs.restartCounter(), ...);
}


} // namespace tacipc