#pragma once
#include <fmt/core.h>
#include <suitesparse/cholmod.h>
#include <cassert>
#include <tuple>
#include <utility>
#include <zensim/profile/CppTimers.hpp>

struct DirectSolver
{
    cholmod_sparse *A = nullptr;
    cholmod_dense *x = nullptr, *b = nullptr, *r = nullptr;
    cholmod_factor *L = nullptr;
    cholmod_common c;
    cholmod_triplet_struct *A_triplets = nullptr;

    DirectSolver() { cholmod_start(&c); }

    void allocate(std::size_t nrows, std::size_t ncols, std::size_t nzmax)
    {
        A_triplets =
            cholmod_allocate_triplet(nrows, ncols, nzmax, -1, CHOLMOD_REAL, &c);
        A_triplets->stype = 1;
        b = cholmod_allocate_dense(ncols, 1, ncols, CHOLMOD_REAL, &c);
    }

    void setNNZ(std::size_t nnz) { A_triplets->nnz = nnz; }

    std::tuple<int *, int *, double *> matData()
    {
        return std::make_tuple((int *)A_triplets->i, (int *)A_triplets->j,
                               (double *)A_triplets->x);
    }

    double *vecData() { return (double *)b->x; }

    double *resultData() { return (double *)x->x; }

    void solve()
    {
        zs::CppTimer timer;
        timer.tick();
        double one[2] = {1, 0}, m1[2] = {-1, 0};
        A = cholmod_triplet_to_sparse(A_triplets, A_triplets->nnz, &c);
        // A->stype = 0;
        L = cholmod_analyze(A, &c);              /* analyze */
        cholmod_factorize(A, L, &c);             /* factorize */
        x = cholmod_solve(CHOLMOD_A, L, b, &c);  /* solve Ax=b */
        r = cholmod_copy_dense(b, &c);           /* r = b */
        cholmod_sdmult(A, 0, m1, one, x, r, &c); /* r = r-Ax */
        timer.tock();
        printf("host solver: norm(b-Ax) %8.1e in %fms\n",
               cholmod_norm_dense(r, 0, &c),
               (float)(timer.elapsed())); /* print norm(r) */
    }

    ~DirectSolver()
    {
        if (A_triplets)
            cholmod_free_triplet(&A_triplets, &c);
        if (L)
            cholmod_free_factor(&L, &c); /* free matrices */
        if (A)
            cholmod_free_sparse(&A, &c);
        if (r)
            cholmod_free_dense(&r, &c);
        if (x)
            cholmod_free_dense(&x, &c);
        if (b)
            cholmod_free_dense(&b, &c);
        cholmod_finish(&c);
    }
};

struct SupernodalDirectSolver
{
    using index_t = SuiteSparse_long;
    cholmod_sparse *A = nullptr;
    cholmod_dense *x = nullptr, *b = nullptr, *r = nullptr;
    cholmod_factor *L = nullptr;
    cholmod_common c;
    cholmod_triplet_struct *A_triplets = nullptr;

    SupernodalDirectSolver()
    {
        cholmod_l_start(&c);
        c.useGPU = 1;
        c.supernodal = CHOLMOD_SUPERNODAL;
    }

    void allocate(std::size_t nrows, std::size_t ncols, std::size_t nzmax)
    {
        A_triplets = cholmod_l_allocate_triplet(nrows, ncols, nzmax, -1,
                                                CHOLMOD_REAL, &c);
        A_triplets->stype = 1;
        b = cholmod_l_allocate_dense(ncols, 1, ncols, CHOLMOD_REAL, &c);
    }

    void setNNZ(std::size_t nnz) { A_triplets->nnz = nnz; }

    std::tuple<index_t *, index_t *, double *> matData()
    {
        return std::make_tuple((index_t *)A_triplets->i,
                               (index_t *)A_triplets->j,
                               (double *)A_triplets->x);
    }

    double *vecData() { return (double *)b->x; }

    double *resultData() { return (double *)x->x; }

    void solve()
    {
        double one[2] = {1, 0}, m1[2] = {-1, 0};

        A = cholmod_l_triplet_to_sparse(A_triplets, A_triplets->nnz, &c);
        L = cholmod_l_analyze(A, &c);              /* analyze */
        cholmod_l_factorize(A, L, &c);             /* factorize */
        x = cholmod_l_solve(CHOLMOD_A, L, b, &c);  /* solve Ax=b */
        r = cholmod_l_copy_dense(b, &c);           /* r = b */
        cholmod_l_sdmult(A, 0, m1, one, x, r, &c); /* r = r-Ax */
        printf("host solver: norm(b-Ax) %8.1e\n",
               cholmod_l_norm_dense(r, 0, &c)); /* print norm(r) */
        cholmod_l_gpu_stats(&c);
    }

    ~SupernodalDirectSolver()
    {
        if (A_triplets)
            cholmod_l_free_triplet(&A_triplets, &c);
        if (L)
            cholmod_l_free_factor(&L, &c); /* free matrices */
        if (A)
            cholmod_l_free_sparse(&A, &c);
        if (r)
            cholmod_l_free_dense(&r, &c);
        if (x)
            cholmod_l_free_dense(&x, &c);
        if (b)
            cholmod_l_free_dense(&b, &c);
        cholmod_l_finish(&c);
    }
};
