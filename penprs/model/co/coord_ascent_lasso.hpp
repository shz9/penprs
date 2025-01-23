#ifndef COORD_ASCENT_LASSO_H
#define COORD_ASCENT_LASSO_H

#include "co_utils.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <type_traits>


template <typename T, typename U, typename I>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value && std::is_integral<I>::value, void>::type
update_beta_lasso(int c_size,
    int *ld_left_bound,
    I *ld_indptr,
    U *ld_data,
    T *std_beta,
    T *beta,
    T *beta_diff,
    T *q,
    T *n_per_snp,
    T lam,
    T lam_min,
    T dq_scale,
    int threads,
    bool low_memory) {

    int start, end;
    I ld_start, ld_end;

    #ifdef _OPENMP
        #pragma omp parallel for private(ld_start, ld_end, start, end) schedule(static) num_threads(threads)
    #endif
    for (int j = 0; j < c_size; ++j) {

        ld_start = ld_indptr[j];
        ld_end = ld_indptr[j + 1];
        start = ld_left_bound[j];
        end = start + (ld_end - ld_start);

        T curr_beta = beta[j];
        T abs_zj = std::abs(n_per_snp[j] * (std_beta[j] - q[j]));

        T sft_diff = abs_zj - lam;
        beta[j] = (1. / (n_per_snp[j]*(1 + lam_min))) * sft_diff *(sft_diff>0) * (-2*(std_beta[j] < q[j]) + 1);

        // difference between updated and prev beta
        beta_diff[j] = beta[j] - curr_beta;

        // no difference if beta isn't updated
        if (beta_diff[j] != 0) {
            /* Update the q-factors for variants that are in LD with variant j */
            blas_axpy(q + start, ld_data + ld_start, dq_scale*beta_diff[j], end - start);

            if (!low_memory){
                q[j] = q[j] - beta_diff[j];
            }
        }
    }

    if (low_memory) {
        update_q_factor(c_size, ld_left_bound, ld_indptr, ld_data, &beta_diff[0], q, dq_scale, threads);
    }
}


#endif
