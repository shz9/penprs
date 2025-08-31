#ifndef COORD_ASCENT_SSL_ALPHA_HPP
#define COORD_ASCENT_SSL_ALPHA_HPP

#include "co_utils.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <type_traits>

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
compute_p_star_j(T l0j, T l1j, T beta_j, T theta) {
    return 1 / (1 + ((1 - theta) / theta) * (l0j / l1j) * exp(-std::abs(beta_j) * (l0j - l1j)));
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
compute_lambda_star_j(T l0j, T l1j, T beta_j, T theta) {
    T p_star_beta_j = compute_p_star_j(l0j, l1j, beta_j, theta);
    return l1j * p_star_beta_j + l0j * (1 - p_star_beta_j);
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
compute_g_j(T n, T l0j, T l1j, T beta_j, T theta, T var) {
    T p_star_beta_j = compute_p_star_j(l0j, l1j, beta_j, theta);
    T lambda_star_beta_j = compute_lambda_star_j(l0j, l1j, beta_j, theta);

    return (lambda_star_beta_j - l1j) * (lambda_star_beta_j - l1j) + (2 * n / var) * std::log(p_star_beta_j);
}


template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
compute_mse_alpha(int c_size, T* beta, T* std_beta, T* q, T lam_min) {
    return (1.0- 2.0* blas_dot(std_beta, beta, c_size))
            + blas_dot(q, beta, c_size)
            + (1.0+lam_min) * blas_dot(beta, beta, c_size);
}

/* ------------------------------------------------------------------------ */
// Main Coordinate Ascent functions

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
update_var_alpha(int c_size, T n, T* beta, T* std_beta, T* q, T lam_min) {
    return (n / (n + 2.0)) * compute_mse_alpha(c_size, beta, std_beta, q, lam_min);
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
update_delta_j(T n,
             T l0j,
             T l1_j,
             T theta,
             T var) {

    T p_star_0 = compute_p_star_j(l0j, l1_j, static_cast<T>(0), theta);
    T lambda_star_0 = compute_lambda_star_j(l0j, l1_j, static_cast<T>(0), theta);

    T g_0 = (lambda_star_0 - l1_j) * (lambda_star_0 - l1_j) + (2 * n / var) * std::log(p_star_0);

    if (g_0 > 0) {
        return sqrt(2 * n * var * log(1 / p_star_0)) + var * l1_j;
    } else {
        return var * lambda_star_0;
    }
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
update_delta_vec(int c_size,
                             T n,
                             T theta,
                             T var,
                             T *l0_vec,
                             T *l1_vec,
                             T *delta,
                             int threads){
    #ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(threads)
    #endif
    for (int j = 0; j < c_size; ++j) {
        delta[j] = update_delta_j(n, l0_vec[j], l1_vec[j], theta, var);
    }
}

template <typename T, typename U, typename I>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value && std::is_integral<I>::value, void>::type
update_beta_ssl_alpha(int c_size,
    T n,
    T *l0_vec,
    T *l1_vec,
    T a,
    T b,
    int update_freq,
    T lam_min,
    int *ld_left_bound,
    I *ld_indptr,
    U *ld_data,
    T *std_beta,
    T *beta,
    T *beta_diff,
    T *theta,
    T *delta,
    T *var,
    T *q,
    T *n_per_snp,
    T init_var,
    T min_var, 
    bool u_var,
    T dq_scale,
    int *p_gamma,
    int threads,
    bool low_memory) {

    int start, end, update_count = 0, p_gamma_diff = 0;
    I ld_start, ld_end;

    T inv_abp = a + b + c_size;
    T curr_theta = theta[0];
    T curr_var = var[0];

    #ifdef _OPENMP
        #pragma omp parallel for reduction(+:p_gamma_diff) private(ld_start, ld_end, start, end, update_count, curr_theta, curr_var) schedule(static) num_threads(threads)
    #endif
    for (int j = 0; j < c_size; ++j) {

        ld_start = ld_indptr[j];
        ld_end = ld_indptr[j + 1];
        start = ld_left_bound[j];
        end = start + (ld_end - ld_start);

        T curr_beta = beta[j];
        T l0j = l0_vec[j];
        T l1j = l1_vec[j];

        delta[j] = update_delta_j(n, l0j, l1j, curr_theta, curr_var);

        T abs_zj = std::abs(n_per_snp[j] * (std_beta[j] - q[j]));
        T lambda_star_val = compute_lambda_star_j(l0j, l1j, curr_beta, theta[0]);

        // beta[j] = (1 / (n*(1+lam_min))) * max((abs(zj) - var*lambda_star_val), 0) * np.sign(zj) * (abs(zj) > delta)
        T sft_diff = abs_zj - var[0] * lambda_star_val;
        T inv_n_lam_min = 1 / (n_per_snp[j] * (1 + lam_min));
        beta[j] = inv_n_lam_min * sft_diff *(sft_diff>0) * (abs_zj > delta[j]) * (-2*(std_beta[j] < q[j]) + 1);

        p_gamma_diff += (beta[j] != 0 && curr_beta == 0) - (beta[j] == 0 && curr_beta != 0);

        // difference between updated and prev beta
        beta_diff[j] = beta[j] - curr_beta;

        // no difference if beta isn't updated
        if (beta_diff[j] != 0) {
            /* Update the q-factors for variants that are in LD with variant j */
            blas_axpy(q + start, ld_data + ld_start, dq_scale*beta_diff[j], end - start);

            if (!low_memory){
                q[j] = q[j] - beta_diff[j];
            }

            update_count++;
        }


        if (update_count == update_freq) {

            // update theta and delta
            curr_theta = static_cast<T>(a + p_gamma[0] + p_gamma_diff) / inv_abp;

            if (u_var){
                curr_var = update_var_alpha(c_size, n, beta, std_beta, q, lam_min);

                if (curr_var < min_var){
                    curr_var = init_var;
                }
            }

            update_count = 0;
        }
    }

    p_gamma[0] += p_gamma_diff;
    theta[0] = static_cast<T>(a + p_gamma[0]) / inv_abp;

    if (u_var){
        var[0] = update_var_alpha(c_size, n, beta, std_beta, q, lam_min);

        if (var[0] < min_var){
            var[0] = init_var;
        }
    }

    update_delta_vec(c_size, n, theta[0], var[0], l0_vec, l1_vec, delta, threads);

    if (low_memory) {
        update_q_factor(c_size, ld_left_bound, ld_indptr, ld_data, &beta_diff[0], q, dq_scale, threads);
    }
}

#endif
