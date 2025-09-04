# distutils: language = c++

from cython cimport floating

# ------------------------------------------
cdef extern from "co_utils.hpp" nogil:

    bint blas_supported() noexcept nogil
    bint omp_supported() noexcept nogil

    void blas_axpy[T](T* y, T* x, T alpha, int size) noexcept nogil
    T blas_dot[T](T* x, T* y, int size) noexcept nogil

cdef extern from "coord_ascent_ssl.hpp" nogil:

    T update_delta[T](T n,
             T l0,
             T l1,
             T theta,
             T var) noexcept nogil
    
    T update_var[T](int c_size,
            T n,
            T *beta,
            T *std_beta,
            T *q,
            T lam_min) noexcept nogil

    void update_beta_ssl[T, U, I](int c_size,
            T n,
            T l0,
            T l1,
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
            bint u_var,
            T dq_scale,
            int *p_gamma,
            int threads,
            bint low_memory) noexcept nogil

cdef extern from "coord_ascent_ssl_alpha.hpp" nogil:
    void update_delta_vec[T](int c_size,
                            T n,
                            T theta,
                            T var,
                            T* l0_vec,
                            T* l1_vec,
                            T* delta,
                            int threads) noexcept nogil

    void update_beta_ssl_alpha[T, U, I](int c_size,
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
            bint u_var,
            T dq_scale,
            int *p_gamma,
            int threads,
            bint low_memory) noexcept nogil

cdef extern from "coord_ascent_lasso.hpp" nogil:

    void update_beta_lasso[T, U, I](int c_size,
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
            bint low_memory) noexcept nogil

# ------------------------------------------

cpdef check_blas_support():
    return blas_supported()

cpdef check_omp_support():
    return omp_supported()

cdef void cpp_blas_axpy(floating[::1] v1, floating[::1] v2, floating alpha) noexcept nogil:
    """v1 := v1 + alpha * v2"""
    cdef int size = v1.shape[0]
    blas_axpy(&v1[0], &v2[0], alpha, size)

cdef floating cpp_blas_dot(floating[::1] v1, floating[::1] v2) noexcept nogil:
    """v1 := v1.Tv2"""
    cdef int size = v1.shape[0]
    return blas_dot(&v1[0], &v2[0], size)


# ---------------- SSL Coordinate Ascent Functions ----------------

cpdef floating cpp_update_delta(floating n,
                        floating l0,
                        floating l1,
                        floating theta,
                        floating var) noexcept nogil:
        return update_delta(n,
                    l0,
                    l1,
                    theta,
                    var)
                    
cpdef void cpp_update_delta_alpha_vec(
                        floating[::1] delta,
                        floating n,
                        floating theta,
                        floating var,
                        floating[::1] l0_vec,
                        floating[::1] l1_vec,
                        int threads) noexcept nogil:
    cdef int c_size = delta.shape[0]
    update_delta_vec(c_size,
                     n,
                     theta,
                     var,
                     &l0_vec[0],
                     &l1_vec[0],
                     &delta[0],
                     threads)


cpdef floating cpp_update_var(floating n,
                        floating[::1] beta,
                        floating[::1] std_beta,
                        floating[::1] q,
                        floating lam_min) noexcept nogil:
        return update_var(beta.shape[0],
                    n,
                    &beta[0],
                    &std_beta[0],
                    &q[0],
                    lam_min)

cpdef void cpp_update_beta_ssl(int[::1] ld_left_bound,
                        int_dtype[::1] ld_indptr,
                        noncomplex_numeric[::1] ld_data,
                        floating[::1] std_beta,
                        floating[::1] beta,
                        floating[::1] beta_diff,
                        floating[::1] q,
                        floating[::1] n_per_snp,
                        int[::1] p_gamma,
                        floating[::1] theta,
                        floating[::1] delta,
                        floating[::1] var,
                        floating n,
                        floating l0,
                        floating l1,
                        floating a,
                        floating b,
                        int update_freq,
                        floating lam_min,
                        floating init_var,
                        floating min_var,
                        bint u_var,
                        floating dq_scale,
                        int threads,
                        bint low_memory) noexcept nogil:

        update_beta_ssl(beta.shape[0],
                    n,
                    l0,
                    l1,
                    a,
                    b,
                    update_freq,
                    lam_min,
                    &ld_left_bound[0],
                    &ld_indptr[0],
                    &ld_data[0],
                    &std_beta[0],
                    &beta[0],
                    &beta_diff[0],
                    &theta[0],
                    &delta[0],
                    &var[0],
                    &q[0],
                    &n_per_snp[0],
                    init_var,
                    min_var,
                    u_var,
                    dq_scale,
                    &p_gamma[0],
                    threads,
                    low_memory)

cpdef void cpp_update_beta_ssl_alpha(int[::1] ld_left_bound,
                        int_dtype[::1] ld_indptr,
                        noncomplex_numeric[::1] ld_data,
                        floating[::1] std_beta,
                        floating[::1] beta,
                        floating[::1] beta_diff,
                        floating[::1] q,
                        floating[::1] n_per_snp,
                        int[::1] p_gamma,
                        floating[::1] theta,
                        floating[::1] delta,
                        floating[::1] var,
                        floating n,
                        floating[::1] l0_vec,
                        floating[::1] l1_vec,
                        floating a,
                        floating b,
                        int update_freq,
                        floating lam_min,
                        floating init_var,
                        floating min_var,
                        bint u_var,
                        floating dq_scale,
                        int threads,
                        bint low_memory) noexcept nogil:

        update_beta_ssl_alpha(beta.shape[0],
                    n,
                    &l0_vec[0],
                    &l1_vec[0],
                    a,
                    b,
                    update_freq,
                    lam_min,
                    &ld_left_bound[0],
                    &ld_indptr[0],
                    &ld_data[0],
                    &std_beta[0],
                    &beta[0],
                    &beta_diff[0],
                    &theta[0],
                    &delta[0],
                    &var[0],
                    &q[0],
                    &n_per_snp[0],
                    init_var,
                    min_var,
                    u_var,
                    dq_scale,
                    &p_gamma[0],
                    threads,
                    low_memory)

# ---------------- Lasso Coordinate Descent functions ----------------


cpdef void cpp_update_beta_lasso(int[::1] ld_left_bound,
                        int_dtype[::1] ld_indptr,
                        noncomplex_numeric[::1] ld_data,
                        floating[::1] std_beta,
                        floating[::1] beta,
                        floating[::1] beta_diff,
                        floating[::1] q,
                        floating[::1] n_per_snp,
                        floating lam,
                        floating lam_min,
                        floating dq_scale,
                        int threads,
                        bint low_memory) noexcept nogil:

        update_beta_lasso(beta.shape[0],
                    &ld_left_bound[0],
                    &ld_indptr[0],
                    &ld_data[0],
                    &std_beta[0],
                    &beta[0],
                    &beta_diff[0],
                    &q[0],
                    &n_per_snp[0],
                    lam,
                    lam_min,
                    dq_scale,
                    threads,
                    low_memory)
