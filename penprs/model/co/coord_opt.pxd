from cython cimport floating
cimport numpy as cnp

# --------------------------------------------------
# Define fused data types:

ctypedef fused np_floating:
    cnp.float32_t
    cnp.float64_t

ctypedef fused int_dtype:
    cnp.int32_t
    cnp.int64_t

ctypedef fused noncomplex_numeric:
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cnp.float32_t
    cnp.float64_t

# --------------------------------------------------

cdef void cpp_blas_axpy(floating[::1] v1, floating[::1] v2, floating alpha) noexcept nogil
cdef floating cpp_blas_dot(floating[::1] v1, floating[::1] v2) noexcept nogil

cpdef floating cpp_update_delta(floating n,
                        floating l0,
                        floating l1,
                        floating theta,
                        floating var) noexcept nogil

cpdef void cpp_update_delta_alpha_vec(
                        floating[::1] delta,   
                        floating n,            
                        floating theta,      
                        floating var,          
                        floating[::1] l0_vec,  
                        floating[::1] l1_vec,  
                        int threads) noexcept nogil

cpdef floating cpp_update_var(floating n,
                        floating[::1] beta,
                        floating[::1] std_beta,
                        floating[::1] q,
                        floating lam_min) noexcept nogil

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
                        bint low_memory) noexcept nogil

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
                        bint low_memory) noexcept nogil


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
                        bint low_memory) noexcept nogil
