#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <type_traits>

// Check for and include `cblas`:
#ifdef HAVE_CBLAS
    #include <cblas.h>
#endif

// Check for and include `omp`:
#ifdef _OPENMP
    #include <omp.h>
#endif

/* ----------------------------- */
// Helper system-related functions to check for BLAS and OpenMP support

bool omp_supported() {
    /* Check if OpenMP is supported by examining compiler flags. */
    #ifdef _OPENMP
        return true;
    #else
        return false;
    #endif
}

bool blas_supported() {
    /* Check if BLAS is supported by examining compiler flags. */
    #ifdef HAVE_CBLAS
        return true;
    #else
        return false;
    #endif
}

/* ------------------------------ */
// Dot product functions

// Define a function pointer for the dot product functions `dot` and `blas_dot`:
template <typename T, typename U>
using dot_func_pt = typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, T>::type (*)(T *, U *, int);

/* * * * * */

template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, T>::type
dot(T *x, U *y, int size) {
    /* Perform dot product between two vectors x and y, each of length `size`

    :param x: Pointer to the first element of the first vector
    :param y: Pointer to the first element of the second vector
    :param size: Length of the vectors

    */

    T s = 0.;

    #ifdef _OPENMP
        #ifndef _WIN32
            #pragma omp simd
        #endif
    #endif
    for (int i = 0; i < size; ++i) {
        s += x[i] * static_cast<T>(y[i]);
    }
    return s;
}

/* * * * * */

template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, T>::type
blas_dot(T *x, U *y, int size) {
    /*
        Use BLAS (if available) to perform dot product
        between two vectors x and y, each of length `size`.

        :param x: Pointer to the first element of the first vector
        :param y: Pointer to the first element of the second vector
        :param size: Length of the vectors
    */

    #ifdef HAVE_CBLAS
        int incx = 1;
        int incy = 1;

        if constexpr (std::is_same<T, float>::value) {
            if constexpr (std::is_same<U, float>::value) {
                return cblas_sdot(size, x, incx, y, incy);
            } else {
                // Handles the case where y is any data type that is not a float:
                std::vector<float> y_float(size);
                std::transform(y, y + size, y_float.begin(), [](U val) { return static_cast<float>(val); });
                return cblas_sdot(size, x, incx, y_float.data(), incy);
            }
        } else if constexpr (std::is_same<T, double>::value) {
            if constexpr (std::is_same<U, double>::value) {
                return cblas_ddot(size, x, incx, y, incy);
            } else {
                // Handles the case where y is any data type that is not a double:
                std::vector<double> y_double(size);
                std::transform(y, y + size, y_double.begin(), [](U val) { return static_cast<double>(val); });
                return cblas_ddot(size, x, incx, y_double.data(), incy);
            }
        }
    #else
        return dot(x, y, size);
    #endif
}

/* * * * * */

// Define a function pointer for the axpy functions `axpy` and `blas_axpy`:
template <typename T, typename U>
using axpy_func_pt = typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, void>::type (*)(T *, U *, T, int);

/* * * * * */

template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, void>::type
axpy(T *x, U *y, T alpha, int size) {
    /*
        Perform axpy operation on two vectors x and y, each of length `size`.
       axpy is a standard linear algebra operation that performs
       element-wise addition and multiplication:
       x := x + a*y.
    */

    #ifdef _OPENMP
        #ifndef _WIN32
            #pragma omp simd
        #endif
    #endif
    for (int i = 0; i < size; ++i) {
        x[i] += static_cast<T>(y[i]) * alpha;
    }
}

/* * * * * */

template <typename T, typename U>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value, void>::type
blas_axpy(T *y, U *x, T alpha, int size) {
    /*
        Use BLAS (if available) to perform axpy operation on two vectors x and y,
        each of length `size`.
       axpy is a standard linear algebra operation that performs
       element-wise addition and multiplication:
       x := x + a*y.
    */

    #ifdef HAVE_CBLAS
        int incx = 1;
        int incy = 1;

        if constexpr (std::is_same<T, float>::value) {
            if constexpr (std::is_same<U, float>::value) {
                cblas_saxpy(size, alpha, x, incx, y, incy);
            } else {
                // Handles the case where x is any data type that is not a float:
                std::vector<float> x_float(size);
                std::transform(x, x + size, x_float.begin(), [](U val) { return static_cast<float>(val); });
                cblas_saxpy(size, alpha, x_float.data(), incx, y, incy);
            }
        } else if constexpr (std::is_same<T, double>::value) {
            if constexpr (std::is_same<U, double>::value) {
                cblas_daxpy(size, alpha, x, incx, y, incy);
            } else {
                // Handles the case where x is any data type that is not a float:
                std::vector<double> x_double(size);
                std::transform(x, x + size, x_double.begin(), [](U val) { return static_cast<double>(val); });
                cblas_daxpy(size, alpha, x_double.data(), incx, y, incy);
            }
        }
    #else
        axpy(y, x, alpha, size);
    #endif
}

/* ------------------------------------------------------------------------ */
// Coordinate Ascent helper functions

template <typename T, typename U, typename I>
typename std::enable_if<std::is_floating_point<T>::value && std::is_arithmetic<U>::value && std::is_integral<I>::value, void>::type
update_q_factor(int c_size,
                 int *ld_left_bound,
                 I *ld_indptr,
                 U *ld_data,
                 T *beta,
                 T *q,
                 T dq_scale,
                 int threads) {
    /*
        Compute or update q factor, defined as the result of a dot product between the
        Linkage-disequilibrium matrix (LD) and the current value of beta (effect sizes).
        The definition of the q-factor excludes the diagonal elements.
    */

    I ld_start, ld_end;

    #ifdef _OPENMP
        #pragma omp parallel for private(ld_start, ld_end) schedule(static) num_threads(threads)
    #endif
    for (int j = 0; j < c_size; ++j) {

        ld_start = ld_indptr[j];
        ld_end = ld_indptr[j + 1];

        q[j] += dq_scale*blas_dot(beta + ld_left_bound[j], ld_data + ld_start, ld_end - ld_start);
    }
}

#endif
