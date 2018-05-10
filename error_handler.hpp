#ifndef __ERROR_HANDLER_HPP__
#define __ERROR_HANDLER_HPP__

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include "meta_utils.hpp"

#define error_check(err) error_handler(err, __LINE__)


// cuBLAS error message
const char *error_message(cublasStatus_t err)
{
    switch(err)
    {
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "library not inirialized";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "failed to allocate resources";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "invalid value";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "device does not support double/half precision";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "failed to launch on the GPU";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "access gpu error";
    }
    return "other error occurred";
}


// cuSPARSE error message
const char *error_message(cusparseStatus_t err)
{
    switch(err)
    {
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "library not initialized";
        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "failed to allocate resources";
        case CUSPARSE_STATUS_INVALID_VALUE:
            return "invalid value";
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "device does not support double/half precision";
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "failed to launch on the GPU";
    }
    return "other error occurred";
}


// cuda error message
const char *error_message(cudaError_t err)
{
    switch(err)
    {
        case cudaErrorMemoryAllocation:
            return "failed to allocate resources";
        case cudaErrorInitializationError:
            return "failed to initialize cuda";
        case cudaErrorInvalidValue:
            return "invalid value";
    }
    return "other error occurred";
}

template<typename errT>
void error_handler(errT err, int line)
{
    static_assert(zex::is_any<errT, cudaError_t, cublasStatus_t, cusparseStatus_t>::value,
                "Error type must be one of the types: cudaError_t, cublasStatus_t or cusparseStatus_t");

    using tidx = zex::type_index<errT, cudaError_t, cublasStatus_t, cusparseStatus_t>;

    if(err != zex::type_case<tidx::value, zex::tv_pair<cudaError_t, cudaSuccess>,
                                          zex::tv_pair<cublasStatus_t, CUBLAS_STATUS_SUCCESS>,
                                          zex::tv_pair<cusparseStatus_t, CUSPARSE_STATUS_SUCCESS>>::type::value)
    {
        std::cout << (tidx::value==0 ? "[cuda ERROR]" :
                            tidx::value==1 ? "[cuBLAS ERROR]":"[cuSPARSE ERROR]")
                  << " in line " << line << ": " << error_message(err) << std::endl;
        exit(1);
    }
}

#endif
