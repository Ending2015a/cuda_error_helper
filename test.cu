#include <iostream>
#include "error_handler.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

int main()
{
    // error: invalid value
    double *d = 0;
    double *a = 0;
    error_check(cudaMemcpy(d, a, sizeof(int), cudaMemcpyHostToDevice));

    return 0;
}
