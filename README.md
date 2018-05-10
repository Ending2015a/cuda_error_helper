# Cuda Error Helper

## Usage
1. #include "error_handler.hpp"
2. Simply wrap your cuda/cuBLAS/cuSPARSE functions with error_check()
3. That's it.

## Example
```clike=
#include <iostream>
#include <cuda_runtime.h>
#include "error_handler.hpp"

int main()
{
    //error: invalid value
    double *a = 0, *b = 0;
    error_check(cudaMemcpy(a, b, 10, cudaMemcpyHostToDevice));
    
    return 0;
}
```

This example will print out following error message:
```
[cuda ERROR] error: invalid value
 in line [9] in func: cudaMemcpy(a, b, 10, cudaMemcpyHostToDevice);
```
