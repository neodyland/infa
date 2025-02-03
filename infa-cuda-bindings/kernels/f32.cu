#ifndef f32_CU
#define f32_CU
#include <curand_kernel.h>

extern "C" __global__ void sqrt_forward_f32_kernel(float *ptr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        ptr[idx] = sqrtf(ptr[idx]);
    }
}

extern "C" __global__ void fill_f32_kernel(float *output_ptr, float value, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = value;
    }
}

extern "C" __global__ void fit_range_f32_kernel(float *ptr, int size, float min, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        ptr[idx] = ptr[idx] * scale + min;
    }
}


extern "C" __global__ void add_f32_kernel(float *output_ptr, float *input_ptr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = output_ptr[idx] + input_ptr[idx];
    }
}

extern "C" __global__ void sub_forward_f32_kernel(float *output_ptr, float *input_ptr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = output_ptr[idx] - input_ptr[idx];
    }
}

extern "C" __global__ void mul_forward_f32_kernel(float *output_ptr, float *input_ptr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = output_ptr[idx] * input_ptr[idx];
    }
}

extern "C" __global__ void div_forward_f32_kernel(float *output_ptr, float *input_ptr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = output_ptr[idx] / input_ptr[idx];
    }
}

extern "C" __global__ void add_scalar_forward_f32_kernel(float *output_ptr, float scalar, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = output_ptr[idx] + scalar;
    }
}

extern "C" __global__ void sub_scalar_forward_f32_kernel(float *output_ptr, float scalar, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = output_ptr[idx] - scalar;
    }
}

extern "C" __global__ void mul_scalar_forward_f32_kernel(float *output_ptr, float scalar, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = output_ptr[idx] * scalar;
    }
}

extern "C" __global__ void div_scalar_forward_f32_kernel(float *output_ptr, float scalar, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = output_ptr[idx] / scalar;
    }
}

#endif