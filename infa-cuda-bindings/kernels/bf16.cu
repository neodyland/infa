#ifndef BF16_CU
#define BF16_CU
#include <cuda_bf16.h>
#include <curand_kernel.h>

extern "C" __global__ void sqrt_forward_bf16_kernel(__nv_bfloat16 *ptr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        ptr[idx] = __float2bfloat16(sqrtf(__bfloat162float(ptr[idx])));
    }
}

extern "C" __global__ void fill_bf16_kernel(__nv_bfloat16 *output_ptr, __nv_bfloat16 value, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = value;
    }
}

extern "C" __global__ void to_f32_bf16_kernel(float *output_ptr, __nv_bfloat16 *input_ptr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = __bfloat162float(input_ptr[idx]);
    }
}

extern "C" __global__ void from_f32_bf16_kernel(__nv_bfloat16 *output_ptr, float *input_ptr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = __float2bfloat16(input_ptr[idx]);
    }
}

extern "C" __global__ void add_forward_bf16_kernel(__nv_bfloat16 *output_ptr, __nv_bfloat16 *input_ptr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = __float2bfloat16(__bfloat162float(output_ptr[idx]) + __bfloat162float(input_ptr[idx]));
    }
}

extern "C" __global__ void sub_forward_bf16_kernel(__nv_bfloat16 *output_ptr, __nv_bfloat16 *input_ptr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = __float2bfloat16(__bfloat162float(output_ptr[idx]) - __bfloat162float(input_ptr[idx]));
    }
}

extern "C" __global__ void mul_forward_bf16_kernel(__nv_bfloat16 *output_ptr, __nv_bfloat16 *input_ptr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = __float2bfloat16(__bfloat162float(output_ptr[idx]) * __bfloat162float(input_ptr[idx]));
    }
}

extern "C" __global__ void div_forward_bf16_kernel(__nv_bfloat16 *output_ptr, __nv_bfloat16 *input_ptr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = __float2bfloat16(__bfloat162float(output_ptr[idx]) / __bfloat162float(input_ptr[idx]));
    }
}

extern "C" __global__ void add_scalar_forward_bf16_kernel(__nv_bfloat16 *output_ptr, float scalar, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = __float2bfloat16(__bfloat162float(output_ptr[idx]) + scalar);
    }
}

extern "C" __global__ void sub_scalar_forward_bf16_kernel(__nv_bfloat16 *output_ptr, float scalar, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = __float2bfloat16(__bfloat162float(output_ptr[idx]) - scalar);
    }
}

extern "C" __global__ void mul_scalar_forward_bf16_kernel(__nv_bfloat16 *output_ptr, float scalar, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = __float2bfloat16(__bfloat162float(output_ptr[idx]) * scalar);
    }
}

extern "C" __global__ void div_scalar_forward_bf16_kernel(__nv_bfloat16 *output_ptr, float scalar, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_ptr[idx] = __float2bfloat16(__bfloat162float(output_ptr[idx]) / scalar);
    }
}

#endif