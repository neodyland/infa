use std::ffi::c_void;

use crate::bindings::{cublasStatus_t, cudaError_enum, cudaError_t, curandStatus_t};

pub fn check_status(status: cudaError_t) -> anyhow::Result<()> {
    if status != cudaError_t::cudaSuccess {
        anyhow::bail!("{:?}", status)
    }
    Ok(())
}

pub fn check_cublas_status(status: cublasStatus_t) -> anyhow::Result<()> {
    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        anyhow::bail!("{:?}", status)
    }
    Ok(())
}

pub fn check_cu_status(status: cudaError_enum) -> anyhow::Result<()> {
    if status != cudaError_enum::CUDA_SUCCESS {
        unsafe {
            let mut str = std::mem::MaybeUninit::uninit();
            crate::bindings::cuGetErrorString(status, str.as_mut_ptr());
            println!(
                "{:?}",
                std::ffi::CString::from_raw(str.assume_init() as *mut i8)
            );
        };
        anyhow::bail!("{:?}", status)
    }
    Ok(())
}

pub fn check_curand_status(status: curandStatus_t) -> anyhow::Result<()> {
    if status != curandStatus_t::CURAND_STATUS_SUCCESS {
        anyhow::bail!("{:?}", status)
    }
    Ok(())
}

#[inline(always)]
pub fn cast_ptr<T>(ptr: &T) -> *mut c_void {
    ptr as *const T as *mut c_void
}

// Macro to launch a kernel
#[macro_export]
macro_rules! launch_kernel(
    ($kernel:ident, $stream:expr, $grid_size:expr, $block_size:expr, $($args:expr),*) => {
        {
            use crate::{check_status, cudaLaunchKernel, cast_ptr,};
            check_status(unsafe {
                cudaLaunchKernel(
                    $kernel as *const c_void,
                    $grid_size,
                    $block_size,
                    [
                        $(cast_ptr(&$args),)*
                    ].as_mut_ptr(),
                    0,
                    $stream.stream(),
                )
            })
        }
    };
);
