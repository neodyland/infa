use infa_core::{Clone, Display, FloatOps, Free};
use std::{
    ffi::{c_int, c_void},
    mem::MaybeUninit,
};

use crate::{
    check_cublas_status, check_curand_status, check_status, cublasComputeType_t, cublasGemmAlgo_t,
    cublasGemmBatchedEx, cublasOperation_t, cudaDataType_t, cudaFree, cudaMallocAsync,
    cudaMemcpyAsync, cudaMemcpyKind, curandGenerateUniform, dim3, launch_kernel, Container,
    ContainerTrait,
};

extern "C" {
    fn sqrt_forward_f32_kernel(ptr: *mut f32, size: c_int) -> c_void;
    fn fill_f32_kernel(ptr: *mut f32, v: f32, size: c_int) -> c_void;
    fn add_forward_f32_kernel(ptr: *mut f32, a: *mut f32, size: c_int) -> c_void;
    fn sub_forward_f32_kernel(ptr: *mut f32, a: *mut f32, size: c_int) -> c_void;
    fn mul_forward_f32_kernel(ptr: *mut f32, a: *mut f32, size: c_int) -> c_void;
    fn div_forward_f32_kernel(ptr: *mut f32, a: *mut f32, size: c_int) -> c_void;
    fn add_scalar_forward_f32_kernel(ptr: *mut f32, a: f32, size: c_int) -> c_void;
    fn sub_scalar_forward_f32_kernel(ptr: *mut f32, a: f32, size: c_int) -> c_void;
    fn mul_scalar_forward_f32_kernel(ptr: *mut f32, a: f32, size: c_int) -> c_void;
    fn div_scalar_forward_f32_kernel(ptr: *mut f32, a: f32, size: c_int) -> c_void;
}

pub struct CudaFloat32Tensor {
    pub ptr: *mut c_void,
    pub size: usize,
    pub container: Container,
    pub shape: Vec<usize>,
}

impl CudaFloat32Tensor {
    pub fn new(shape: &[usize], c: &impl ContainerTrait) -> anyhow::Result<Self> {
        let mut ptr = MaybeUninit::uninit();
        let size = shape.iter().product();
        check_status(unsafe {
            cudaMallocAsync(ptr.as_mut_ptr(), size_of::<f32>() * size, c.stream())
        })?;
        Ok(Self {
            ptr: unsafe { ptr.assume_init() },
            size,
            container: c.container().clone(),
            shape: shape.to_vec(),
        })
    }
}

impl ContainerTrait for CudaFloat32Tensor {
    fn container(&self) -> &Container {
        &self.container
    }
}

impl Free for CudaFloat32Tensor {
    fn free(self) -> anyhow::Result<()> {
        check_status(unsafe { cudaFree(self.ptr) })?;
        Ok(())
    }
}

impl Clone for CudaFloat32Tensor {
    fn clone(&self) -> anyhow::Result<Self> {
        let mut ptr = MaybeUninit::uninit();
        check_status(unsafe {
            cudaMallocAsync(
                ptr.as_mut_ptr(),
                size_of::<f32>() * self.size,
                self.stream(),
            )
        })?;
        self.sync()?;
        check_status(unsafe {
            cudaMemcpyAsync(
                ptr.assume_init(),
                self.ptr,
                size_of::<f32>() * self.size,
                crate::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                self.stream(),
            )
        })?;
        Ok(Self {
            ptr: unsafe { ptr.assume_init() },
            size: self.size,
            container: self.container.clone(),
            shape: self.shape.clone(),
        })
    }
}

impl Display for CudaFloat32Tensor {
    fn display(&self) -> anyhow::Result<String> {
        let mut data = vec![0.0; self.size];
        check_status(unsafe {
            cudaMemcpyAsync(
                data.as_mut_ptr() as *mut c_void,
                self.ptr,
                size_of::<f32>() * self.size,
                crate::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                self.stream(),
            )
        })?;
        self.sync()?;
        Ok(format!("{:?}", data))
    }
}

impl FloatOps for CudaFloat32Tensor {
    fn sqrt_(&self) -> anyhow::Result<()> {
        let block_size = 256;
        let grid_size = (self.size as u32 + block_size - 1) / block_size;
        launch_kernel!(
            sqrt_forward_f32_kernel,
            self,
            dim3 {
                x: grid_size,
                y: 1,
                z: 1
            },
            dim3 {
                x: block_size,
                y: 1,
                z: 1
            },
            self.ptr,
            self.size as c_int
        )?;
        Ok(())
    }
    fn fill_(&self, value: f32) -> anyhow::Result<()> {
        let block_size = 256;
        let grid_size = (self.size as u32 + block_size - 1) / block_size;
        launch_kernel!(
            fill_f32_kernel,
            self,
            dim3 {
                x: grid_size,
                y: 1,
                z: 1
            },
            dim3 {
                x: block_size,
                y: 1,
                z: 1
            },
            self.ptr,
            value,
            self.size as c_int
        )?;
        Ok(())
    }
    fn rand_(&self) -> anyhow::Result<()> {
        check_curand_status(unsafe {
            curandGenerateUniform(self.container.rand(), self.ptr as *mut f32, self.size)
        })?;
        Ok(())
    }
    fn add_(&self, a: &Self) -> anyhow::Result<()> {
        launch_kernel!(
            add_forward_f32_kernel,
            self,
            dim3 {
                x: (self.size as u32 + 255) / 256,
                y: 1,
                z: 1
            },
            dim3 { x: 256, y: 1, z: 1 },
            self.ptr,
            a.ptr,
            self.size as c_int
        )?;
        Ok(())
    }
    fn sub_(&self, a: &Self) -> anyhow::Result<()> {
        launch_kernel!(
            sub_forward_f32_kernel,
            self,
            dim3 {
                x: (self.size as u32 + 255) / 256,
                y: 1,
                z: 1
            },
            dim3 { x: 256, y: 1, z: 1 },
            self.ptr,
            a.ptr,
            self.size as c_int
        )?;
        Ok(())
    }
    fn mul_(&self, a: &Self) -> anyhow::Result<()> {
        launch_kernel!(
            mul_forward_f32_kernel,
            self,
            dim3 {
                x: (self.size as u32 + 255) / 256,
                y: 1,
                z: 1
            },
            dim3 { x: 256, y: 1, z: 1 },
            self.ptr,
            a.ptr,
            self.size as c_int
        )?;
        Ok(())
    }
    fn div_(&self, a: &Self) -> anyhow::Result<()> {
        launch_kernel!(
            div_forward_f32_kernel,
            self,
            dim3 {
                x: (self.size as u32 + 255) / 256,
                y: 1,
                z: 1
            },
            dim3 { x: 256, y: 1, z: 1 },
            self.ptr,
            a.ptr,
            self.size as c_int
        )?;
        Ok(())
    }
    fn add_scalar_(&self, a: f32) -> anyhow::Result<()> {
        launch_kernel!(
            add_scalar_forward_f32_kernel,
            self,
            dim3 {
                x: (self.size as u32 + 255) / 256,
                y: 1,
                z: 1
            },
            dim3 { x: 256, y: 1, z: 1 },
            self.ptr,
            a,
            self.size as c_int
        )?;
        Ok(())
    }
    fn sub_scalar_(&self, a: f32) -> anyhow::Result<()> {
        launch_kernel!(
            sub_scalar_forward_f32_kernel,
            self,
            dim3 {
                x: (self.size as u32 + 255) / 256,
                y: 1,
                z: 1
            },
            dim3 { x: 256, y: 1, z: 1 },
            self.ptr,
            a,
            self.size as c_int
        )?;
        Ok(())
    }
    fn mul_scalar_(&self, a: f32) -> anyhow::Result<()> {
        launch_kernel!(
            mul_scalar_forward_f32_kernel,
            self,
            dim3 {
                x: (self.size as u32 + 255) / 256,
                y: 1,
                z: 1
            },
            dim3 { x: 256, y: 1, z: 1 },
            self.ptr,
            a,
            self.size as c_int
        )?;
        Ok(())
    }
    fn div_scalar_(&self, a: f32) -> anyhow::Result<()> {
        launch_kernel!(
            div_scalar_forward_f32_kernel,
            self,
            dim3 {
                x: (self.size as u32 + 255) / 256,
                y: 1,
                z: 1
            },
            dim3 { x: 256, y: 1, z: 1 },
            self.ptr,
            a,
            self.size as c_int
        )?;
        Ok(())
    }
    fn matmul(&self, other: &Self, bias: Option<Self>) -> anyhow::Result<Self> {
        if self.shape.len() != other.shape.len() {
            anyhow::bail!("Matmul requires two tensors with the same number of dimensions");
        }
        if self.shape.len() < 2 || other.shape.len() < 2 {
            anyhow::bail!("Matmul requires two tensors with at least 2 dimensions");
        }
        let mut arest = self.shape.clone();
        let ay = arest.pop().unwrap();
        let ax = arest.pop().unwrap();
        let mut brest = other.shape.clone();
        let by = brest.pop().unwrap();
        let bx = brest.pop().unwrap();
        if ay != bx {
            anyhow::bail!("Matmul requires two tensors with compatible shapes");
        }
        if arest != brest {
            anyhow::bail!("Matmul requires two tensors with same batch shapes");
        }
        let mut cshape = arest;
        let batch: usize = cshape.iter().product();
        cshape.push(ax);
        cshape.push(by);
        let is_bias = bias.is_some();
        let c = bias.unwrap_or(Self::new(&cshape, &self.container)?);
        let m = ax as i32;
        let n = by as i32;
        let k = ay as i32;
        let lda = m;
        let ldb = k;
        let ldc = m;
        let mut da = Vec::with_capacity(batch);
        let mut db = Vec::with_capacity(batch);
        let mut dc = Vec::with_capacity(batch);
        for i in 0..batch {
            let a = self.ptr;
            let b = other.ptr;
            let c = c.ptr;
            da.push(unsafe { a.add(i * ax * ay) } as *const c_void);
            db.push(unsafe { b.add(i * bx * by) } as *const c_void);
            dc.push(unsafe { c.add(i * ay * ax) });
        }
        let da = unsafe {
            let mut ptr = MaybeUninit::uninit();
            check_status(cudaMallocAsync(
                ptr.as_mut_ptr(),
                size_of::<*const c_void>() * batch,
                self.stream(),
            ))?;
            let ptr = ptr.assume_init();
            check_status(cudaMemcpyAsync(
                ptr,
                da.as_ptr() as *const _,
                size_of::<*const c_void>() * batch,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream(),
            ))?;
            ptr as *const _
        };
        let db = unsafe {
            let mut ptr = MaybeUninit::uninit();
            check_status(cudaMallocAsync(
                ptr.as_mut_ptr(),
                size_of::<*const c_void>() * batch,
                self.stream(),
            ))?;
            let ptr = ptr.assume_init();
            check_status(cudaMemcpyAsync(
                ptr,
                db.as_ptr() as *const _,
                size_of::<*const c_void>() * batch,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream(),
            ))?;
            ptr as *const _
        };
        let dc = unsafe {
            let mut ptr = MaybeUninit::uninit();
            check_status(cudaMallocAsync(
                ptr.as_mut_ptr(),
                size_of::<*mut c_void>() * batch,
                self.stream(),
            ))?;
            let ptr = ptr.assume_init();
            check_status(cudaMemcpyAsync(
                ptr,
                dc.as_ptr() as *const _,
                size_of::<*mut c_void>() * batch,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream(),
            ))?;
            ptr as *mut _
        };
        check_cublas_status(unsafe {
            cublasGemmBatchedEx(
                self.container.blas(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                m,
                n,
                k,
                &1.0_f32 as *const _ as *const c_void,
                da,
                cudaDataType_t::CUDA_R_32F,
                lda,
                db,
                cudaDataType_t::CUDA_R_32F,
                ldb,
                &if is_bias { 1.0 } else { 0.0_f32 } as *const _ as *const c_void,
                dc,
                cudaDataType_t::CUDA_R_32F,
                ldc,
                batch as i32,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        })?;
        check_status(unsafe { cudaFree(da as *mut c_void) })?;
        check_status(unsafe { cudaFree(db as *mut c_void) })?;
        check_status(unsafe { cudaFree(dc as *mut c_void) })?;
        Ok(c)
    }
}
