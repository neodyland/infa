use infa_core::Free;

use crate::{
    check_cu_status, check_cublas_status, check_curand_status, check_status, cuInit,
    cublasCreate_v2, cublasDestroy_v2, cublasHandle_t, cublasSetStream_v2, cudaStreamCreate,
    cudaStreamDestroy, cudaStreamSynchronize, cudaStream_t, curandCreateGenerator,
    curandDestroyGenerator, curandGenerator_t, curandRngType_t, curandSetPseudoRandomGeneratorSeed,
};
use std::mem::MaybeUninit;

#[derive(Clone)]
pub struct Container {
    stream: cudaStream_t,
    rand: curandGenerator_t,
    blas: cublasHandle_t,
}

impl Container {
    pub fn new(seed: u64) -> anyhow::Result<Self> {
        check_cu_status(unsafe { cuInit(0) })?;
        let stream = unsafe {
            let mut stream = MaybeUninit::uninit();
            check_status(cudaStreamCreate(stream.as_mut_ptr()))?;
            stream.assume_init()
        };
        let rand = unsafe {
            let mut rand = MaybeUninit::uninit();
            check_curand_status(curandCreateGenerator(
                rand.as_mut_ptr(),
                curandRngType_t::CURAND_RNG_PSEUDO_DEFAULT,
            ))?;
            let rand = rand.assume_init();
            check_curand_status(curandSetPseudoRandomGeneratorSeed(rand, seed))?;
            rand
        };
        let blas = unsafe {
            let mut blas = MaybeUninit::uninit();
            check_cublas_status(cublasCreate_v2(blas.as_mut_ptr()))?;
            blas.assume_init()
        };
        check_cublas_status(unsafe { cublasSetStream_v2(blas, stream) })?;
        Ok(Self { stream, rand, blas })
    }
}

impl ContainerTrait for Container {
    fn container(&self) -> &Container {
        self
    }
}

pub trait ContainerTrait {
    fn stream(&self) -> cudaStream_t {
        self.container().stream
    }
    fn rand(&self) -> curandGenerator_t {
        self.container().rand
    }
    fn blas(&self) -> cublasHandle_t {
        self.container().blas
    }
    fn container(&self) -> &Container;
    fn sync(&self) -> anyhow::Result<()> {
        check_status(unsafe { cudaStreamSynchronize(self.stream()) })?;
        Ok(())
    }
}

impl Free for Container {
    fn free(self) -> anyhow::Result<()> {
        check_curand_status(unsafe { curandDestroyGenerator(self.rand) })?;
        check_status(unsafe { cudaStreamDestroy(self.stream) })?;
        check_cublas_status(unsafe { cublasDestroy_v2(self.blas) })?;
        Ok(())
    }
}
