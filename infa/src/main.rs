use infa_core::{FloatOps, Free};
use infa_cuda_bindings::{Container, ContainerTrait, CudaBFloat16Tensor};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let cnt = Container::new(1234)?;
    let a = CudaBFloat16Tensor::new(&[2, 48, 32], &cnt)?;
    a.fill_(2.0)?;
    let b = CudaBFloat16Tensor::new(&[2, 32, 16], &cnt)?;
    b.fill_(3.0)?;
    let start = Instant::now();
    for _ in 0..100 {
        let c = a.matmul(&b, None)?;
        c.sync()?;
        c.free()?;
    }
    let duration = start.elapsed();
    println!("Time: {:?}", duration);
    a.free()?;
    b.free()?;
    cnt.free()?;
    Ok(())
}
