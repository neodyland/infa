use std::{env, fs::File, path::Path};

fn main() {
    let cwd = env::current_dir().unwrap();
    println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64/");
    println!("cargo:rustc-link-lib=dylib=cudnn");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=curand");
    let h = cwd.join("wrapper.h");
    let h = h.to_str().unwrap();
    println!("cargo:rerun-if-changed={}", h);
    println!(
        "cargo:rerun-if-changed={}",
        cwd.join("kernels").to_str().unwrap()
    );
    let bindings = bindgen::Builder::default()
        .header(h)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .size_t_is_usize(true)
        .allowlist_type("^cu.*")
        .allowlist_function("^cu.*")
        .allowlist_var("^cu.*")
        .layout_tests(false)
        .clang_arg("-I/usr/include/")
        .clang_arg("-I/usr/local/cuda/include")
        .generate_comments(false)
        .generate()
        .expect("Unable to generate bindings");
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir);
    let writer = File::create(dest_path.join("bindings.rs")).unwrap();
    bindings
        .write(Box::new(writer))
        .expect("Couldn't write bindings!");
    std::process::Command::new("nvcc")
        .args(&[
            "-c",
            "./kernels/kernel.cu",
            "-o",
            dest_path.join("libfusedkernel.o").to_str().unwrap(),
            "-O3",
            "--gpu-code=sm_80",
            "-arch=compute_80",
            "-Xcompiler",
            "-fPIC",
            "--use_fast_math",
        ])
        .output()
        .expect("Failed to compile kernel.cu");
    std::process::Command::new("ar")
        .args(&[
            "crus",
            dest_path.join("libfusedkernel.a").to_str().unwrap(),
            dest_path.join("libfusedkernel.o").to_str().unwrap(),
        ])
        .output()
        .expect("Failed to archive kernel.o");
    println!("cargo:rustc-link-search=native={}", dest_path.display());
    println!("cargo:rustc-link-lib=static=fusedkernel");
}
