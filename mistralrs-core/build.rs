#[cfg(all(feature = "cuda", not(feature = "rocm")))]
const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");
#[cfg(all(feature = "cuda", not(feature = "rocm")))]
const FLASHINFER_GDN_COMMIT: &str = "28406af5b9134757acbd6bc44647fd00261d163f";
#[cfg(all(feature = "cuda", not(feature = "rocm")))]
const CUTLASS_COMMIT: &str = "7127592069c2fe01b041e174ba4345ef9b279671";
#[cfg(all(feature = "cuda", not(feature = "rocm")))]
const FLASHINFER_GDN_MIN_CUDA: u32 = 1208;
#[cfg(all(feature = "cuda", not(feature = "rocm")))]
const GDN_FP8_PRODUCER_MIN_CUDA: u32 = 1108;
#[cfg(all(feature = "cuda", not(feature = "rocm")))]
const CUDA_BUILD_ROOT_ENV: &str = "MISTRALRS_CUDA_BUILD_ROOT";

/// Opt in: MISTRALRS_CCACHE_HIPCC=1 (set in ~/.cargo/config.toml).
#[allow(dead_code)]
fn compiler_command(program: impl AsRef<std::path::Path>) -> std::process::Command {
    if std::env::var("MISTRALRS_CCACHE_HIPCC").as_deref() == Ok("1") {
        let mut cmd = std::process::Command::new("ccache");
        cmd.arg(program.as_ref());
        cmd
    } else {
        std::process::Command::new(program.as_ref())
    }
}

#[cfg(all(feature = "cuda", not(feature = "rocm")))]
fn cuda_build_dir(out_dir: &std::path::Path, component: &str) -> std::path::PathBuf {
    println!("cargo:rerun-if-env-changed={CUDA_BUILD_ROOT_ENV}");
    let Some(root) = std::env::var_os(CUDA_BUILD_ROOT_ENV) else {
        return out_dir.to_path_buf();
    };
    let build_dir = std::path::PathBuf::from(root).join("core").join(component);
    std::fs::create_dir_all(&build_dir).expect("failed to create shared CUDA build directory");
    build_dir
}

#[cfg(all(feature = "cuda", not(feature = "rocm")))]
fn cuda_header_hash(dir: &std::path::Path) -> std::io::Result<u64> {
    fn update(hash: &mut u64, bytes: &[u8]) {
        for byte in bytes {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(0x100000001b3);
        }
    }

    fn visit(path: &std::path::Path, hash: &mut u64) -> std::io::Result<()> {
        if path.is_dir() {
            let mut entries = std::fs::read_dir(path)?
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<std::io::Result<Vec<_>>>()?;
            entries.sort();
            for entry in entries {
                visit(&entry, hash)?;
            }
            return Ok(());
        }

        let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
            return Ok(());
        };
        if !matches!(extension, "h" | "cuh" | "hpp") {
            return Ok(());
        }

        update(hash, path.to_string_lossy().as_bytes());
        update(hash, &std::fs::read(path)?);
        Ok(())
    }

    let mut hash = 0xcbf29ce484222325;
    visit(dir, &mut hash)?;
    Ok(hash)
}

fn main() {
    set_git_revision();

    println!("cargo::rustc-check-cfg=cfg(has_flashinfer_gdn_sm90_kernel)");
    println!("cargo::rustc-check-cfg=cfg(has_gdn_fp8_producer)");

    #[cfg(feature = "cudnn")]
    add_cudnn_link_search();

    // Dual builds serve the AMD role (all C kernels hipcc-built): the rocm
    // block owns the GDN kernels and links; the cuda block stays for
    // cuda-only builds.
    #[cfg(feature = "rocm")]
    {
        build_rocm();
    }

    #[cfg(feature = "cuda")]
    {
        build_cuda_core_companion();
    }
}

// GDN (Gated DeltaNet) kernels built with hipcc for hybrid SSM models. Only
// the portable recurrence/conv/gating kernels are needed; they use standard
// warp shuffles and fp16/bf16 math, so they compile for the target GCN/CDNA/RDNA.
#[cfg(feature = "rocm")]
fn build_rocm() {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    let root = std::env::var("CANDLE_ROCM_PATH")
        .ok()
        .or_else(|| std::env::var("ROCM_HOME").ok())
        .or_else(|| std::env::var("ROCM_PATH").ok())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "/opt/rocm".to_string());
    let arch = std::env::var("CANDLE_ROCM_ARCH")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "gfx1151".to_string());
    // CUDA compat shims (cuda_bf16.h etc.) shipped with the candle fork.
    let compat = std::env::var("MISTRALRS_ROCM_COMPAT_INCLUDE")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "../../candle/candle-kernels/src/rocm_compat".to_string());

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda/gdn.cu");
    println!("cargo:rerun-if-changed=src/cuda/attention_prep.cu");
    println!("cargo:rerun-if-changed=src/cuda/input_packing.cu");
    println!("cargo:rerun-if-changed=src/cuda/indexed_copy.cu");
    println!("cargo:rerun-if-changed=src/cuda/dflash_context.cu");
    println!("cargo:rerun-if-changed=src/cuda/dflash_selector.cu");
    println!("cargo:rerun-if-changed=src/cuda/dynamic_conv.cu");
    println!("cargo:rerun-if-changed=src/cuda/speculative_rejection.cu");
    println!("cargo:rerun-if-changed=src/cuda/ssm.cu");
    println!("cargo:rerun-if-changed=src/cuda/sort.cu");
    println!("cargo:rerun-if-changed=src/cuda/graph.cu");
    println!("cargo:rerun-if-changed=src/cuda/moe_gemm.cu");
    println!("cargo:rerun-if-changed=src/cuda/moe_gemm_wmma.cu");
    println!("cargo:rerun-if-changed=src/cuda/moe_gemv.cu");
    println!("cargo:rerun-if-changed=src/cuda/moe_utils.h");
    println!("cargo:rerun-if-changed=src/rocm_ck_flash_attn");
    println!("cargo:rerun-if-env-changed=CANDLE_ROCM_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_HOME");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=CANDLE_ROCM_ARCH");
    println!("cargo:rerun-if-env-changed=MISTRALRS_ROCM_COMPAT_INCLUDE");

    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let mut objects = Vec::new();
    // (source, extra force-include); moe_gemm needs the thrust::cuda::par shim.
    let kernels: &[(&str, Option<&str>)] = &[
        ("src/cuda/gdn.cu", None),
        ("src/cuda/sort.cu", None),
        ("src/cuda/graph.cu", None),
        ("src/cuda/attention_prep.cu", None),
        ("src/cuda/input_packing.cu", None),
        ("src/cuda/indexed_copy.cu", None),
        ("src/cuda/dflash_context.cu", None),
        ("src/cuda/dflash_selector.cu", None),
        ("src/cuda/dynamic_conv.cu", None),
        ("src/cuda/speculative_rejection.cu", None),
        ("src/cuda/ssm.cu", None),
        ("src/cuda/moe_gemm.cu", Some("cuda_thrust.h")),
        ("src/cuda/moe_gemm_wmma.cu", Some("cuda_thrust.h")),
        ("src/cuda/moe_gemv.cu", Some("cuda_thrust.h")),
    ];
    for (src, extra_include) in kernels {
        let stem = Path::new(src)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let object = build_dir.join(format!("{stem}.o"));
        let mut cmd = compiler_command(format!("{root}/bin/hipcc"));
        cmd.arg(src)
            .arg("-I")
            .arg(&compat)
            .arg("-include")
            .arg("cuda_runtime.h");
        if let Some(header) = extra_include {
            cmd.arg("-include").arg(header);
        }
        let status = cmd
            .arg(format!("--offload-arch={arch}"))
            .arg("-DUSE_ROCM")
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-fPIC")
            .arg("-c")
            .arg("-o")
            .arg(&object)
            .status()
            .expect("failed to run hipcc");
        if !status.success() {
            panic!("hipcc failed for {src}");
        }
        objects.push(object);
    }

    // CK flash attention: compile .cpp files with -std=c++20 and CK include path.
    let ck_dir = "src/rocm_ck_flash_attn";
    let ck_sources: Vec<String> = std::fs::read_dir(ck_dir)
        .expect("failed to read ck_flash_attn dir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "cpp" || ext == "cu")
        })
        .map(|e| e.path().to_string_lossy().into_owned())
        .collect();

    for src in &ck_sources {
        let stem = Path::new(src)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let object = build_dir.join(format!("{stem}.o"));
        let status = compiler_command(format!("{root}/bin/hipcc"))
            .arg(src)
            .arg("-I")
            .arg(&compat)
            .arg("-I")
            .arg(format!("{root}/include"))
            .arg(format!("--offload-arch={arch}"))
            .arg("-DUSE_ROCM")
            .arg("-DCK_TILE_FMHA_FWD_FAST_EXP2=0")
            .arg("-std=c++20")
            .arg("-O3")
            .arg("-fPIC")
            .arg("-c")
            .arg("-o")
            .arg(&object)
            .status()
            .expect("failed to run hipcc for ck_flash_attn");
        if !status.success() {
            panic!("hipcc failed for {src}");
        }
        objects.push(object);
    }

    // Every rocm build links the same -fPIC objects into a companion
    // shared library instead of the static archive (the .so carries its
    // own amdhip64 DT_NEEDED; the exe links no HIP symbols and starts on
    // ROCm-less machines).
    println!("cargo:rustc-link-search={}", build_dir.display());
    {
        let so = build_dir.join("libmistralrscuda_hip.so");
        if so.exists() {
            std::fs::remove_file(&so).expect("remove stale libmistralrscuda_hip.so");
        }
        let status = Command::new(format!("{root}/bin/hipcc"))
            .arg("--shared")
            .arg("-o")
            .arg(&so)
            .args(&objects)
            .arg(format!("-L{root}/lib"))
            .arg("-lamdhip64")
            .status()
            .expect("failed to run hipcc");
        if !status.success() {
            panic!("hipcc failed to link libmistralrscuda_hip.so");
        }
        println!("cargo:rustc-env=MISTRALRS_CORE_HIP_PLUGIN={}", so.display());
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

#[cfg(all(feature = "cuda", not(feature = "rocm")))]
fn set_cuda_toolkit_version() -> Option<u32> {
    let (version, code) = cuda_toolkit_version()?;
    {
        println!("cargo:rustc-env=MISTRALRS_BUILD_CUDA_VERSION={version}");
        println!("cargo:rustc-env=MISTRALRS_BUILD_CUDA_VERSION_CODE={code}");
    }
    Some(code)
}

#[cfg(all(feature = "cuda", not(feature = "rocm")))]
fn cuda_toolkit_version() -> Option<(String, u32)> {
    let version = cudaforge::CudaToolkit::detect().ok()?.version?;
    parse_cuda_version(&version)
}

#[cfg(all(feature = "cuda", not(feature = "rocm")))]
fn parse_cuda_version(version: &str) -> Option<(String, u32)> {
    let mut parts = version.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next().unwrap_or("0").parse().ok()?;
    Some((format!("{major}.{minor}"), major * 100 + minor))
}

#[cfg(feature = "cudnn")]
fn add_cudnn_link_search() {
    use std::path::PathBuf;

    println!("cargo:rerun-if-env-changed=CUDNN_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let target = std::env::var("TARGET").unwrap_or_default();
    if !target.contains("msvc") {
        return;
    }

    if let Ok(dir) = std::env::var("CUDNN_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
        return;
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        candidates.push(PathBuf::from(&cuda_path).join("lib").join("x64"));
    }
    let cudnn_root = PathBuf::from(r"C:\Program Files\NVIDIA\CUDNN");
    if let Ok(versions) = std::fs::read_dir(&cudnn_root) {
        for version in versions.flatten() {
            let lib = version.path().join("lib");
            candidates.push(lib.join("x64"));
            if let Ok(cuda_vers) = std::fs::read_dir(&lib) {
                for cuda_ver in cuda_vers.flatten() {
                    candidates.push(cuda_ver.path().join("x64"));
                }
            }
        }
    }

    for dir in candidates {
        if dir.join("cudnn.lib").is_file() {
            println!("cargo:rustc-link-search=native={}", dir.display());
            return;
        }
    }

    println!(
        "cargo:warning=cudnn feature enabled but cudnn.lib not found; set CUDNN_LIB_DIR to its directory"
    );
}

fn set_git_revision() {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=MISTRALRS_GIT_REVISION={commit}");
    println!("cargo:rerun-if-changed=.git/HEAD");
    if let Ok(head) = std::fs::read_to_string(".git/HEAD") {
        if let Some(ref_path) = head.strip_prefix("ref:") {
            let ref_path = ref_path.trim();
            if !ref_path.is_empty() {
                println!("cargo:rerun-if-changed=.git/{}", ref_path);
            }
        }
    }
}

/// NVIDIA-role GDN companion for dual builds: nvcc-compiles the GDN TU
/// (recurrence/conv/gating kernels - portable warp-shuffle and fp16/bf16
/// math, no tensor-core PTX) into a dlopened .so, mirroring the hip
/// companion and the quant/paged-attn CUDA companions.
#[cfg(feature = "cuda")]
fn build_cuda_core_companion() {
    use std::path::PathBuf;
    use std::process::Command;

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let nvcc = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .map(|p| format!("{p}/bin/nvcc"))
        .unwrap_or_else(|_| "nvcc".to_string());
    let cap: usize = std::env::var("CUDA_COMPUTE_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(80);

    const SOURCES: &[&str] = &[
        "src/cuda/attention_prep.cu",
        "src/cuda/dflash_context.cu",
        "src/cuda/dflash_selector.cu",
        "src/cuda/dynamic_conv.cu",
        "src/cuda/gdn.cu",
        "src/cuda/graph.cu",
        "src/cuda/indexed_copy.cu",
        "src/cuda/input_packing.cu",
        "src/cuda/sort.cu",
        "src/cuda/speculative_rejection.cu",
        "src/cuda/ssm.cu",
    ];
    for src in SOURCES {
        println!("cargo:rerun-if-changed={src}");
    }

    let build_dir = out_dir.join("cuda_companion");
    std::fs::create_dir_all(&build_dir).expect("create core cuda_companion dir");
    let mut objects = Vec::new();
    for src in SOURCES {
        let object = build_dir.join(
            std::path::Path::new(src)
                .file_stem()
                .expect("source stem")
                .to_string_lossy()
                .into_owned()
                + ".o",
        );
        let status = Command::new(&nvcc)
            .arg(src)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("--use_fast_math")
            .arg("-U__CUDA_NO_HALF_OPERATORS__")
            .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
            .arg("-U__CUDA_NO_HALF2_OPERATORS__")
            .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg("--compiler-options")
            .arg("-fPIC")
            .arg("-gencode")
            .arg(format!("arch=compute_{cap},code=sm_{cap}"))
            .arg("-gencode")
            .arg(format!("arch=compute_{cap},code=compute_{cap}"))
            .arg("-c")
            .arg("-o")
            .arg(&object)
            .status()
            .expect("failed to run nvcc for the core companion");
        if !status.success() {
            panic!("nvcc failed for {src}");
        }
        objects.push(object);
    }

    let so = out_dir.join("libmistralrscuda_cuda.so");
    if so.exists() {
        std::fs::remove_file(&so).expect("remove stale libmistralrscuda_cuda.so");
    }
    let status = Command::new(&nvcc)
        .arg("--shared")
        .arg("--cudart")
        .arg("shared")
        .arg("-o")
        .arg(&so)
        .args(&objects)
        .status()
        .expect("failed to link the core companion");
    if !status.success() {
        panic!("nvcc failed to link libmistralrscuda_cuda.so");
    }
    println!(
        "cargo:rustc-env=MISTRALRS_CORE_CUDA_PLUGIN={}",
        so.display()
    );
}
