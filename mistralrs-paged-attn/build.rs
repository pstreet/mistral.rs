use anyhow::Result;

#[cfg(feature = "metal")]
include!("src/metal/kernels/source_set.rs");

#[cfg(all(feature = "cuda", target_family = "unix"))]
const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");
#[cfg(all(feature = "cuda", target_family = "unix"))]
const FA3_CUTLASS_COMMIT: &str = "62750a2b75c802660e4894434dc55e839f322277";
#[cfg(all(feature = "cuda", target_family = "unix"))]
const FA3_SOURCES: [&str; 4] = [
    "src/cuda/fa3/fa3_decode_api.cu",
    "third_party/flash-attention/hopper/instantiations/flash_fwd_hdim256_e4m3_paged_split_sm90.cu",
    "third_party/flash-attention/hopper/flash_fwd_combine.cu",
    "third_party/flash-attention/hopper/flash_prepare_scheduler.cu",
];
#[cfg(all(feature = "cuda", target_family = "unix"))]
const CUDA_BUILD_ROOT_ENV: &str = "MISTRALRS_CUDA_BUILD_ROOT";

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn cuda_build_dir(out_dir: &std::path::Path, component: &str) -> std::path::PathBuf {
    println!("cargo:rerun-if-env-changed={CUDA_BUILD_ROOT_ENV}");
    let Some(root) = std::env::var_os(CUDA_BUILD_ROOT_ENV) else {
        return out_dir.to_path_buf();
    };
    let build_dir = std::path::PathBuf::from(root)
        .join("paged-attn")
        .join(component);
    std::fs::create_dir_all(&build_dir).expect("failed to create shared CUDA build directory");
    build_dir
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn cuda_header_hash(dir: &str, excluded_dirs: &[&str]) -> Result<u64> {
    use std::path::Path;

    fn update(hash: &mut u64, bytes: &[u8]) {
        for byte in bytes {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(0x100000001b3);
        }
    }

    fn visit(path: &Path, excluded_dirs: &[&str], hash: &mut u64) -> Result<()> {
        if excluded_dirs
            .iter()
            .any(|excluded| path == Path::new(excluded))
        {
            return Ok(());
        }
        if path.is_dir() {
            let mut entries = std::fs::read_dir(path)?
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<std::io::Result<Vec<_>>>()?;
            entries.sort();
            for entry in entries {
                visit(&entry, excluded_dirs, hash)?;
            }
            return Ok(());
        }

        let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
            return Ok(());
        };
        if ext != "cuh" && ext != "h" && ext != "hpp" {
            return Ok(());
        }

        println!("cargo:rerun-if-changed={}", path.display());
        update(hash, path.to_string_lossy().as_bytes());
        update(hash, &std::fs::read(path)?);
        Ok(())
    }

    let mut hash = 0xcbf29ce484222325;
    visit(Path::new(dir), excluded_dirs, &mut hash)?;
    Ok(hash)
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn build_cuda() -> Result<()> {
    use std::path::PathBuf;

    // Declare expected cfg values for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_fp8)");
    println!("cargo::rustc-check-cfg=cfg(has_fa3_fp8_paged)");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_NVCC_FLAGS");
    println!("cargo:rerun-if-changed=src/cuda/pagedattention.cuh");
    println!("cargo:rerun-if-changed=src/cuda/copy_blocks_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/reshape_and_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/concat_and_cache_mla_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/gather_mla_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/gather_kv_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer_decode.cu");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer_mla_decode.cu");
    println!("cargo:rerun-if-changed=src/cuda/update_kvscales.cu");
    println!("cargo:rerun-if-changed=src/cuda/flash_attn_sinks.cu");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/cp_async.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/exception.h");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/fastdiv.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/fp16.h");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/frag_layout_swizzle.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/layout.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/math.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/mma.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/page.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/permuted_smem.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/pos_enc.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/utils.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/vec_dtypes.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/cascade.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/decode.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/default_decode_params.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/mask.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/state.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/variant_helper.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/variants.cuh");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let kernel_build_dir = cuda_build_dir(&out_dir, "kernels");

    let header_hash_arg = format!(
        "-DMISTRALRS_CUDA_HEADER_HASH={:016x}",
        cuda_header_hash("src/cuda", &["src/cuda/fa3"])?
    );

    let mut builder = cudaforge::KernelBuilder::new()
        .source_glob("src/cuda/*.cu")
        .watch(["src/cuda"])
        .out_dir(&kernel_build_dir)
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("--verbose")
        .arg("--compiler-options")
        .arg("-fPIC")
        .arg(&header_hash_arg);

    let compute_cap = builder.get_compute_cap().unwrap_or(80);
    // Enable FP8 if compute capability >= 8.0 (Ampere and newer)
    let using_fp8 = if compute_cap >= 80 {
        builder = builder.arg("-DENABLE_FP8");
        true
    } else {
        false
    };

    // https://github.com/EricLBuehler/mistral.rs/issues/286
    if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
        builder = builder.arg("--compiler-options");
        builder = builder.arg(cuda_nvcc_flags_env);
    }
    println!("cargo:info={builder:?}");

    let target = std::env::var("TARGET").unwrap();
    // https://github.com/EricLBuehler/mistral.rs/issues/588
    let out_file = if target.contains("msvc") {
        // Windows case
        out_dir.join("mistralrspagedattention.lib")
    } else {
        out_dir.join("libmistralrspagedattention.a")
    };
    builder
        .build_lib(out_file)
        .expect("Build paged attention lib failed!");

    let using_fa3_fp8_paged = compute_cap == 90;
    if using_fa3_fp8_paged {
        let fa3_header_hash = cuda_header_hash("third_party/flash-attention", &[])?
            .wrapping_mul(0x100000001b3)
            ^ cuda_header_hash("src/cuda/fa3", &[])?;
        let fa3_header_hash_arg = format!("-DMISTRALRS_FA3_HEADER_HASH={fa3_header_hash:016x}");
        let fa3_build_dir = cuda_build_dir(&out_dir, "fa3");
        let mut fa3_builder = cudaforge::KernelBuilder::new()
            .source_files(FA3_SOURCES)
            .out_dir(&fa3_build_dir)
            .compute_cap_arch("90a")
            .with_cutlass(Some(FA3_CUTLASS_COMMIT))
            .include_path("src/cuda/fa3")
            .include_path("third_party/flash-attention/hopper")
            .watch(["src/cuda/fa3", "third_party/flash-attention"])
            .arg("-std=c++17")
            .arg("-O3")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg("--use_fast_math")
            .arg("-diag-suppress")
            .arg("20013")
            .arg("-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED")
            .arg("-DCUTLASS_ENABLE_GDC_FOR_SM90")
            .arg("-DCUTLASS_DEBUG_TRACE_LEVEL=0")
            .arg("-DNDEBUG")
            .arg("-DFLASHATTENTION_DISABLE_BACKWARD")
            .arg("-DFLASHATTENTION_DISABLE_CLUSTER")
            .arg("-DFLASHATTENTION_DISABLE_SM8x")
            .arg("-DFLASHATTENTION_DISABLE_LOCAL")
            .arg("-DFLASHATTENTION_DISABLE_APPENDKV")
            .arg("-DFLASHATTENTION_DISABLE_SOFTCAP")
            .arg("-DFLASHATTENTION_PACKGQA_ONLY")
            .arg("-DFLASHATTENTION_VARLEN_ONLY")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg("-Xcompiler")
            .arg("-fvisibility=hidden")
            .arg(&fa3_header_hash_arg)
            .max_threads(1)
            .nvcc_thread_patterns(
                &["flash_fwd", "flash_prepare_scheduler", "fa3_decode_api"],
                4,
            );
        if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
            fa3_builder = fa3_builder
                .arg("--compiler-options")
                .arg(cuda_nvcc_flags_env);
        }
        fa3_builder
            .build_lib(out_dir.join("libmistralrsfa3paged.a"))
            .expect("Build FA3 FP8 paged attention lib failed!");
    }

    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=mistralrspagedattention");
    println!("cargo:rustc-link-lib=dylib=cudart");

    if using_fa3_fp8_paged {
        println!("cargo:rustc-link-lib=mistralrsfa3paged");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-cfg=has_fa3_fp8_paged");
    }

    if using_fp8 {
        println!("cargo:rustc-cfg=has_fp8");
    }
    Ok(())
}

// Dense-model paged-attention kernels built with hipcc. The FlashInfer
// (ldmatrix/cp.async) and MLA kernels need tensor-core instructions RDNA
// lacks and are excluded; FP8 KV cache is off (no ENABLE_FP8).
#[cfg(feature = "rocm")]
const PAGED_ATTN_ROCM_KERNELS: &[&str] = &[
    "src/cuda/pagedattention_v1_f16.cu",
    "src/cuda/pagedattention_v1_bf16.cu",
    "src/cuda/pagedattention_v1_f32.cu",
    "src/cuda/pagedattention_v2_f16.cu",
    "src/cuda/pagedattention_v2_bf16.cu",
    "src/cuda/pagedattention_v2_f32.cu",
    "src/cuda/reshape_and_cache_kernel.cu",
    "src/cuda/copy_blocks_kernel.cu",
    "src/cuda/gather_kv_cache_kernel.cu",
];

#[cfg(feature = "rocm")]
fn build_rocm() -> Result<()> {
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
    println!("cargo:rerun-if-env-changed=CANDLE_ROCM_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_HOME");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=CANDLE_ROCM_ARCH");
    println!("cargo:rerun-if-env-changed=MISTRALRS_ROCM_COMPAT_INCLUDE");
    for kernel in PAGED_ATTN_ROCM_KERNELS {
        println!("cargo:rerun-if-changed={kernel}");
    }

    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let objects = PAGED_ATTN_ROCM_KERNELS
        .iter()
        .map(|kernel| {
            build_dir
                .join(Path::new(kernel).file_stem().unwrap())
                .with_extension("o")
        })
        .collect::<Vec<_>>();

    for (kernel, object) in PAGED_ATTN_ROCM_KERNELS.iter().zip(&objects) {
        let status = Command::new(format!("{root}/bin/hipcc"))
            .arg(kernel)
            .arg("-I")
            .arg(&compat)
            .arg("-Isrc/cuda")
            .arg("-include")
            .arg("cuda_runtime.h")
            .arg(format!("--offload-arch={arch}"))
            .arg("-DUSE_ROCM")
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-fPIC")
            .arg("-c")
            .arg("-o")
            .arg(object)
            .status()?;
        anyhow::ensure!(status.success(), "hipcc failed for {kernel}");
    }

    let out_file = build_dir.join("libmistralrspagedattention.a");
    let status = Command::new("ar")
        .arg("crs")
        .arg(&out_file)
        .args(&objects)
        .status()?;
    anyhow::ensure!(status.success(), "ar failed");

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=mistralrspagedattention");
    // Scale tensors are always allocated (default 1.0) and only read by the FP8
    // KV path (cache_dtype==3), which is never selected on RDNA. Marking FP8
    // "present" lets those pointers pass through without triggering the
    // !USE_FP8 bail; the FP8 kernel instantiations stay gated off (no ENABLE_FP8).
    println!("cargo:rustc-cfg=has_fp8");
    Ok(())
}

fn main() -> Result<()> {
    // Declare expected cfg values for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_fp8)");
    println!("cargo::rustc-check-cfg=cfg(has_fa3_fp8_paged)");

    // Exactly one backend block compiles; each yields the build result as the
    // function's return value.
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    {
        build_cuda()
    }

    #[cfg(feature = "rocm")]
    {
        build_rocm()
    }

    #[cfg(feature = "metal")]
    {
        mistralrs_metal_compile::compile_metallibs(&PAGED_ATTENTION_METAL_SOURCE_SET)
    }

    // No backend feature selected: nothing to build.
    #[cfg(not(any(
        all(feature = "cuda", target_family = "unix"),
        feature = "rocm",
        feature = "metal"
    )))]
    {
        Ok(())
    }
}
