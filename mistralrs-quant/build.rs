#[cfg(feature = "cuda")]
const CUTLASS_COMMIT: &str = "7127592069c2fe01b041e174ba4345ef9b279671";
#[cfg(feature = "cuda")]
const DEEPGEMM_CUTLASS_COMMIT: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";
#[cfg(feature = "cuda")]
const CUTLASS_COMMIT_ENV: &str = "MISTRALRS_CUTLASS_COMMIT";
#[cfg(feature = "cuda")]
const CUDA_BUILD_ROOT_ENV: &str = "MISTRALRS_CUDA_BUILD_ROOT";
#[cfg(feature = "cuda")]
const SUPPORTED_CUDA_TOOLKIT_VERSIONS: &[(usize, usize)] = &[
    (13, 3),
    (13, 2),
    (13, 1),
    (13, 0),
    (12, 9),
    (12, 8),
    (12, 6),
    (12, 5),
    (12, 4),
    (12, 3),
    (12, 2),
    (12, 1),
    (12, 0),
    (11, 8),
    (11, 7),
    (11, 6),
    (11, 5),
    (11, 4),
];

/// Route hipcc TU compiles (`-c` steps only) through ccache when opted
/// in: identical invocations across target dirs are served from cache.
/// Verified: ccache hits repeat `-c` compiles with identical objects.
/// Verified ccache hits on repeat hipcc AND nvcc `-c` compiles.
/// Deliberately excluded: all link steps (drivers decline them), and
/// sccache for nvcc (reports phantom hits on failed compiles).
/// Opt in: MISTRALRS_CCACHE_GPU=1 (set in ~/.cargo/config.toml).
/// Linker args selecting mold for GPU companion `.so` links when it is
/// on PATH (fast links); empty otherwise, keeping the driver default.
/// hipcc (clang-based) takes `-fuse-ld` directly; nvcc needs it routed
/// through the host compiler driver via `-Xcompiler`.
fn mold_available() -> bool {
    std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|dir| dir.join("mold").is_file()))
}

#[allow(dead_code)]
fn mold_link_args_for_hipcc() -> Vec<&'static str> {
    if mold_available() {
        vec!["-fuse-ld=mold"]
    } else {
        Vec::new()
    }
}

#[allow(dead_code)]
fn mold_link_args_for_nvcc() -> Vec<&'static str> {
    if mold_available() {
        vec!["-Xcompiler", "-fuse-ld=mold"]
    } else {
        Vec::new()
    }
}

#[allow(dead_code)]
fn compiler_command(program: impl AsRef<std::path::Path>) -> std::process::Command {
    if std::env::var("MISTRALRS_CCACHE_GPU").as_deref() == Ok("1") {
        let mut cmd = std::process::Command::new("ccache");
        cmd.arg(program.as_ref());
        cmd
    } else {
        std::process::Command::new(program.as_ref())
    }
}

#[cfg(feature = "cuda")]
fn cuda_build_dir(out_dir: &std::path::Path, component: &str) -> std::path::PathBuf {
    println!("cargo:rerun-if-env-changed={CUDA_BUILD_ROOT_ENV}");
    let Some(root) = std::env::var_os(CUDA_BUILD_ROOT_ENV) else {
        return out_dir.join(component);
    };
    let build_dir = std::path::PathBuf::from(root).join("quant").join(component);
    std::fs::create_dir_all(&build_dir).expect("failed to create shared CUDA build directory");
    build_dir
}

#[cfg(feature = "cuda")]
fn cuda_kernel_builder(build_dir: &std::path::Path) -> cudaforge::KernelBuilder {
    cudaforge::KernelBuilder::new()
        .out_dir(build_dir)
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
}

#[cfg(feature = "metal")]
include!("src/metal_kernels/source_set.rs");

#[cfg(feature = "rocm")]
fn rocm_root() -> String {
    for var in ["CANDLE_ROCM_PATH", "ROCM_HOME", "ROCM_PATH"] {
        if let Ok(p) = std::env::var(var) {
            if !p.is_empty() {
                return p;
            }
        }
    }
    "/opt/rocm".to_string()
}

#[cfg(feature = "rocm")]
fn build_rocm() -> Result<(), String> {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    let rocm = rocm_root();
    let arch = std::env::var("CANDLE_ROCM_ARCH")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "gfx1151".to_string());
    // AMD-encoded host cc (0x1000000 | gfxNNNN) so the MMQ host launch decisions
    // (use_stream_k, mmq_x, granularity) track the RDNA WMMA device path. Without
    // it the host sees a small cc (1030), assumes NVIDIA/turing (8-wide granularity,
    // stream-k) and can pick an mmq_x (e.g. 40) the 16-wide device path rejects.
    let amd_host_cc: Option<String> = arch
        .strip_prefix("gfx")
        .and_then(|num| u32::from_str_radix(num, 16).ok())
        .map(|gfx| format!("{:#x}", 0x0100_0000 | gfx));
    // CUDA compat shims (cuda_bf16.h etc.) shipped with the candle fork; override via env.
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
    println!("cargo:rerun-if-changed=kernels/rotary/rotary.cu");
    println!("cargo:rerun-if-changed=kernels/rotary/cuda_compat.h");

    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let hipcc = Path::new(&rocm).join("bin/hipcc");

    // (source, extra force-include); ops.cu needs the thrust compat shim.
    let kernels: &[(&str, Option<&str>)] = &[
        ("kernels/rotary/rotary.cu", None),
        ("kernels/ops/ops.cu", Some("cuda_thrust.h")),
        // GGUF MMQ/MMVQ family: NVIDIA MMA PTX is disabled under USE_ROCM in
        // mmq_common.cuh; RDNA3/RDNA4 use WMMA intrinsics, other targets fall
        // back to generic load/FMA routes.
        ("kernels/mmvq_gguf/mmvq_gguf.cu", None),
        ("kernels/gemv/gemv.cu", None),
        ("kernels/hqq/hqq.cu", None),
        ("kernels/hqq/hqq_bitpack.cu", None),
        ("kernels/afq/afq.cu", None),
        ("kernels/afq/afq_gemm.cu", None),
        ("kernels/gptq/q_gemm.cu", None),
        ("kernels/moe/moe_align.cu", None),
        ("kernels/moe/gelu_tanh_and_mul.cu", None),
        ("kernels/indexed_moe/indexed_moe.cu", None),
        ("kernels/moe_grouped/moe_grouped.cu", None),
        ("kernels/mmq_gguf/mmq_quantize.cu", None),
        ("kernels/mmq_gguf/mmq_instance_q2_k.cu", None),
        ("kernels/mmq_gguf/mmq_instance_q3_k.cu", None),
        ("kernels/mmq_gguf/mmq_instance_q4_0.cu", None),
        ("kernels/mmq_gguf/mmq_instance_q4_1.cu", None),
        ("kernels/mmq_gguf/mmq_instance_q4_k.cu", None),
        ("kernels/mmq_gguf/mmq_instance_q5_0.cu", None),
        ("kernels/mmq_gguf/mmq_instance_q5_1.cu", None),
        ("kernels/mmq_gguf/mmq_instance_q5_k.cu", None),
        ("kernels/mmq_gguf/mmq_instance_q6_k.cu", None),
        ("kernels/mmq_gguf/mmq_instance_q8_0.cu", None),
        // FP8 dequant/quant + blockwise GEMM; portable FMA routes, no tensor-core PTX.
        ("kernels/scalar_fp8/scalar_fp8.cu", None),
        ("kernels/vector_fp8/vector_fp8.cu", None),
        ("kernels/blockwise_fp8/blockwise_fp8.cu", None),
        ("kernels/blockwise_fp8/blockwise_fp8_gemm.cu", None),
        // bitsandbytes (BNB) int8/fp4/nf4 blockwise dequant; CUB block load/store.
        ("kernels/bitsandbytes/dequant.cu", None),
    ];

    let mut objects = Vec::new();
    for (src, force_include) in kernels {
        println!("cargo:rerun-if-changed={src}");
        let stem = Path::new(src)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap();
        let obj = build_dir.join(format!("{stem}.o"));
        let mut cmd = compiler_command(&hipcc);
        cmd.arg(src)
            .arg(format!("-I{compat}"))
            .arg("-include")
            .arg("cuda_runtime.h");
        if let Some(header) = force_include {
            cmd.arg("-include").arg(header);
        }
        if src.contains("mmq_gguf/") {
            if let Some(cc) = &amd_host_cc {
                cmd.arg(format!("-DCANDLE_ROCM_HOST_CC={cc}"));
            }
        }
        cmd.arg("-c")
            .arg("-o")
            .arg(&obj)
            .arg(format!("--offload-arch={arch}"))
            .arg("-DUSE_ROCM")
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-fPIC");
        let status = cmd.status().map_err(|e| e.to_string())?;
        if !status.success() {
            return Err(format!(
                "hipcc failed to compile {src} (is {hipcc:?} present?)"
            ));
        }
        objects.push(obj);
    }

    // Headers shared across the kernel TUs: a header edit must retrigger hipcc.
    for h in [
        "mmq_common.cuh",
        "mmq_gguf.cuh",
        "mmq_mma.cuh",
        "mmq_vecdotq.cuh",
    ] {
        println!("cargo:rerun-if-changed=kernels/mmq_gguf/{h}");
    }

    let lib = build_dir.join("libmistralrsquant.a");
    if lib.exists() {
        std::fs::remove_file(&lib).expect("remove stale libmistralrsquant.a");
    }
    let ar = if Path::new(&rocm).join("lib/llvm/bin/llvm-ar").exists() {
        Path::new(&rocm)
            .join("lib/llvm/bin/llvm-ar")
            .display()
            .to_string()
    } else {
        "ar".to_string()
    };
    let mut ar_cmd = Command::new(&ar);
    ar_cmd.arg("rcs").arg(&lib);
    for obj in &objects {
        ar_cmd.arg(obj);
    }
    let status = ar_cmd.status().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("failed to archive libmistralrsquant.a".to_string());
    }

    // FP8 kernels are portable FMA routes (no tensor-core PTX), so all three
    // families are available on every ROCm target.
    println!("cargo:rustc-cfg=has_scalar_fp8_kernels");
    println!("cargo:rustc-cfg=has_vector_fp8_kernels");
    println!("cargo:rustc-cfg=has_blockwise_fp8_kernels");

    // Every rocm build links the same -fPIC objects into a companion
    // shared library instead of the static archive. The .so carries its
    // own amdhip64 DT_NEEDED (graceful dlopen failure when ROCm is
    // absent); the exe links no HIP symbols at all, so the binary starts
    // on machines with no ROCm installed.
    {
        let so = build_dir.join("libmistralrsquant_hip.so");
        if so.exists() {
            std::fs::remove_file(&so).expect("remove stale libmistralrsquant_hip.so");
        }
        let mut cmd = Command::new(&hipcc);
        cmd.args(mold_link_args_for_hipcc());
        cmd.arg("--shared").arg("-o").arg(&so);
        for obj in &objects {
            cmd.arg(obj);
        }
        cmd.arg(format!("-L{}/lib", rocm)).arg("-lamdhip64");
        let status = cmd.status().map_err(|e| e.to_string())?;
        if !status.success() {
            return Err("hipcc failed to link libmistralrsquant_hip.so".to_string());
        }
        println!(
            "cargo:rustc-env=MISTRALRS_QUANT_HIP_PLUGIN={}",
            so.display()
        );
    }

    println!("cargo:rustc-link-search={}", build_dir.display());
    // Every rocm build links neither the archive nor the HIP runtime:
    // both resolve at runtime through the companion plugin above, so the
    // binary starts on machines with no ROCm installed.
    println!("cargo:rustc-link-search=native={}/lib", rocm);
    Ok(())
}

#[cfg(feature = "cuda")]
#[path = "src/build_support/cuda_headers.rs"]
mod cuda_headers;

#[cfg(feature = "cuda")]
#[path = "src/build_support/deepgemm_jit_bundle.rs"]
mod deepgemm_jit_bundle;

#[cfg(feature = "cuda")]
#[allow(unused)]
fn cuda_version_from_build_system() -> (usize, usize) {
    let output = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .expect("Failed to execute `nvcc`");

    if !output.status.success() {
        panic!(
            "`nvcc --version` failed.\nstdout:\n{}\n\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let (major, minor) = parse_cuda_version_from_nvcc(&stdout).unwrap_or_else(|| {
        panic!("Unsupported cuda toolkit version from `nvcc --version`:\n{stdout}")
    });
    if SUPPORTED_CUDA_TOOLKIT_VERSIONS.contains(&(major, minor)) {
        (major, minor)
    } else {
        panic!("Unsupported cuda toolkit version: `{major}.{minor}`. Please raise a github issue.")
    }
}

#[cfg(feature = "cuda")]
fn parse_cuda_version_from_nvcc(stdout: &str) -> Option<(usize, usize)> {
    let release = stdout.split("release ").nth(1)?;
    let version = release
        .split(|c: char| c == ',' || c.is_whitespace())
        .next()?;
    let mut parts = version.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor))
}

#[cfg(feature = "cuda")]
fn cutile_supported_for_build_cuda(major: usize, minor: usize, compute_cap: usize) -> bool {
    let cuda_code = major * 100 + minor;
    ((80..90).contains(&compute_cap) && cuda_code >= 1302)
        || (compute_cap == 90 && cuda_code >= 1303)
        || (compute_cap >= 100 && cuda_code >= 1302)
}

fn main() -> Result<(), String> {
    // Declare expected cfg values for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_marlin_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_blockwise_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_cutlass_fp8_sm90_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_deepgemm_fp8_sm90_provider)");
    println!("cargo::rustc-check-cfg=cfg(has_scalar_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_vector_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_mxfp4_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_mxfp4_wmma_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_cutlass_moe_kernels)");
    println!("cargo::rustc-check-cfg=cfg(cuda_ge_13000)");

    #[cfg(feature = "metal")]
    {
        mistralrs_metal_compile::compile_metallibs(&QUANT_METAL_SOURCE_SET)
    }

    #[cfg(feature = "rocm")]
    {
        build_rocm()?;
    }

    #[cfg(feature = "cuda")]
    {
        build_cuda_moe_companion()?;
    }

    Ok(())
}

/// NVIDIA-role MoE companion: nvcc-compiles the fused decode TU
/// (indexed_moe.cu is self-contained: Q8_1 quantize + fused gate_up +
/// down_aggregate launches for every GGUF dtype) into a dlopened .so
/// for any cuda build shape, mirroring mistralrs-paged-attn's CUDA
/// companion. Never linked, so the exe keeps zero GPU DT_NEEDED.
#[cfg(feature = "cuda")]
fn build_cuda_moe_companion() -> Result<(), String> {
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
        "kernels/hqq/hqq_bitpack.cu",
        "kernels/lora/dynamic_lora.cu",
        "kernels/lora/routed_moe_lora.cu",
        "kernels/rotary/rotary.cu",
        "kernels/ops/ops.cu",
        "kernels/gemv/gemv.cu",
        "kernels/hqq/hqq.cu",
        "kernels/afq/afq.cu",
        "kernels/afq/afq_gemm.cu",
        "kernels/gptq/q_gemm.cu",
        "kernels/moe/moe_align.cu",
        "kernels/moe/gelu_tanh_and_mul.cu",
        "kernels/scalar_fp8/scalar_fp8.cu",
        "kernels/vector_fp8/vector_fp8.cu",
        "kernels/blockwise_fp8/blockwise_fp8.cu",
        "kernels/blockwise_fp8/blockwise_fp8_gemm.cu",
        "kernels/indexed_moe/indexed_moe.cu",
        // Dense decode (batch 1-8) MMVQ family.
        "kernels/mmvq_gguf/mmvq_gguf.cu",
        // Grouped-prefill MoE + dense/grouped MMQ family: dispatch tables,
        // grouped GEMMs, weighted reduce, Q8_1 activation quantize, and the
        // MMQ dequant-MMA kernels (NVIDIA PTX paths under nvcc; the same
        // sources serve the hip companion through the USE_ROCM WMMA route).
        "kernels/moe_grouped/moe_grouped.cu",
        "kernels/mmq_gguf/mmq_quantize.cu",
        // MMQ is instantiated per dtype (no umbrella TU); each instance
        // brings the common/mma/vecdotq headers along.
        "kernels/mmq_gguf/mmq_instance_q2_k.cu",
        "kernels/mmq_gguf/mmq_instance_q3_k.cu",
        "kernels/mmq_gguf/mmq_instance_q4_0.cu",
        "kernels/mmq_gguf/mmq_instance_q4_1.cu",
        "kernels/mmq_gguf/mmq_instance_q4_k.cu",
        "kernels/mmq_gguf/mmq_instance_q5_0.cu",
        "kernels/mmq_gguf/mmq_instance_q5_1.cu",
        "kernels/mmq_gguf/mmq_instance_q5_k.cu",
        "kernels/mmq_gguf/mmq_instance_q6_k.cu",
        "kernels/mmq_gguf/mmq_instance_q8_0.cu",
    ];
    for src in SOURCES {
        println!("cargo:rerun-if-changed={src}");
    }
    for h in [
        "mmq_common.cuh",
        "mmq_gguf.cuh",
        "mmq_mma.cuh",
        "mmq_vecdotq.cuh",
    ] {
        println!("cargo:rerun-if-changed=kernels/mmq_gguf/{h}");
    }

    let build_dir = out_dir.join("cuda_companion");
    std::fs::create_dir_all(&build_dir).map_err(|e| e.to_string())?;
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
        let status = compiler_command(&nvcc)
            .arg(src)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("--use_fast_math")
            .arg("-U__CUDA_NO_HALF_OPERATORS__")
            .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
            .arg("-U__CUDA_NO_HALF2_OPERATORS__")
            .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
            .arg("--expt-relaxed-constexpr")
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
            .map_err(|e| e.to_string())?;
        if !status.success() {
            return Err(format!("nvcc failed for {src}"));
        }
        objects.push(object);
    }

    let so = out_dir.join("libmistralrsquant_cuda.so");
    if so.exists() {
        std::fs::remove_file(&so).map_err(|e| e.to_string())?;
    }
    let status = Command::new(&nvcc)
        .args(mold_link_args_for_nvcc())
        .arg("--shared")
        .arg("--cudart")
        .arg("shared")
        .arg("-o")
        .arg(&so)
        .args(&objects)
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("nvcc failed to link libmistralrsquant_cuda.so".to_string());
    }
    println!(
        "cargo:rustc-env=MISTRALRS_QUANT_CUDA_PLUGIN={}",
        so.display()
    );
    Ok(())
}
