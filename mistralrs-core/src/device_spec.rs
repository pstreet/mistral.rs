//! Device selection, resolution, and capability reporting for the serving
//! stack (S3b of the multi-backend roadmap).
//!
//! Device strings select API + device together - the API qualifier IS the
//! vendor-API choice (`cuda:N` = CUDA, `hip:N` = ROCm/HIP, `cpu`, and later
//! `vulkan:N` for the vendor-agnostic API on any vendor). The index is
//! per-API enumeration, so `hip:0` and `cuda:0` may address the same card.
//!
//! Resolution is probe-gated: with runtime dlopen (`dynamic-loading` on both
//! vendor crates), calling a driver without its runtime present panics inside
//! the crate rather than erroring, so `is_culib_present()` MUST gate every
//! first driver call.
//!
//! `BackendCaps` is the per-device capability contract: what the serving
//! stack can run on this device in THIS build shape. It exists so features
//! degrade loudly (a startup log line naming what is off, and an early
//! refusal where a whole model family cannot load), never silently.

use candle_core::{Device, GpuArch, Result};

/// A parsed device string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceSpec {
    /// `"cpu"` - no GPU.
    Cpu,
    /// `"cuda:N"` - the CUDA API. The NVIDIA role in dual builds; the
    /// compiled vendor (which may be HIP-bound) in single-vendor builds.
    Cuda(usize),
    /// `"hip:N"` - the HIP API. The AMD role in dual builds; an alias for
    /// the compiled vendor in single-vendor rocm builds (configs stay
    /// portable across build shapes).
    Hip(usize),
}

impl DeviceSpec {
    /// Parse a device string: `cpu`, `cuda:<idx>`, `hip:<idx>`, `vulkan:<idx>`
    /// (reserved, S4). Unknown forms list the valid ones.
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }
        let (api, idx) = s
            .rsplit_once(':')
            .and_then(|(api, idx)| Some((api, idx.parse::<usize>().ok()?)))
            .ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "invalid device string {s:?}: expected \"cpu\", \"cuda:<idx>\", \
                     \"hip:<idx>\", or \"vulkan:<idx>\""
                ))
            })?;
        let spec = match api.to_ascii_lowercase().as_str() {
            "cuda" => Self::Cuda(idx),
            "hip" => Self::Hip(idx),
            "vulkan" => {
                return Err(candle_core::Error::msg(
                    "device \"vulkan:<idx>\" is not supported yet (the Vulkan backend is \
                     the S4 slice of the multi-backend roadmap)",
                ));
            }
            _ => {
                return Err(candle_core::Error::msg(format!(
                    "unknown device API {api:?} in {s:?}: expected \"cpu\", \"cuda:<idx>\", \
                     \"hip:<idx>\", or \"vulkan:<idx>\""
                )));
            }
        };
        Ok(spec)
    }
}

/// What the serving stack can run on a device in this build shape.
#[derive(Clone, Copy, Debug)]
pub struct BackendCaps {
    /// Resolved device architecture; `None` on CPU.
    pub arch: Option<GpuArch>,
    /// Decode-graph capture (`MISTRALRS_CUDA_GRAPHS`).
    pub graphs: bool,
    /// Paged attention (block KV cache attention kernels).
    pub paged_attention: bool,
    /// GGUF quantized weights (`QStorage` on this role).
    pub quantized_gguf: bool,
    /// MTP speculative decoding.
    pub mtp: bool,
}

impl BackendCaps {
    /// Refuse a GGUF model early (before any weight loading) when this
    /// device cannot serve it, naming why.
    pub fn ensure_gguf(&self) -> Result<()> {
        if self.quantized_gguf {
            Ok(())
        } else {
            Err(candle_core::Error::msg(
                "GGUF weights cannot load on this device in this build: hip-side quantized \
                 storage (QStorage::Hip) is the S2 slice of the multi-backend roadmap. Serve \
                 GGUF from a single-vendor rocm build, or use unquantized weights here",
            ))
        }
    }

    fn log(&self, device: &Device) {
        let arch = self.arch.map(|a| format!(" {a:?}")).unwrap_or_default();
        let on = |b: bool| if b { "on" } else { "OFF" };
        tracing::info!(
            "[device] {device:?}{arch}: graphs={} paged_attn={} gguf={} mtp={}",
            on(self.graphs),
            on(self.paged_attention),
            on(self.quantized_gguf),
            on(self.mtp),
        );
    }
}

/// The serving-role probe: is this build's serving-role GPU runtime present?
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn serving_role_present() -> bool {
    #[cfg(not(all(feature = "cuda", feature = "rocm")))]
    {
        // Single-vendor builds: the compiled vendor's runtime.
        unsafe { candle_core::role::backend::cudarc::driver::sys::is_culib_present() }
    }
    #[cfg(all(feature = "cuda", feature = "rocm"))]
    {
        // Dual builds serve the AMD role (all C kernels are hipcc-built).
        unsafe { candle_core::hip_backend::cudarc::driver::sys::is_culib_present() }
    }
}

/// The NVIDIA runtime probe (dual builds and cuda-only builds).
#[cfg(feature = "cuda")]
fn nvidia_present() -> bool {
    unsafe { candle_core::cuda_backend::cudarc::driver::sys::is_culib_present() }
}

/// Resolve an explicit spec to a concrete device, probe-gated.
pub fn resolve(spec: &DeviceSpec) -> Result<Device> {
    match spec {
        DeviceSpec::Cpu => Ok(Device::Cpu),
        DeviceSpec::Cuda(idx) => {
            #[cfg(not(any(feature = "cuda", feature = "rocm")))]
            {
                let _ = idx;
                candle_core::bail!("this binary has no GPU backend: use \"cpu\"");
            }
            #[cfg(all(feature = "cuda", not(feature = "rocm")))]
            {
                if !nvidia_present() {
                    candle_core::bail!(
                        "device \"cuda:{idx}\" was requested but no NVIDIA runtime is \
                         present (is_culib_present() == false)"
                    );
                }
                Device::new_cuda(*idx)
            }
            #[cfg(all(feature = "rocm", not(feature = "cuda")))]
            {
                // Single-vendor rocm: "cuda:N" is the compiled (HIP) vendor -
                // the upstream semantic, kept for config compatibility.
                if !serving_role_present() {
                    candle_core::bail!(
                        "device \"cuda:{idx}\" was requested but no HIP runtime is present"
                    );
                }
                Device::new_cuda(*idx)
            }
            #[cfg(all(feature = "cuda", feature = "rocm"))]
            {
                // Dual builds: the NVIDIA role. It has no serving kernels
                // (everything hipcc-built), so this resolves for explicit
                // requests but reports zero capability caps.
                if !nvidia_present() {
                    candle_core::bail!(
                        "device \"cuda:{idx}\" was requested but no NVIDIA runtime is \
                         present (is_culib_present() == false); dual builds serve the AMD \
                         role - use \"hip:N\""
                    );
                }
                Device::new_cuda(*idx)
            }
        }
        DeviceSpec::Hip(idx) => {
            #[cfg(not(any(feature = "cuda", feature = "rocm")))]
            {
                let _ = idx;
                candle_core::bail!("this binary has no GPU backend: use \"cpu\"");
            }
            #[cfg(feature = "rocm")]
            {
                if !serving_role_present() {
                    candle_core::bail!(
                        "device \"hip:{idx}\" was requested but no HIP runtime is present"
                    );
                }
                // Dual builds: the AMD role directly. Single-vendor rocm:
                // "hip:N" aliases the compiled vendor (portable configs).
                #[cfg(all(feature = "cuda", feature = "rocm"))]
                return Device::new_hip(*idx);
                #[cfg(not(feature = "cuda"))]
                return Device::new_cuda(*idx);
            }
            #[cfg(all(feature = "cuda", not(feature = "rocm")))]
            {
                let _ = idx;
                candle_core::bail!(
                    "this binary has no HIP backend (built cuda-only): use \"cuda:N\" or \"cpu\""
                );
            }
        }
    }
}

/// Automatic device selection when no explicit device is configured:
/// the native GPU serving this build's role if its runtime is present,
/// else CPU - with the choice logged so "best" is auditable, not magic.
/// (Metal builds keep their existing selection path in the server core.)
pub fn auto_select() -> Result<Device> {
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        tracing::info!("[device] auto-selected cpu (no GPU backend in this binary)");
        return Ok(Device::Cpu);
    }
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        if serving_role_present() {
            let device = resolve_serving_role(0)?;
            tracing::info!(
                "[device] auto-selected the serving-role GPU (native API, runtime probed): \
                 {device:?}"
            );
            return Ok(device);
        }
        #[cfg(all(feature = "cuda", feature = "rocm"))]
        {
            // Dual builds serve the AMD role; an NVIDIA-only machine cannot
            // serve here (all C kernels are hipcc-built). Say so instead of
            // silently falling to a CPU that would drown.
            if nvidia_present() {
                candle_core::bail!(
                    "auto-selection found an NVIDIA runtime but this is a dual cuda+rocm \
                     build, which serves the AMD role (hipcc-built kernels). Use \"hip:N\" \
                     with a HIP runtime present, or build single-vendor cuda for NVIDIA \
                     serving"
                );
            }
        }
        tracing::info!("[device] auto-selected cpu (no GPU runtime probed)");
        Ok(Device::Cpu)
    }
}

/// Construct serving-role device 0 on a machine where the probe passed.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn resolve_serving_role(idx: usize) -> Result<Device> {
    #[cfg(all(feature = "cuda", feature = "rocm"))]
    return Device::new_hip(idx);
    #[cfg(not(feature = "cuda"))]
    return Device::new_cuda(idx);
    #[cfg(all(feature = "cuda", not(feature = "rocm")))]
    return Device::new_cuda(idx);
}

/// Resolve and report: the entry point the server uses. Logs the caps line
/// for the chosen device before returning.
pub fn resolve_and_report(spec: Option<&DeviceSpec>) -> Result<Device> {
    let device = match spec {
        Some(spec) => resolve(spec)?,
        None => auto_select()?,
    };
    let caps = backend_caps(&device);
    caps.log(&device);
    Ok(device)
}

/// Capability contract for a device in this build shape.
#[allow(unused_variables)]
pub fn backend_caps(device: &Device) -> BackendCaps {
    match device {
        Device::Cpu => BackendCaps {
            arch: None,
            graphs: false,
            paged_attention: false,
            quantized_gguf: false,
            mtp: false,
        },
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        Device::Cuda(dev) => {
            let arch = GpuArch::resolve(dev).ok();
            #[cfg(not(all(feature = "cuda", feature = "rocm")))]
            {
                // Single-vendor builds: the fork's full serving stack.
                BackendCaps {
                    arch,
                    graphs: true,
                    paged_attention: true,
                    quantized_gguf: true,
                    mtp: true,
                }
            }
            #[cfg(all(feature = "cuda", feature = "rocm"))]
            {
                // Dual builds: the NVIDIA role resolves for explicit requests
                // but has no serving kernels (everything is hipcc-built).
                BackendCaps {
                    arch,
                    graphs: false,
                    paged_attention: false,
                    quantized_gguf: false,
                    mtp: false,
                }
            }
        }
        #[cfg(all(feature = "cuda", feature = "rocm"))]
        Device::Hip(dev) => {
            let arch = GpuArch::resolve_hip(dev).ok();
            // The AMD role: graphs/paged/mtp serve; GGUF waits for S2
            // (QStorage::Hip).
            BackendCaps {
                arch,
                graphs: true,
                paged_attention: true,
                quantized_gguf: false,
                mtp: true,
            }
        }
        // Metal: the variant exists in every build; serving capability is
        // reported the same either way (informational in non-metal builds,
        // where the dummy type is never constructed).
        Device::Metal(_) => BackendCaps {
            arch: None,
            graphs: false,
            paged_attention: false,
            quantized_gguf: true,
            mtp: false,
        },
        // CPU-only builds: the Cuda variant exists in the enum but no GPU
        // backend is compiled, so such a device can never be constructed.
        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        Device::Cuda(_) => unreachable!("no GPU backend compiled"),
    }
}

// Serde in the string form (`device = "hip:0"` in config files), not the
// enum form: configs stay human-shaped and validation runs at parse time.
impl serde::Serialize for DeviceSpec {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for DeviceSpec {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        DeviceSpec::parse(&s).map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Display for DeviceSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda(idx) => write!(f, "cuda:{idx}"),
            Self::Hip(idx) => write!(f, "hip:{idx}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_forms() {
        assert_eq!(DeviceSpec::parse("cpu").unwrap(), DeviceSpec::Cpu);
        assert_eq!(DeviceSpec::parse(" cpu ").unwrap(), DeviceSpec::Cpu);
        assert_eq!(DeviceSpec::parse("cuda:0").unwrap(), DeviceSpec::Cuda(0));
        assert_eq!(DeviceSpec::parse("HIP:7").unwrap(), DeviceSpec::Hip(7));
        // Vulkan parses as a known API but reports the S4 status.
        let err = DeviceSpec::parse("vulkan:1").unwrap_err().to_string();
        assert!(err.contains("S4"), "{err}");
        // Malformed forms list the valid ones.
        for bad in ["cuda:x", "cuda", "gpu:0", "", "cuda:0:1"] {
            let err = DeviceSpec::parse(bad).unwrap_err().to_string();
            assert!(err.contains("cuda:<idx>"), "{bad}: {err}");
        }
    }

    #[test]
    fn serde_roundtrip() {
        let spec = DeviceSpec::parse("hip:3").unwrap();
        let s = serde_json::to_string(&spec).unwrap();
        assert_eq!(s, "\"hip:3\"");
        assert_eq!(serde_json::from_str::<DeviceSpec>(&s).unwrap(), spec);
    }

    /// Dual builds: `hip:N` routes to the AMD role with its capability
    /// contract (GGUF gated on S2), auto-selection lands on the serving
    /// role, and an explicit NVIDIA request without a driver errors
    /// cleanly through the probe gate.
    #[cfg(all(feature = "cuda", feature = "rocm"))]
    #[test]
    fn dual_routing_and_caps() {
        let device = resolve(&DeviceSpec::parse("hip:0").unwrap()).unwrap();
        assert!(device.is_hip());
        let caps = backend_caps(&device);
        assert!(caps.arch.is_some(), "arch resolves on the AMD role");
        assert!(caps.graphs && caps.paged_attention && caps.mtp);
        assert!(!caps.quantized_gguf, "GGUF waits for S2 on the hip role");
        let gate = caps.ensure_gguf().unwrap_err().to_string();
        assert!(gate.contains("S2"), "{gate}");

        let auto = auto_select().unwrap();
        assert!(
            auto.is_hip(),
            "auto-selection serves the AMD role in dual builds"
        );

        // Without an NVIDIA runtime the explicit request must refuse with
        // the probe error naming the alternative - on driver-equipped
        // boxes this assertion is skipped by the probe check below.
        if !unsafe { candle_core::cuda_backend::cudarc::driver::sys::is_culib_present() } {
            let err = resolve(&DeviceSpec::parse("cuda:0").unwrap())
                .unwrap_err()
                .to_string();
            assert!(err.contains("no NVIDIA runtime"), "{err}");
            assert!(err.contains("hip:N"), "{err}");
        }
    }

    /// Single-vendor rocm builds: `hip:N` aliases the compiled vendor, so
    /// configs stay portable across build shapes, and the full serving
    /// stack reports capable (including GGUF).
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    #[test]
    fn rocm_alias_routing() {
        let device = resolve(&DeviceSpec::parse("hip:0").unwrap()).unwrap();
        assert!(device.is_cuda(), "hip:N aliases the compiled vendor");
        let caps = backend_caps(&device);
        assert!(caps.quantized_gguf, "single-vendor builds serve GGUF");
        assert!(caps.graphs && caps.paged_attention);
        let auto = auto_select().unwrap();
        assert!(auto.is_cuda());
    }
}
