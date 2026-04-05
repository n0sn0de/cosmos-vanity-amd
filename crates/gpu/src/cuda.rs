//! CUDA GPU acceleration for Cosmos vanity address hashing.
//!
//! This backend reuses the existing OpenCL kernel sources and compiles them for
//! CUDA/NVRTC using a thin compatibility preamble. That keeps the CUDA path
//! behavior-aligned with the existing OpenCL implementation instead of growing
//! a second, drifting kernel codebase.

use std::sync::Arc;

use cudarc::driver::{
    result::DriverError, CudaContext as DriverContext, CudaFunction, CudaModule, CudaStream,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileError, CompileOptions};
use thiserror::Error;
use tracing::{debug, info};

/// Size of a compressed secp256k1 public key.
const PUBKEY_SIZE: usize = 33;
/// Size of a RIPEMD-160 hash (Cosmos address hash).
const HASH_SIZE: usize = 20;
/// Size of a raw private key.
const PRIVKEY_SIZE: usize = 32;

const CUDA_COMPAT_PREAMBLE: &str = r#"
typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned long long ulong;

#define __kernel extern "C" __global__
#define __global
#define __constant __constant__
#define get_global_id(dim) (blockIdx.x * blockDim.x + threadIdx.x)
"#;

const HASH_KERNEL_SOURCE: &str = include_str!("kernels/vanity_search.cl");
const SECP256K1_KERNEL_SOURCE: &str = include_str!("kernels/secp256k1.cl");
const MNEMONIC_KERNEL_SOURCE: &str = include_str!("kernels/mnemonic_pipeline.cl");

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("No CUDA device found")]
    NoDevice,
    #[error("CUDA runtime unavailable: {0}")]
    RuntimeUnavailable(String),
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),
    #[error("CUDA compilation error: {0}")]
    Nvrtc(String),
    #[error("GPU batch size must be > 0")]
    InvalidBatchSize,
}

impl From<CompileError> for GpuError {
    fn from(value: CompileError) -> Self {
        Self::Nvrtc(value.to_string())
    }
}

/// Check if CUDA acceleration is available.
pub fn is_available() -> bool {
    std::panic::catch_unwind(|| match DriverContext::device_count() {
        Ok(count) if count > 0 => DriverContext::new(0).is_ok(),
        _ => false,
    })
    .unwrap_or(false)
}

/// CUDA context holding the stream, compiled modules, and device info.
pub struct GpuContext {
    stream: Arc<CudaStream>,
    hash_function: CudaFunction,
    secp256k1_function: Option<CudaFunction>,
    #[cfg(test)]
    mnemonic_module: Option<Arc<CudaModule>>,
    mnemonic_function: Option<CudaFunction>,
    device_name: String,
    max_threads_per_block: u32,
    max_compute_units: u32,
}

impl GpuContext {
    /// Initialize CUDA context — finds an NVIDIA GPU and compiles the kernels.
    pub fn new() -> Result<Self, GpuError> {
        let device_count =
            std::panic::catch_unwind(DriverContext::device_count).map_err(|_| {
                GpuError::RuntimeUnavailable(
                    "CUDA driver library could not be loaded on this machine".to_string(),
                )
            })??;
        if device_count <= 0 {
            return Err(GpuError::NoDevice);
        }

        let ctx = std::panic::catch_unwind(|| DriverContext::new(0)).map_err(|_| {
            GpuError::RuntimeUnavailable(
                "CUDA context creation panicked while loading the driver".to_string(),
            )
        })??;
        let stream = ctx.default_stream();

        let device_name = ctx
            .name()
            .unwrap_or_else(|_| "Unknown NVIDIA GPU".to_string());
        let max_compute_units = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )?
            .max(1) as u32;

        let hash_module = compile_module(
            &ctx,
            &cuda_source(&[HASH_KERNEL_SOURCE]),
            "vanity_search.cu",
        )?;
        let hash_function = hash_module.load_function("compute_address_hashes")?;
        let max_threads_per_block = hash_function.max_threads_per_block()?.max(1) as u32;
        info!(
            "CUDA device: {} (SMs: {}, max threads/block: {})",
            device_name, max_compute_units, max_threads_per_block
        );
        info!("CUDA hash kernel compiled successfully");

        let secp256k1_function = match compile_module(
            &ctx,
            &cuda_source(&[SECP256K1_KERNEL_SOURCE]),
            "secp256k1.cu",
        ) {
            Ok(module) => match module.load_function("generate_addresses") {
                Ok(function) => {
                    info!("CUDA secp256k1 kernel compiled successfully");
                    Some(function)
                }
                Err(err) => {
                    tracing::warn!(
                        "Failed to load CUDA secp256k1 kernel: {err}. Raw key mode will be unavailable."
                    );
                    None
                }
            },
            Err(err) => {
                tracing::warn!(
                    "Failed to compile CUDA secp256k1 kernel: {err}. Raw key mode will be unavailable."
                );
                None
            }
        };

        let mnemonic_kernel = match compile_module(
            &ctx,
            &cuda_source(&[SECP256K1_KERNEL_SOURCE, MNEMONIC_KERNEL_SOURCE]),
            "mnemonic_pipeline.cu",
        ) {
            Ok(module) => match module.load_function("mnemonic_to_address") {
                Ok(function) => {
                    info!("CUDA mnemonic pipeline kernel compiled successfully");
                    (Some(module), Some(function))
                }
                Err(err) => {
                    tracing::warn!(
                        "Failed to load CUDA mnemonic pipeline kernel: {err}. GPU mnemonic mode will be unavailable."
                    );
                    (None, None)
                }
            },
            Err(err) => {
                tracing::warn!(
                    "Failed to compile CUDA mnemonic pipeline kernel: {err}. GPU mnemonic mode will be unavailable."
                );
                (None, None)
            }
        };
        #[cfg(test)]
        let (mnemonic_module, mnemonic_function) = mnemonic_kernel;
        #[cfg(not(test))]
        let (_mnemonic_module, mnemonic_function) = mnemonic_kernel;

        Ok(Self {
            stream,
            hash_function,
            secp256k1_function,
            #[cfg(test)]
            mnemonic_module,
            mnemonic_function,
            device_name,
            max_threads_per_block,
            max_compute_units,
        })
    }

    /// Device name string.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Max compute units / SM count.
    pub fn max_compute_units(&self) -> u32 {
        self.max_compute_units
    }

    /// Check if the secp256k1 kernel is available.
    pub fn has_secp256k1_kernel(&self) -> bool {
        self.secp256k1_function.is_some()
    }

    /// Check if the mnemonic pipeline kernel is available.
    pub fn has_mnemonic_kernel(&self) -> bool {
        self.mnemonic_function.is_some()
    }

    /// Compute SHA-256 → RIPEMD-160 hashes for a batch of compressed public keys on CUDA.
    pub fn hash_pubkeys_batch(&self, pubkeys: &[u8]) -> Result<Vec<u8>, GpuError> {
        let n = pubkeys.len() / PUBKEY_SIZE;
        if n == 0 {
            return Err(GpuError::InvalidBatchSize);
        }
        debug!("CUDA hashing batch of {} pubkeys", n);

        let pubkey_buf = self.stream.clone_htod(pubkeys)?;
        let mut hash_buf = self.stream.alloc_zeros::<u8>(n * HASH_SIZE)?;
        let prefix_buf = self.stream.alloc_zeros::<u8>(1)?;
        let mut matches_buf = self.stream.alloc_zeros::<u32>(n)?;

        let prefix_len = 0u32;
        let count = n as u32;
        let launch = self.launch_config(count);

        unsafe {
            self.stream
                .launch_builder(&self.hash_function)
                .arg(&pubkey_buf)
                .arg(&mut hash_buf)
                .arg(&prefix_buf)
                .arg(&prefix_len)
                .arg(&mut matches_buf)
                .arg(&count)
                .launch(launch)?;
        }
        self.stream.synchronize()?;

        let hashes = self.stream.clone_dtoh(&hash_buf)?;
        debug!("CUDA batch complete: {} hashes computed", n);
        Ok(hashes)
    }

    /// Compute hashes and check for prefix matches on CUDA.
    pub fn hash_and_match_batch(
        &self,
        pubkeys: &[u8],
        prefix_bytes: &[u8],
    ) -> Result<(Vec<u8>, Vec<u32>), GpuError> {
        let n = pubkeys.len() / PUBKEY_SIZE;
        if n == 0 {
            return Err(GpuError::InvalidBatchSize);
        }

        let pubkey_buf = self.stream.clone_htod(pubkeys)?;
        let mut hash_buf = self.stream.alloc_zeros::<u8>(n * HASH_SIZE)?;
        let mut matches_buf = self.stream.alloc_zeros::<u32>(n)?;
        let prefix_len = prefix_bytes.len() as u32;
        let prefix_buf = if prefix_bytes.is_empty() {
            self.stream.alloc_zeros::<u8>(1)?
        } else {
            self.stream.clone_htod(prefix_bytes)?
        };
        let count = n as u32;
        let launch = self.launch_config(count);

        unsafe {
            self.stream
                .launch_builder(&self.hash_function)
                .arg(&pubkey_buf)
                .arg(&mut hash_buf)
                .arg(&prefix_buf)
                .arg(&prefix_len)
                .arg(&mut matches_buf)
                .arg(&count)
                .launch(launch)?;
        }
        self.stream.synchronize()?;

        Ok((
            self.stream.clone_dtoh(&hash_buf)?,
            self.stream.clone_dtoh(&matches_buf)?,
        ))
    }

    /// Generate public keys and address hashes from raw private keys entirely on CUDA.
    pub fn generate_and_hash_batch(
        &self,
        privkeys: &[u8],
        prefix_bytes: &[u8],
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u32>), GpuError> {
        let function = self
            .secp256k1_function
            .as_ref()
            .ok_or_else(|| GpuError::Nvrtc("secp256k1 kernel not compiled".to_string()))?;

        let n = privkeys.len() / PRIVKEY_SIZE;
        if n == 0 {
            return Err(GpuError::InvalidBatchSize);
        }
        debug!("CUDA secp256k1 batch: {} private keys", n);

        let privkey_buf = self.stream.clone_htod(privkeys)?;
        let mut pubkey_buf = self.stream.alloc_zeros::<u8>(n * PUBKEY_SIZE)?;
        let mut hash_buf = self.stream.alloc_zeros::<u8>(n * HASH_SIZE)?;
        let mut matches_buf = self.stream.alloc_zeros::<u32>(n)?;
        let prefix_len = prefix_bytes.len() as u32;
        let prefix_buf = if prefix_bytes.is_empty() {
            self.stream.alloc_zeros::<u8>(1)?
        } else {
            self.stream.clone_htod(prefix_bytes)?
        };
        let count = n as u32;
        let launch = self.launch_config(count);

        unsafe {
            self.stream
                .launch_builder(function)
                .arg(&privkey_buf)
                .arg(&mut pubkey_buf)
                .arg(&mut hash_buf)
                .arg(&prefix_buf)
                .arg(&prefix_len)
                .arg(&mut matches_buf)
                .arg(&count)
                .launch(launch)?;
        }
        self.stream.synchronize()?;

        Ok((
            self.stream.clone_dtoh(&pubkey_buf)?,
            self.stream.clone_dtoh(&hash_buf)?,
            self.stream.clone_dtoh(&matches_buf)?,
        ))
    }

    /// Suggested batch size for hybrid mode.
    pub fn suggested_batch_size(&self) -> usize {
        let warps_per_sm = 16;
        let warp_size = 32;
        let base = self.max_compute_units as usize * warps_per_sm * warp_size;
        base.max(32_768).next_power_of_two()
    }

    /// Batch size for pure GPU mode.
    pub fn pure_gpu_batch_size(&self) -> usize {
        let warps_per_sm = 32;
        let warp_size = 32;
        let base = self.max_compute_units as usize * warps_per_sm * warp_size;
        base.max(65_536).min(131_072).next_power_of_two()
    }

    /// Batch size for mnemonic GPU mode.
    pub fn mnemonic_batch_size(&self) -> usize {
        let warps_per_sm = 4;
        let warp_size = 32;
        let base = self.max_compute_units as usize * warps_per_sm * warp_size;
        base.max(2_048).min(8_192)
    }

    /// Run the full mnemonic pipeline on CUDA.
    pub fn mnemonic_batch(
        &self,
        mnemonics_flat: &[u8],
        mnemonic_lens: &[u32],
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u32>), GpuError> {
        let function = self
            .mnemonic_function
            .as_ref()
            .ok_or_else(|| GpuError::Nvrtc("Mnemonic pipeline kernel not compiled".to_string()))?;

        let n = mnemonic_lens.len();
        if n == 0 {
            return Err(GpuError::InvalidBatchSize);
        }
        debug!("CUDA mnemonic batch: {} candidates", n);

        let mnemonics_buf = self.stream.clone_htod(mnemonics_flat)?;
        let lens_buf = self.stream.clone_htod(mnemonic_lens)?;
        let mut privkeys_buf = self.stream.alloc_zeros::<u8>(n * 32)?;
        let mut hashes_buf = self.stream.alloc_zeros::<u8>(n * 20)?;
        let prefix_buf = self.stream.alloc_zeros::<u8>(1)?;
        let mut matches_buf = self.stream.alloc_zeros::<u32>(n)?;
        let prefix_len = 0u32;
        let count = n as u32;
        let launch = self.launch_config(count);

        unsafe {
            self.stream
                .launch_builder(function)
                .arg(&mnemonics_buf)
                .arg(&lens_buf)
                .arg(&mut privkeys_buf)
                .arg(&mut hashes_buf)
                .arg(&prefix_buf)
                .arg(&prefix_len)
                .arg(&mut matches_buf)
                .arg(&count)
                .launch(launch)?;
        }
        self.stream.synchronize()?;

        Ok((
            self.stream.clone_dtoh(&privkeys_buf)?,
            self.stream.clone_dtoh(&hashes_buf)?,
            self.stream.clone_dtoh(&matches_buf)?,
        ))
    }

    fn launch_config(&self, count: u32) -> LaunchConfig {
        let block = self.max_threads_per_block.min(256).max(1);
        let grid = count.div_ceil(block);
        LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    #[cfg(test)]
    fn mnemonic_module(&self) -> Option<Arc<CudaModule>> {
        self.mnemonic_module.clone()
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("device", &self.device_name)
            .field("compute_units", &self.max_compute_units)
            .field("max_threads_per_block", &self.max_threads_per_block)
            .finish()
    }
}

fn cuda_source(parts: &[&str]) -> String {
    let mut source = String::from(CUDA_COMPAT_PREAMBLE);
    for part in parts {
        source.push('\n');
        source.push_str(part);
        source.push('\n');
    }
    source
}

fn compile_module(
    ctx: &Arc<DriverContext>,
    src: &str,
    name: &str,
) -> Result<Arc<CudaModule>, GpuError> {
    let arch = ctx.compute_capability().ok().map(|(major, minor)| {
        Box::leak(format!("compute_{major}{minor}").into_boxed_str()) as &'static str
    });

    let ptx = compile_ptx_with_opts(
        src,
        CompileOptions {
            arch,
            name: Some(name.to_string()),
            options: vec!["--std=c++14".to_string()],
            ..Default::default()
        },
    )?;

    Ok(ctx.load_module(ptx)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend_tests::{self, BackendHarness};

    impl BackendHarness for GpuContext {
        fn label(&self) -> &'static str {
            "cuda"
        }

        fn hash_pubkeys_batch(&self, pubkeys: &[u8]) -> anyhow::Result<Vec<u8>> {
            Ok(GpuContext::hash_pubkeys_batch(self, pubkeys)?)
        }

        fn has_secp256k1_kernel(&self) -> bool {
            GpuContext::has_secp256k1_kernel(self)
        }

        fn generate_and_hash_batch(
            &self,
            privkeys: &[u8],
            prefix_bytes: &[u8],
        ) -> anyhow::Result<(Vec<u8>, Vec<u8>, Vec<u32>)> {
            Ok(GpuContext::generate_and_hash_batch(
                self,
                privkeys,
                prefix_bytes,
            )?)
        }

        fn has_mnemonic_kernel(&self) -> bool {
            GpuContext::has_mnemonic_kernel(self)
        }

        fn mnemonic_batch(
            &self,
            mnemonics_flat: &[u8],
            mnemonic_lens: &[u32],
        ) -> anyhow::Result<(Vec<u8>, Vec<u8>, Vec<u32>)> {
            Ok(GpuContext::mnemonic_batch(
                self,
                mnemonics_flat,
                mnemonic_lens,
            )?)
        }
    }

    #[test]
    fn test_is_available() {
        let _ = is_available();
    }

    #[test]
    fn test_cuda_hash_matches_cpu() {
        if !is_available() {
            eprintln!("No CUDA device available, skipping CUDA hash test");
            return;
        }

        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("Could not initialize CUDA backend: {err}, skipping");
                return;
            }
        };

        backend_tests::assert_hash_matches_cpu(&ctx);
    }

    #[test]
    fn test_cuda_secp256k1_known_vector() {
        if !is_available() {
            eprintln!("No CUDA device available, skipping CUDA secp256k1 test");
            return;
        }

        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("Could not initialize CUDA backend: {err}, skipping");
                return;
            }
        };

        backend_tests::assert_secp256k1_known_vector(&ctx);
    }

    #[test]
    fn test_cuda_secp256k1_matches_cpu() {
        if !is_available() {
            eprintln!("No CUDA device available, skipping CUDA secp256k1 CPU parity test");
            return;
        }

        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("Could not initialize CUDA backend: {err}, skipping");
                return;
            }
        };

        backend_tests::assert_secp256k1_matches_cpu(&ctx);
    }

    #[test]
    fn test_cuda_mnemonic_pipeline() {
        if !is_available() {
            eprintln!("No CUDA device available, skipping CUDA mnemonic pipeline test");
            return;
        }

        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("Could not initialize CUDA backend: {err}, skipping");
                return;
            }
        };

        backend_tests::assert_mnemonic_pipeline(&ctx);
    }

    #[test]
    fn test_cuda_kernel_source_contains_expected_entrypoints() {
        let hash_src = cuda_source(&[HASH_KERNEL_SOURCE]);
        assert!(hash_src.contains("compute_address_hashes"));

        let secp_src = cuda_source(&[SECP256K1_KERNEL_SOURCE]);
        assert!(secp_src.contains("generate_addresses"));

        let mnemonic_src = cuda_source(&[SECP256K1_KERNEL_SOURCE, MNEMONIC_KERNEL_SOURCE]);
        assert!(mnemonic_src.contains("mnemonic_to_address"));
        assert!(mnemonic_src.contains("test_sha512_kernel"));
    }

    #[test]
    fn test_cuda_mnemonic_module_keeps_diagnostic_kernels() {
        if !is_available() {
            eprintln!("No CUDA device available, skipping CUDA diagnostic module test");
            return;
        }

        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("Could not initialize CUDA backend: {err}, skipping");
                return;
            }
        };

        if let Some(module) = ctx.mnemonic_module() {
            module
                .load_function("test_sha512_kernel")
                .expect("missing diagnostic kernel");
            module
                .load_function("test_hmac_sha512_kernel")
                .expect("missing diagnostic kernel");
            module
                .load_function("test_pbkdf2_kernel")
                .expect("missing diagnostic kernel");
            module
                .load_function("test_bip32_kernel")
                .expect("missing diagnostic kernel");
        }
    }
}
