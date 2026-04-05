# cosmos-vanity-amd ⚡

Generate vanity wallet addresses for the Cosmos ecosystem with GPU acceleration. Supports AMD/ROCm/OpenCL and modern NVIDIA/CUDA backends, including full secp256k1 elliptic curve math and BIP-39/BIP-32 derivation on GPU.

## Performance

Benchmarked on AMD Radeon RX 9070 XT (gfx1201, 32 CUs, ROCm 7.2):

| Mode | Throughput | Description |
|------|-----------|-------------|
| **GPU raw** | ~1,000,000/s | Private key → secp256k1 → hash on GPU |
| **GPU mnemonic** | ~21,000/s | Full BIP-39 pipeline (PBKDF2 + BIP-32 + secp256k1) on GPU |
| CPU mnemonic | ~12,000/s | Traditional CPU-only search |

### Expected search times (GPU raw mode)

| Prefix | Attempts | Time |
|--------|----------|------|
| 3 chars | ~32K | < 1s |
| 4 chars | ~1M | ~1s |
| 5 chars | ~33M | ~30s |
| 6 chars | ~1B | ~17min |
| 7 chars | ~34B | ~9h |

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          CLI (clap)                                  │
│  --mode gpu|hybrid|cpu  --gpu-api auto|opencl|cuda  --key-mode ...   │
├────────────┬───────────┬───────────────────┬──────────┬──────────────┤
│  keyderiv  │  address  │       gpu         │  verify  │    bench     │
│            │           │                   │          │              │
│ BIP-39     │ SHA-256   │ GPU Kernels:      │ CPU re-  │ Throughput   │
│ BIP-32/44  │ RIPEMD160 │  • secp256k1.cl   │ derive   │ measurement  │
│ secp256k1  │ Bech32    │  • vanity_search  │ & verify │              │
│            │           │  • mnemonic_pipe  │          │              │
└────────────┴───────────┴───────────────────┴──────────┴──────────────┘
```

### GPU Backends

- **OpenCL**: native path for AMD/ROCm and other OpenCL-capable GPUs
- **CUDA**: NVIDIA path via the CUDA driver + NVRTC
- **Parity approach**: the CUDA backend reuses the existing OpenCL kernel sources through a thin compatibility preamble, so both backends run the same kernel bodies instead of maintaining two diverging implementations

### GPU Kernels

| Kernel | Operations | Used by |
|--------|-----------|---------|
| `secp256k1.cl` | 256-bit field math, Jacobian EC point ops, scalar mul, SHA-256, RIPEMD-160, prefix matching | `--key-mode raw` |
| `mnemonic_pipeline.cl` | SHA-512, HMAC-SHA512, PBKDF2 (2048 rounds), BIP-32 derivation, + secp256k1 pipeline | `--key-mode mnemonic` |
| `vanity_search.cl` | SHA-256, RIPEMD-160, prefix matching | hybrid/legacy modes |

### Data Flow

**Raw mode** (fastest):
```
CPU: Random 32-byte private key → GPU
GPU: secp256k1(privkey) → pubkey → SHA-256 → RIPEMD-160 → prefix match
CPU: Verify match, output address + private key
```

**Mnemonic mode** (BIP-39 compatible):
```
CPU: Random entropy → BIP-39 mnemonic (word lookup) → GPU
GPU: PBKDF2(mnemonic, 2048 rounds) → BIP-32 derive → secp256k1 → SHA-256 → RIPEMD-160 → match
CPU: Verify match, output address + mnemonic phrase
```

## Build Instructions (Ubuntu 24.04)

### Prerequisites

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Build essentials
sudo apt install -y build-essential pkg-config libssl-dev
```

### CPU-only build

```bash
cargo build --release
```

### GPU build (AMD ROCm/OpenCL)

```bash
# Install ROCm (Ubuntu 24.04)
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/noble/amdgpu-install_6.4.60400-1_all.deb
sudo dpkg -i amdgpu-install_6.4.60400-1_all.deb
sudo amdgpu-install --usecase=rocm,opencl

# Install OpenCL dev headers
sudo apt install -y rocm-opencl-dev ocl-icd-opencl-dev

# Add user to GPU groups
sudo usermod -aG render,video $USER
# Log out and back in (or reboot)

# Verify GPU is detected
clinfo | grep "Board name"

# Build with OpenCL support
cargo build --release --features opencl
```

### GPU build (NVIDIA CUDA)

```bash
# Install the NVIDIA driver and CUDA toolkit with NVRTC available.
# Ubuntu packages vary by driver/toolkit version; verify these on your box:
#   nvidia-smi
#   nvcc --version
#   ls /usr/lib/x86_64-linux-gnu/libcuda.so* /usr/lib/x86_64-linux-gnu/libnvrtc.so* 2>/dev/null

# Build with CUDA support
cargo build --release --features cuda
```

### Multi-backend build

```bash
# Build both backends; runtime selection prefers CUDA, then OpenCL
cargo build --release --features "opencl cuda"
```

## Usage

### Quick start

```bash
# Fastest: raw private key mode (GPU default)
cosmos-vanity search -p abc

# Force CUDA on NVIDIA
cosmos-vanity search -p abc --gpu-api cuda

# Force OpenCL
cosmos-vanity search -p abc --gpu-api opencl

# With mnemonic phrase output (12-word, Keplr compatible)
cosmos-vanity search -p abc -k mnemonic

# 24-word mnemonic
cosmos-vanity search -p abc -k mnemonic -w 24

# Different chain (Osmosis)
cosmos-vanity search -p cool --hrp osmo

# CPU only (no GPU needed)
cosmos-vanity search -p abc -m cpu -k mnemonic
```

### GPU backend API (`--gpu-api`)

| API | Flag | Description |
|-----|------|-------------|
| **auto** | `--gpu-api auto` | Prefer CUDA when available, otherwise OpenCL |
| opencl | `--gpu-api opencl` | Force OpenCL |
| cuda | `--gpu-api cuda` | Force CUDA |

### Search modes (`--mode`, `-m`)

| Mode | Flag | Description |
|------|------|-------------|
| **gpu** | `-m gpu` | Pure GPU — all CPU cores feed keys, GPU does all compute. **Default.** |
| hybrid | `-m hybrid` | CPU search threads + GPU pipeline in parallel |
| cpu | `-m cpu` | CPU only, no GPU acceleration |

### Key modes (`--key-mode`, `-k`)

| Mode | Flag | Output | Speed |
|------|------|--------|-------|
| **raw** | `-k raw` | Hex private key | ~1M/s on GPU |
| mnemonic | `-k mnemonic` | BIP-39 seed phrase | ~21K/s on GPU |

### Mnemonic word count (`--words`, `-w`)

| Words | Flag | Entropy | Compatible with |
|-------|------|---------|----------------|
| **12** | `-w 12` | 128-bit | Keplr, MetaMask, Cosmostation (default for new wallets) |
| 24 | `-w 24` | 256-bit | Ledger, older wallets, maximum security |

### All search options

```bash
cosmos-vanity search [OPTIONS] --pattern <PATTERN>

Options:
  -p, --pattern <PATTERN>      Pattern to search for
  -t, --match-type <TYPE>      prefix (default), suffix, contains, regex
      --hrp <HRP>              Chain prefix: cosmos, osmo, juno, etc. [default: cosmos]
  -m, --mode <MODE>            gpu, hybrid, cpu [default: gpu]
      --gpu-api <API>          auto, opencl, cuda [default: auto]
  -k, --key-mode <MODE>        raw, mnemonic [default: raw]
  -w, --words <N>              Mnemonic words: 12 or 24 [default: 12]
  -j, --threads <N>            CPU threads [default: all cores]
  -n, --max-matches <N>        Stop after N matches [default: 1]
      --path <PATH>            BIP-44 derivation path [default: m/44'/118'/0'/0/0]
      --state-file <FILE>      Save/resume search state
      --format <FMT>           text or json [default: text]
```

### Other commands

```bash
# Generate a random address
cosmos-vanity generate --hrp cosmos

# Verify a mnemonic produces an address
cosmos-vanity verify --mnemonic "word1 word2 ..." --address "cosmos1..."

# Run benchmarks
cosmos-vanity bench --iterations 10000
```

### Supported chains

Any Cosmos SDK chain — just set `--hrp`:

| Chain | HRP |
|-------|-----|
| Cosmos Hub | `cosmos` |
| Osmosis | `osmo` |
| Juno | `juno` |
| Stargaze | `stars` |
| Akash | `akash` |
| Celestia | `celestia` |
| Injective | `inj` |
| dYdX | `dydx` |
| Noble | `noble` |
| ... | any bech32 HRP |

### Bech32 character set

Valid pattern characters: `qpzry9x8gf2tvdw0s3jn54khce6mua7l`

Note: letters `b`, `i`, `o`, `1` are NOT valid in Bech32.

## Security

### ⚠️ CRITICAL

- **Mnemonic/private key = full wallet control.** Never share, log, or store unencrypted.
- **Zeroize** — sensitive memory is cleared on drop via the `zeroize` crate.
- **GPU memory** — VRAM may retain data after kernel execution. Power cycle GPU for production keys.
- **Verification** — every GPU match is independently re-derived and verified on CPU.

### Recommended practices

```bash
# Disable swap
sudo swapoff -a

# Disable core dumps
ulimit -c 0

# Run on air-gapped machine for production keys
# Use encrypted disk
# Clear terminal history after use
# Verify generated address with an independent wallet before funding
```

## Development

```bash
# CPU-only tests (no GPU needed)
cargo test --workspace

# OpenCL validation on available OpenCL hardware
cargo test --workspace --features opencl

# CUDA compile + runtime-skip validation (runs real CUDA tests when a CUDA driver is present)
cargo test -p cosmos-vanity-gpu --features cuda

# Combined build/test surface
cargo test --workspace --features "opencl cuda"

# Lint / build checks
cargo clippy --workspace --features "opencl cuda"
cargo check --workspace --features "opencl cuda"
```

## Crate structure

| Crate | Purpose |
|-------|---------|
| `cli` | Command-line interface, progress display |
| `keyderiv` | BIP-39 mnemonic, BIP-32/44 HD derivation, secp256k1 |
| `address` | SHA-256, RIPEMD-160, Bech32 encoding, pattern matching |
| `gpu` | OpenCL/CUDA kernel management, GPU search engine, CPU fallback |
| `verify` | Independent CPU verification of matches |
| `bench` | Benchmarking utilities |

## License

MIT
