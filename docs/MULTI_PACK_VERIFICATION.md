# Multi-Pack release verification

Checked on 2026-10-27 on macOS arm64 with Rust 1.99.1. The original Alpha 44
archive predated these multi-Pack changes and has now been refreshed.

- The 27 Deployer release tests, Bash syntax checks, and workflow lint passed.
- Public Pack release `pack-v1.5.0-alpha-2` built successfully. FCS downloaded
  both ZIPs and their sidecars from GitHub and validated their internal
  inventories before building a combined registry.
- Local Debug and Release Gateway and FCS binaries, two simulated Slurm ranks,
  and the repository dummy vLLM passed eight integration checks per build. The
  test used both public release ZIPs, a separate diagnostic Pack, per-Pack
  inventory and archive pins, mTLS, discovery, service routing, and cleanup.
- The real Bash entry passed seven local checks in both public HTTP and HTTPS
  modes with both ZIPs. Each retained internal Gateway/FCS mTLS.
- Gateway and FCS library tests passed 281/281 and 272/272, respectively, with
  local loopback access enabled. The Pack repository's six packaging tests passed.
- Existing real FCS network and authentication fixtures passed 22 and 64
  checks. Tapis contracts, vLLM contracts and archive packaging passed 3, 13
  and 2 tests, respectively.
- Gateway and FCS alpha-2 releases each passed native Linux x86_64 and aarch64
  Actions builds, tests, ABI checks, and publication. Pack alpha-2 passed its
  release workflow. Deployer run `40621197855` validated the released inputs,
  standalone and combined Packs, assembled the bundle, and refreshed Alpha 44.
- The archive downloaded from the fixed Alpha 44 URL has SHA-256
  `764a33c8399687a27b5d0b59462b86fe75ef32c8d17d92b11315638e567c05f9`.
  Its 19 unique files and complete 18-file inventory passed verification; it
  includes four binaries and `pack-tag.txt`, and contains no Pack ZIP. BUILD.json
  binds the three alpha-2 component tags and Deployer commit `150c3d8`.
  The original Alpha 44 Git tag still points to commit `2cdd56a`.

The local fixture uses simulated Slurm and native macOS executables. It does not
prove the Linux ABI, real TACC/OSC filesystem permissions, a Slurm allocation,
site access, physical GPUs, or a real vLLM installation. Checksums retrieved from the
same release detect corruption but are not publisher signatures.
