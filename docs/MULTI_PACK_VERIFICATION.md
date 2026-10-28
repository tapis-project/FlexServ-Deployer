# Multi-Pack release verification

Checked on 2026-10-27 on macOS arm64 with Rust 1.99.1. This record concerns
the multi-Pack changes; the previously published Alpha 44 archive predates them.

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
- FCS library tests passed 272/272 with local loopback access enabled. The Pack
  repository's six packaging tests passed.

The local fixture uses simulated Slurm and native macOS executables. It does not
prove the Linux ABI, real TACC/OSC filesystem permissions, a Slurm allocation,
site access, physical GPUs, or a real vLLM installation. Current component
and Deployer Actions outcomes must be checked independently; historical Alpha
44 success does not certify the new assembly. Checksums retrieved from the
same release detect corruption but are not publisher signatures.
