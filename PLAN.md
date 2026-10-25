# FlexServ 1.5 Alpha delivery

Approved on 2026-10-25. Ship only Gateway and Capability Server, one v1alpha2 Pack, and simple Slurm Bash entrypoints. Preserve unrelated files.

- [x] Verify remote branches and establish the official FCS checkout. Proof: remote SHA and clean tracked state.
- [x] Remove embedded inference, rename FCS and integrate probe/job startup. Proof: Rust checks and real local job TLS/discovery after staging deletion.
- [x] Consolidate Packs in FlexServ-FCS-Pack. Proof: inventory, dependencies, visibility and runtime-generated poll approvals.
- [x] Implement environment/install/site/launch/tunnel scripts. Proof: bash syntax and controlled lifecycle cases, without installing models or changing host services.
- [ ] Publish reusable dual-architecture CI and complete ZIP. Proof: Actions results, ELF ABI gates, exact inventory and checksums.
- [x] Run Debug/Release dummy and Tapis fixture regression, review and record remaining site-only checks.

No GPU/HPC result may be inferred from fixtures. No restart recovery, PID adoption, live external Tapis resources or automatic model download.

## Verification checkpoint

Nine public packaging/script tests pass. The actual Bash entry passes five checks with real Release Gateway/FCS and full 65-tool Pack; only uname/Slurm are explicitly fixture replacements. TACC/OSC/GPU installation remains unexecuted. All new Actions workflows pass actionlint. Shared Linux scripts are used by component and Deployer CI. Private-source publishing requires FLEXSERV_SOURCE_TOKEN (or an existing all-source read token); authentication and fixed-revision checkout now pass. Never request/publish its value.
