# FlexServ 1.5 Alpha delivery

## Stable Alpha App URL and persistent state (2026-10-25)

- [x] Keep the App URL at alpha-44 and promote only verified Alpha archives into
  that fixed channel. Freeze promotion when a published 1.5.0 Beta exists; never
  move alpha-44's Git tag. Prove filtering and digest checks with CLI fixtures.
- [x] Add FLEXSERV_RUNTIME_ROOT=HOST_EVAL($SCRATCH)/flexserv to the App, support
  explicit job overrides, and document its separation from per-job/venv state.
- [x] Clarify control-plane-only behavior and run unit/workflow/shell checks.
  Evidence: recordings/alpha-channel-2026-10-25/verification.json. Live promotion
  remains unexecuted; the existing Alpha 44 assets are unchanged by this update.

## Bash launcher correction (approved 2026-10-25)

- [x] Replace runtime shell/Python glue with Bash, jq and Linux utilities. Prove
  with syntax checks, a no-Python shell guard, checksum/Pack and launch fixtures.
- [x] Parse `--enable-https` and `--trust-remote-code`; keep PRI_MODEL_REPO and
  PUB_MODEL_REPO as environment inputs. The latter flag only exports a variable.
- [x] Default to public HTTP, use TAP certificates only for explicit HTTPS, and
  support secure random tokens and local login-port selection without TAP.
- [x] Add an explicit Gateway plaintext opt-in and focused configuration tests;
  keep internal job mTLS independent. Do not change FCS remote-code execution.
- [x] Generate the TACC FlexServ-1.5.0 / 1.5.0 App and registration instructions,
  validate definitions, publish a matching bundle and record actual checks.

Lessons: when deployment logic belongs in a shell entrypoint, use Bash and
structured CLI utilities rather than embedding Python programs. Remote-code
acceptance is an environment declaration only until backend support is approved.
`--enable-https` controls only the public listener. Internal Gateway/FCS transport
always uses the existing job-scoped mutual TLS, including external HTTP mode.

Approved on 2026-10-25. Ship only Gateway and Capability Server, one v1alpha2 Pack, and simple Slurm Bash entrypoints. Preserve unrelated files.

- [x] Verify remote branches and establish the official FCS checkout. Proof: remote SHA and clean tracked state.
- [x] Remove embedded inference, rename FCS and integrate probe/job startup. Proof: Rust checks and real local job TLS/discovery after staging deletion.
- [x] Consolidate Packs in FlexServ-FCS-Pack. Proof: inventory, dependencies, visibility and runtime-generated poll approvals.
- [x] Implement environment/install/site/launch/tunnel scripts. Proof: bash syntax and controlled lifecycle cases, without installing models or changing host services.
- [x] Publish reusable dual-architecture CI and complete ZIP. Proof: Actions results, ELF ABI gates, exact inventory and checksums.
- [x] Run Debug/Release dummy and Tapis fixture regression, review and record remaining site-only checks.

No GPU/HPC result may be inferred from fixtures. No restart recovery, PID adoption, live external Tapis resources or automatic model download.

## Publication corrections

All four native builds and ZIP assembly pass in run 40239046952. Initial release
creation failed because the workflow target differs from the default branch;
the standard token cannot authorize new workflow-bearing tags. Creating the
exact tag with the authorized Git connection resolved creation. Draft validation
then hit 404 through the public tag lookup. Re-plan publication around existing
tags, authenticated draft enumeration, numeric release ID and verified assets.
Keep builds unchanged and repeat actual publication after workflow lint/tests.

## Verification checkpoint

Ten public packaging/script tests pass locally (nine are in the frozen Alpha 43 release source). The actual Bash entry passes five checks with real Release Gateway/FCS and full 65-tool Pack; only uname/Slurm are explicitly fixture replacements. TACC/OSC/GPU installation remains unexecuted. All new Actions workflows pass actionlint. Shared Linux scripts are used by component and Deployer CI. Private-source publishing requires FLEXSERV_SOURCE_TOKEN (or an existing all-source read token); authentication and fixed-revision checkout now pass. Never request/publish its value.

## Alpha release completed (2026-10-25)

Published tapis-flexserv-1.5.0-alpha-43; Actions run 40241317109 succeeds.
Downloaded ZIP SHA-256: 50ff5ed683837fefbbe7b4bd48b640c81e9661608a1f70f80467dc752b65aac1.
All 16 archive files, four ELF identities, Pack inventory and source commits verify.
Real Slurm/TACC/OSC, GPU/RDMA/JIT and vLLM/MAX installation remain unexecuted;
strict Clippy baseline diagnostics are recorded rather than claimed as passing.

## Bash correction completed

Alpha 44 is published; run 40254076514 and downloaded archive verification pass.
Native Gateway279/FCS267 tests pass per Linux architecture, packaging16 and
HTTP/HTTPS real-host fixtures7 each pass. Both public modes retain internal mTLS
and reject no-client-certificate connections. The TACC App definition points to
the verified release; registration and real Slurm/TAP checks remain unexecuted.
# Release-only assembly (2026-10-26)

- [x] Preserve current work on feature/1.5.0-alpha. Remove Deployer Rust build jobs
  and download published Gateway/FCS/Pack assets from tapis-project repositories.
- [x] Resolve explicit release tags or latest successful releases on the approved
  branches once, pin release IDs, check inventories/digests and reject bad assets.
- [x] Merge the two verified Packs into the existing runtime catalog, recording
  component tags, commits and asset hashes in BUILD.json; preserve stable Alpha URL.
- [x] Run packaging, download-integrity, script and workflow checks. Record actual
  Actions execution separately; do not claim a native build from local fixtures.

- [x] Verify live component releases and final release-only ZIP assembly, including
  fixed Alpha 44 asset refresh and downloaded checksums.

Publication refinement: keep the existing Alpha 44 tag/release only. Automatic
review rejected an unnecessary extra Deployer Alpha tag; none was created. Use
verified branch assemblies plus run-specific Actions artifacts instead. Component
repository release tags remain independent and immutable to their publishers.

Live proof: Gateway 40352353290, FCS 40352353080, Pack 40352358386 and Deployer
40359644218 all succeed. Downloaded fixed Alpha 44 has SHA-256
04fe8bf6a45524f4c6a944dda57c359a0bbb2f8749babfcf22dc8ecfe023b945; 19 files,
four verified ELF binaries and the 65-tool runtime Pack. Original Alpha 44 tag
remains 2cdd56a01b4adb3bfd559e810a0a5d22d50c31ad. No new Deployer tag exists.
