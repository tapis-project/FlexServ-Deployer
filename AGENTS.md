# Deployment scripts

- Keep `.sh` files as Bash orchestration using command-line utilities. Do not embed
  Python or invoke Python helper scripts for configuration, JSON, probing or cleanup.
- Use `jq` for structured JSON and Linux utilities for filesystem and network work.
  Python application executables may be inspected with `--version`; do not use `-c`
  or heredocs to import modules from a shell script.
- Keep scripts simple, quote paths and arguments, and use arrays instead of `eval`.
- `--trust-remote-code` only exports `FLEXSERV_TRUST_REMOTE_CODE`; it does not change
  FCS authorization or backend arguments in this iteration.
- `--enable-https` controls public Gateway TLS only. Never disable or weaken the
  independent job-scoped Gateway/FCS mutual TLS in either public mode.
- Write documentation and comments in English. Preserve unrelated local files.
- Publish binaries in their component repositories. Deployer downloads verified
  releases from tapis-project and assembles them; never reintroduce Rust builds
  or personal-fork component sources into its release workflow.
- Keep the 1.5.0 Tapis App URL at `tapis-flexserv-1.5.0-alpha-44` until the Beta
  transition. Refresh its assets only from a verified Alpha release; retain each
  candidate's immutable archive and record actual commits in BUILD.json.
