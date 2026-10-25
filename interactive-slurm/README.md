# Interactive Slurm launch

Download and extract the published `Tapis-FlexServ.zip` onto shared storage.
Inside an existing allocation, set `FLEXSERV_BUNDLE_DIR` to that directory:

```bash
export FLEXSERV_BUNDLE_DIR=/shared/flexserv/bundle
bash interactive-slurm/launch.sh --control-plane-only
```

The wrapper delegates to the same `app/run_flexserv.sh` used by Tapis. It does
not compile code, use a Bootstrap binary, or maintain separate policies. Drop
`--control-plane-only` to prepare the configured inference environments. Set an
owned/private `FLEXSERV_VENV_ROOT` if the shared site default is unsuitable.

See [the Alpha deployment guide](../docs/ALPHA_DEPLOYMENT.md) for site variables,
TAP/OOD access, environment preparation, TLS cleanup and credential handling.
The ignored `site.env` is never overwritten. Retired BOOTSTRAP_BIN/FCS_BIN and
manual policy variables are not used by this wrapper.

For a manual laptop tunnel, the standalone helper remains available:

```bash
bash interactive-slurm/tunnel.sh USER@LOGIN_HOST COMPUTE_NODE 8000 8000
```

The compute listener must be reachable from the login node; for a loopback-only
listener use an SSH jump to the compute node or the bundle's site tunnel. Keep
the main launcher foreground; Ctrl-C stops its owned processes. No restart
recovery, model download or external Tapis submission is performed.
