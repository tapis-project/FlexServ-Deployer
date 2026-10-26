# FlexServ 1.5.0 on the TACC tenant

`app.json` defines application **ID** `FlexServ-1.5.0`, version `1.5.0`.
Tapis represents the application name using `id`; there is no additional top-level
`name` field. The definition uses ZIP/BATCH with native executables and a pinned
Alpha 44 bundle, without Apptainer or an embedded inference engine.

This directory creates no Tapis resources automatically. With a token issued for
the TACC tenant, register the definition from this directory:

```bash
curl --fail -sS -X POST https://tacc.tapis.io/v3/apps \
  -H "X-Tapis-Token: $TAPIS_TOKEN" -H 'Content-Type: application/json' \
  --data-binary @app.json
curl --fail -sS \
  -H "X-Tapis-Token: $TAPIS_TOKEN" \
  https://tacc.tapis.io/v3/apps/FlexServ-1.5.0/1.5.0
```

The caller becomes the owner; use a TACC token, not a token from the `public`
tenant. No fixed owner, access token or allocation is embedded in the definition.
See the [Tapis application contract](https://tapis.readthedocs.io/en/latest/technical/apps.html).

## First startup/shutdown check

Choose an execution system you already own/can execute on, one of its logical
queues, and your allocation. Generate a job request without submitting it:

```bash
export TAPIS_EXEC_SYSTEM_ID=your-existing-system-id
export TAPIS_LOGICAL_QUEUE=your-logical-queue
export TAPIS_ALLOCATION=your-allocation
bash prepare-job.sh > job.json
```

The generated request defaults to `--control-plane-only`, so this first check does
not install vLLM/MAX or download models. Set `TAPIS_CONTROL_PLANE_ONLY=0` to prepare
backend environments. The application itself has no forced control-plane-only
argument. Override `PRI_MODEL_REPO` and `PUB_MODEL_REPO` before generating the job
when the App defaults do not match your filesystem.

Submit or inspect this file manually using the TACC Jobs API. Set
`TAPIS_ENABLE_HTTPS=1` to include `--enable-https`. Without that argument the
Gateway exposes HTTP. HTTPS requires TAP certificate/key files; missing TAP
functions still allow HTTP startup using a secure random token and the launcher's
login-port selection. Successful public forwarding requires the site's SSH
`GatewayPorts` policy to permit the requested remote bind.

**Gateway/FCS communication always uses job-scoped mutual TLS**, independent of
the external protocol. A missing/invalid internal identity prevents readiness.
`TAPIS_TRUST_REMOTE_CODE=1` only adds an argument that exports
`FLEXSERV_TRUST_REMOTE_CODE=true`; no backend behavior changes in this release.

The private `flexserv_access_info.txt`, logs and reports remain under the job
working/output directory for archive and inspection. Stop/cancel the test job
through Tapis or Slurm and verify that Gateway, FCS and their owned processes exit.
Actual tenant registration and HPC startup/shutdown remain operator checks.
