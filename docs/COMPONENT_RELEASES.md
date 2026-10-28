# Independent component releases

Gateway binaries are released by `tapis-project/FlexServ` from `feature/1.5.0`.
Capability Server binaries are released by `tapis-project/FlexServ-FCS` from
`feature/new_fcs`. Both releases contain Linux x86_64 and aarch64 binaries,
source/version/toolchain/ABI sidecars, and `release-assets.sha256` covering every
asset other than the inventory itself.

`tapis-project/FlexServ-FCS-Pack`, branch `feature/new_fcs`, releases
`tapis-workflow-pack.zip` and `vllm-pack.zip`. They are generated from one reviewed
catalog, each with its own complete Pack checksum and required helpers.
Their release includes source metadata and a release asset inventory.

The Deployer's `feature/1.5.0-alpha` workflow assembles published assets; it never
checks out component source, installs Rust, invokes component build workflows or
compiles binaries. Manual dispatch requires an explicit Pack release tag. The
selected tag is stored in `pack-tag.txt`; no Pack ZIP is embedded. Gateway and
FCS tags remain optional selections. Branch pushes run checks only.

For the first multi-ZIP release, select `gateway-1.5.0-alpha-2` and
`fcs-1.5.0-alpha-2` explicitly. The earlier Gateway alpha-1 release does not
enforce the new protected per-Pack ZIP arguments. The assembler verifies asset
integrity and the FCS registry, but does not yet prove Gateway/FCS protocol
compatibility for every arbitrary combination of release tags. Check any
different pair against its release and integration evidence before publishing.

Every CI download must match both the release inventory and GitHub's asset
digest. Binary source sidecars must match the selected tag commit, and the two
architectures must report the same version. CI validates both standalone Packs
and their combined registry using the selected FCS; their files are not merged
or included in the published bundle. No downloaded hooks or scripts are executed.

On the head node, the FCS prepares the selected public release ZIPs and verifies
their sidecars and complete inventories. Its protected per-job result records
their paths, ZIP digests and inventory pins; all ranks load the same ordered
selection. A semicolon-delimited `PACK_TO_LOAD` may also select local ZIP paths.
Sidecars fetched from the same release provide integrity, not independent
publisher authentication. Existing self-contained single-Pack bundles remain
supported. BUILD.json records the selected release and source metadata. Internal
Gateway/FCS mutual TLS and public HTTP/HTTPS remain independently configured.

Gateway/FCS branch pushes run lightweight publication fixtures. Native binaries
are built for their component release tags or explicit manual dispatch. Pack
branch checks package data only. Deployer branch pushes run checks only. Manual
assembly supplies a Pack tag and can refresh Alpha 44 after all checks pass. No
new Deployer tag or release is created.

Component releases require an existing tag at their selected source commit. Push
the tag to the appropriate tapis-project component repository, or select it in
that repository's manual workflow. Pack and script changes need no Gateway/FCS
rebuild.

The existing `FLEXSERV_SOURCE_TOKEN` is used only to read component releases.
The Deployer's own `GITHUB_TOKEN` refreshes
`tapis-flexserv-1.5.0-alpha-44` until the Beta transition. The original Alpha 44
Git tag remains unchanged. Each successful assembly retains a run/attempt-specific
Actions artifact for 90 days; BUILD.json and the archive checksum identify its
exact source and component assets. The release notes link to the current assembly
commit and run. GitHub replaces ZIP and checksum separately; verify and retry a
download during updates. A published 1.5.0 Beta freezes automatic Alpha promotion.
