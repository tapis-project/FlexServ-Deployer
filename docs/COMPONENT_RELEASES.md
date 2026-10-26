# Independent component releases

Gateway binaries are released by `tapis-project/FlexServ` from `feature/1.5.0`.
Capability Server binaries are released by `tapis-project/FlexServ-FCS` from
`feature/new_fcs`. Both releases contain Linux x86_64 and aarch64 binaries,
source/version/toolchain/ABI sidecars, and `release-assets.sha256` covering every
asset other than the inventory itself.

`tapis-project/FlexServ-FCS-Pack`, branch `feature/new_fcs`, releases
`tapis-workflow-pack.zip` and `vllm-pack.zip`. They are generated from one reviewed
catalog, each with its own complete Pack checksum and required shared helpers.
Their release includes source metadata and a release asset inventory.

The Deployer's `feature/1.5.0-alpha` workflow assembles published assets; it never
checks out component source, installs Rust, invokes component build workflows or
compiles binaries. Supply component release tags for reproducibility, or leave
them empty to select the newest published release whose commit belongs to the
approved component branch. Selection is resolved once to a release ID and commit.
Drafts and the mutable Deployer Alpha channel are not component inputs.

Every downloaded file must match both the release inventory and GitHub's asset
digest. Binary source sidecars must match the selected tag commit, and the two
architectures must report the same version. The two Pack payloads are merged at
build time: identical shared files are deduplicated, conflicting files or toolset
identities fail assembly. Fresh manifest and checksums are generated for the
merged runtime catalog. No downloaded hooks or scripts are executed.

The final ZIP keeps the established runtime layout and Slurm scripts. BUILD.json
records repository, tag, commit, release ID and asset hashes for each component,
plus the Deployer revision and merged Pack digest. Internal Gateway/FCS mutual
TLS and the public HTTP/HTTPS switch remain independent and unchanged.

Component releases require an existing tag at the selected source commit. Push
the tag to the appropriate tapis-project repository, or select it in that
repository's manual workflow. This avoids granting Actions workflow-write access
merely to create tags. Pack and script changes need no Gateway/FCS rebuild.

The existing `FLEXSERV_SOURCE_TOKEN` is used only to read component releases.
The Deployer's own `GITHUB_TOKEN` publishes its final ZIP and refreshes the fixed
`tapis-flexserv-1.5.0-alpha-44` URL until the Beta transition. Exact component tags
remain independently selectable after later releases appear.
