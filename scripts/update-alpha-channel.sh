#!/usr/bin/env bash
# Refresh only the existing Alpha URL from a verified assembly on the approved branch.
set -euo pipefail
artifacts=${1:?Artifact directory required}
channel=tapis-flexserv-1.5.0-alpha-44
: "${GITHUB_REPOSITORY:?Repository required}" "${GITHUB_SHA:?Source commit required}" "${GITHUB_RUN_ID:?Run ID required}"
[[ ${GITHUB_REF:-} == refs/heads/feature/1.5.0-alpha && "$GITHUB_RUN_ID" =~ ^[0-9]+$ ]] || {
    echo 'Alpha publication requires the approved branch and an Actions run.' >&2; exit 1;
}
for command in gh jq sha256sum unzip; do command -v "$command" >/dev/null; done
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
gh api "repos/$GITHUB_REPOSITORY/releases?per_page=100" --paginate --slurp > "$scratch/releases.json"
if jq -e 'any(.[][]; (.draft|not) and (.tag_name|startswith("tapis-flexserv-1.5.0-beta")))' "$scratch/releases.json" >/dev/null; then
    echo 'Published Beta found; Alpha channel is frozen.'; exit 0
fi
zip="$artifacts/Tapis-FlexServ.zip"
checksum="$artifacts/Tapis-FlexServ.zip.sha256"
zip_digest=$(sha256sum "$zip"); zip_digest=${zip_digest%% *}
checksum_digest=$(sha256sum "$checksum"); checksum_digest=${checksum_digest%% *}
read -r expected filename < "$checksum"
[[ "$expected" == "$zip_digest" && "$filename" == Tapis-FlexServ.zip ]] || { echo 'Local archive checksum mismatch.' >&2; exit 1; }
unzip -p "$zip" BUILD.json > "$scratch/build.json"
jq -e --arg sha "$GITHUB_SHA" --arg repo "$GITHUB_REPOSITORY" '
    .schema_version==1 and .components.deployer.repository==$repo and .components.deployer.commit==$sha and
    .components.gateway.repository=="tapis-project/FlexServ" and
    .components.fcs.repository=="tapis-project/FlexServ-FCS" and
    .components.pack.repository=="tapis-project/FlexServ-FCS-Pack" and
    all(.components|to_entries[]|select(.key!="deployer");
        (.value.release_id|type)=="number" and (.value.commit|test("^[a-f0-9]{40}$")) and (.value.assets|length)>0)
' "$scratch/build.json" >/dev/null || { echo 'Bundle source identity mismatch.' >&2; exit 1; }
assets_match() {
    jq -e --arg zip "sha256:$zip_digest" --arg checksum "sha256:$checksum_digest" \
        '(.assets|length)==2 and
         any(.assets[]; .name=="Tapis-FlexServ.zip" and .digest==$zip) and
         any(.assets[]; .name=="Tapis-FlexServ.zip.sha256" and .digest==$checksum)' "$1" >/dev/null
}
gh api "repos/$GITHUB_REPOSITORY/releases/tags/$channel" > "$scratch/channel.json"
jq -e --arg tag "$channel" '.tag_name==$tag and (.draft|not) and (.immutable!=true) and
    ([.assets[].name]|sort)==["Tapis-FlexServ.zip","Tapis-FlexServ.zip.sha256"]' "$scratch/channel.json" >/dev/null || {
    echo 'Alpha channel is immutable or has an unexpected asset set.' >&2; exit 1;
}
if assets_match "$scratch/channel.json"; then echo 'Alpha channel already matches this build.'; exit 0; fi
# A queued older run must not replace a newer branch assembly.
current=$(gh api "repos/$GITHUB_REPOSITORY/commits/feature%2F1.5.0-alpha" --jq .sha)
[[ "$current" == "$GITHUB_SHA" ]] || { echo 'Assembly superseded by a newer branch commit; channel unchanged.'; exit 0; }
# The run-specific Actions artifact is retained before this non-atomic asset replacement.
gh release upload "$channel" "$zip" "$checksum" --repo "$GITHUB_REPOSITORY" --clobber
release_id=$(jq -er '.id|numbers' "$scratch/channel.json")
body="Rolling FlexServ 1.5 Alpha channel for the Tapis App. Current assembly: [$GITHUB_SHA](https://github.com/$GITHUB_REPOSITORY/commit/$GITHUB_SHA), [Actions run $GITHUB_RUN_ID](https://github.com/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID). Use BUILD.json and SHA-256 for exact component identity. The Alpha 44 tag remains at its original commit; no new Deployer tag is created."
gh api --method PATCH "repos/$GITHUB_REPOSITORY/releases/$release_id" \
    -f name='FlexServ 1.5 Alpha channel (fixed Tapis URL)' -f body="$body" -F prerelease=true > "$scratch/updated.json"
assets_match "$scratch/updated.json" || { echo 'Alpha channel verification failed; retry the same verified assembly.' >&2; exit 1; }
echo "Alpha channel now serves assembly $GITHUB_SHA."
