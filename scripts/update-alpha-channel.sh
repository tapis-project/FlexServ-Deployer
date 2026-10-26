#!/usr/bin/env bash
# Refresh a stable Tapis download URL only from an already verified Alpha release.
set -euo pipefail
candidate=${1:?Published candidate tag required}
artifacts=${2:?Artifact directory required}
channel=tapis-flexserv-1.5.0-alpha-44
case "$candidate" in tapis-flexserv-1.5.0-alpha-*) ;; *) echo 'Not an Alpha candidate; channel unchanged.'; exit 0 ;; esac
[[ "$candidate" != "$channel" ]] || { echo 'Initial channel release; no promotion required.'; exit 0; }
: "${GITHUB_REPOSITORY:?Repository required}"
for command in gh jq sha256sum; do command -v "$command" >/dev/null || exit 1; done
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
assets_match() {
    jq -e --arg zip "sha256:$zip_digest" --arg checksum "sha256:$checksum_digest" \
        '(.assets|length)==2 and
         any(.assets[]; .name=="Tapis-FlexServ.zip" and .digest==$zip) and
         any(.assets[]; .name=="Tapis-FlexServ.zip.sha256" and .digest==$checksum)' "$1" >/dev/null
}
gh api "repos/$GITHUB_REPOSITORY/releases/tags/$candidate" > "$scratch/source.json"
jq -e --arg tag "$candidate" '.tag_name==$tag and (.draft|not)' "$scratch/source.json" >/dev/null
assets_match "$scratch/source.json" || { echo 'Published candidate assets differ from the verified build.' >&2; exit 1; }
gh api "repos/$GITHUB_REPOSITORY/releases/tags/$channel" > "$scratch/channel.json"
jq -e --arg tag "$channel" '.tag_name==$tag and (.draft|not) and (.immutable!=true) and
    ([.assets[].name]|sort)==["Tapis-FlexServ.zip","Tapis-FlexServ.zip.sha256"]' "$scratch/channel.json" >/dev/null || {
    echo 'Alpha channel is immutable or has an unexpected asset set.' >&2; exit 1;
}
if assets_match "$scratch/channel.json"; then echo 'Alpha channel already matches this build.'; exit 0; fi
# Versioned source assets remain untouched. GitHub asset replacement is not atomic.
gh release upload "$channel" "$zip" "$checksum" --repo "$GITHUB_REPOSITORY" --clobber
release_id=$(jq -er '.id|numbers' "$scratch/channel.json")
body="Rolling FlexServ 1.5 Alpha channel for the Tapis App. Current build: [$candidate](https://github.com/$GITHUB_REPOSITORY/releases/tag/$candidate). Use BUILD.json and SHA-256 for exact source identity; this channel tag remains at its original commit."
gh api --method PATCH "repos/$GITHUB_REPOSITORY/releases/$release_id" \
    -f name='FlexServ 1.5 Alpha channel (fixed Tapis URL)' -f body="$body" -F prerelease=true > "$scratch/updated.json"
assets_match "$scratch/updated.json" || { echo 'Alpha channel verification failed; retry promotion from the versioned release.' >&2; exit 1; }
echo "Alpha channel now serves $candidate."
