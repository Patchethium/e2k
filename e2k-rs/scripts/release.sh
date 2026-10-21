#!/usr/bin/env bash
# Builds the GitHub release archive: the crate source at a git ref plus the
# model files, so downstream crates can use it as a `path` dependency with
# the `embed` feature.
#
# Usage: scripts/release.sh [git ref, default HEAD]
# Models are taken from $E2K_MODEL_DIR, default ../src/e2k/models.
set -euo pipefail

crate_dir="$(cd "$(dirname "$0")/.." && pwd)"
repo_dir="$(git -C "$crate_dir" rev-parse --show-toplevel)"
ref="${1:-HEAD}"
model_dir="${E2K_MODEL_DIR:-$repo_dir/src/e2k/models}"
models=(model-c2k.npz model-p2k.npz accent.npz ngram.json.zip)

rel_path="$(realpath --relative-to="$repo_dir" "$crate_dir")"
version="$(git -C "$repo_dir" show "$ref:$rel_path/Cargo.toml" |
    sed -n 's/^version = "\(.*\)"/\1/p' | head -n1)"

if [ "$ref" = HEAD ] && [ -n "$(git -C "$repo_dir" status --porcelain -- "$rel_path")" ]; then
    echo "warning: $rel_path has uncommitted changes, they are NOT in the archive" >&2
fi
for f in "${models[@]}"; do
    [ -f "$model_dir/$f" ] || { echo "error: missing $model_dir/$f" >&2; exit 1; }
done

out_dir="$crate_dir/target/release-archive"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$out_dir" "$stage/e2k-rs"

git -C "$repo_dir" archive "$ref:$rel_path" | tar -x -C "$stage/e2k-rs"
mkdir -p "$stage/e2k-rs/models"
cp "${models[@]/#/$model_dir/}" "$stage/e2k-rs/models/"

archive="$out_dir/e2k-rs-$version.zip"
rm -f "$archive"
(cd "$stage" && zip -qr -X "$archive" e2k-rs)
echo "$archive"
sha256sum "$archive"
