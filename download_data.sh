#!/usr/bin/env bash
# Fetch the dataset volumes from Zenodo and unpack them into ./data/.
#
# The volumes (~3.9 GB) are NOT stored in git; they are archived on Zenodo and
# downloaded on demand. The rest of the capsule (code + the pretrained corrector
# checkpoints, the latter via Git LFS) lives in the repo.
#
# Usage:  ./download_data.sh
set -euo pipefail

ZENODO_RECORD="21223591"          # https://zenodo.org/records/21223591
TARBALL="data.tar"
URL="https://zenodo.org/records/${ZENODO_RECORD}/files/${TARBALL}?download=1"

cd "$(dirname "$(readlink -f "$0")")"   # repo root (this script lives here)

if [ -d data ] && [ -n "$(ls -A data 2>/dev/null)" ]; then
  echo "data/ already populated -- nothing to do (delete it to re-download)."
  exit 0
fi

echo "Downloading ${TARBALL} from Zenodo record ${ZENODO_RECORD} ..."
curl -L --fail --retry 3 -o "${TARBALL}" "${URL}"

echo "Unpacking ${TARBALL} -> ./data/ ..."
tar xf "${TARBALL}"

echo "Done. Volumes are under ./data/  (safe to 'rm ${TARBALL}' to reclaim space)."
