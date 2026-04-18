#!/usr/bin/env bash
# Clones AryanM27/photoprism fork into docker/photoprism-src/ and patches
# the yt-dlp install step, which downloads from GitHub releases — a URL that
# times out on Chameleon CHI@TACC nodes during docker build.
#
# Run once before first build, and again to pick up upstream changes:
#   bash docker/scripts/clone-photoprism.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/../photoprism-src"
REPO="https://github.com/AryanM27/photoprism.git"

if [ -d "$DEST/.git" ]; then
  echo "Updating existing clone at $DEST …"
  git -C "$DEST" pull
else
  echo "Cloning $REPO → $DEST …"
  git clone "$REPO" "$DEST"
fi

# No patching needed — the production Dockerfile (docker/photoprism/noble/Dockerfile)
# does not install yt-dlp, so the build works in network-restricted environments.

echo "Done — run: docker compose -f docker/docker-compose.yml build photoprism"
