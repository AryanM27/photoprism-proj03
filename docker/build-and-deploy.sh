#!/usr/bin/env bash
set -euo pipefail

echo "=== Building Photoprism ==="
cd ~/photoprism
git switch feature/semantic-search
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd):/go/src/github.com/photoprism/photoprism \
  -w /go/src/github.com/photoprism/photoprism \
  photoprism/develop:latest bash -c "make build-go && cd frontend && npm ci && npm run build"

cd ~
echo "=== Writing Dockerfile.semantic ==="
cat > ~/Dockerfile.semantic <<'EOF'
FROM photoprism/photoprism:latest
COPY photoprism/photoprism /opt/photoprism/bin/photoprism
COPY photoprism/assets /opt/photoprism/assets
ENV LD_LIBRARY_PATH=/opt/photoprism/lib
EOF
echo "=== Building photoprism-semantic image ==="
docker build -t photoprism-semantic:latest -f Dockerfile.semantic ~

echo "=== Deploying stack ==="
cd ~/photoprism-proj03
docker compose -f docker/docker-compose.yml --env-file Data/docker/.env up -d

echo "=== Done ==="
