#!/bin/bash
set -e

VERSION="${1:-0.1.0}"
IMAGE_NAME="fiber-inference"
OUTPUT_FILE="${IMAGE_NAME}-v${VERSION}.tar.gz"

echo "Building Docker image..."
docker compose -f docker/docker-compose.yml build

echo "Exporting image to ${OUTPUT_FILE}..."
docker save "${IMAGE_NAME}:latest" | gzip > "${OUTPUT_FILE}"

echo "Image exported successfully: ${OUTPUT_FILE}"
echo "Size: $(du -h "${OUTPUT_FILE}" | cut -f1)"

cat << 'EOF'

=== Offline Deployment Instructions ===

1. Transfer the image to the target server:
   scp fiber-inference-v*.tar.gz user@server:/path/

2. Load the image on the target server:
   docker load -i fiber-inference-v*.tar.gz

3. Run the container:
   docker run --gpus all -p 8000:8000 \
     -v /path/to/weights:/app/weights:ro \
     -v /path/to/cfg:/app/cfg:ro \
     -v /path/to/libdarknet.so:/app/lib/libdarknet.so:ro \
     fiber-inference:latest

4. Test the service:
   curl http://127.0.0.1:8000/health

EOF
