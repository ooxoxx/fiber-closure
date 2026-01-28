#!/bin/bash
#
# Data Preparation Script for Fiber Closure Detection
# Slices 4K images and generates train/valid file lists.
#
# Usage: ./prepare.sh [options]
#

set -e

# Default configuration (based on CLAUDE.md specifications)
RAW_IMAGES="data/raw/images"
RAW_LABELS="data/raw/labels"
SLICED_DIR="data/sliced"
OUTPUT_DIR="data"
TILE_SIZE=928
OVERLAP=0.2
MIN_AREA=0.6
TRAIN_RATIO=0.9
NEGATIVE_RATIO=0.5
SEED=42
MIN_VALID_CLASS_COUNT=0
NUM_CLASSES=5

# Project root directory for locating Python scripts
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat << EOF
Usage: $(basename "$0") [options]

Data preparation script for fiber closure detection training.
Slices 4K images into tiles and generates train/valid file lists.

Options:
    --raw-images DIR     Directory containing raw 4K images (default: $RAW_IMAGES)
    --raw-labels DIR     Directory containing YOLO labels (default: $RAW_LABELS)
    --sliced-dir DIR     Output directory for sliced tiles (default: $SLICED_DIR)
    --output-dir DIR     Output directory for train.txt/valid.txt (default: $OUTPUT_DIR)
    --tile-size SIZE     Tile size in pixels (default: $TILE_SIZE)
    --overlap RATIO      Overlap ratio between tiles (default: $OVERLAP)
    --min-area RATIO     Minimum area ratio to keep labels (default: $MIN_AREA)
    --train-ratio RATIO  Training set ratio (default: $TRAIN_RATIO)
    --negative-ratio R   Negative sample ratio (default: $NEGATIVE_RATIO)
    --seed NUM           Random seed (default: $SEED)
    --min-valid-class N  Minimum instances per class in validation set (default: $MIN_VALID_CLASS_COUNT)
    --num-classes N      Number of classes in dataset (default: $NUM_CLASSES)
    -h, --help           Show this help message

Example:
    ./prepare.sh
    ./prepare.sh --tile-size 512 --overlap 0.3
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --raw-images)
            RAW_IMAGES="$2"
            shift 2
            ;;
        --raw-labels)
            RAW_LABELS="$2"
            shift 2
            ;;
        --sliced-dir)
            SLICED_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --tile-size)
            TILE_SIZE="$2"
            shift 2
            ;;
        --overlap)
            OVERLAP="$2"
            shift 2
            ;;
        --min-area)
            MIN_AREA="$2"
            shift 2
            ;;
        --train-ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --negative-ratio)
            NEGATIVE_RATIO="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --min-valid-class)
            MIN_VALID_CLASS_COUNT="$2"
            shift 2
            ;;
        --num-classes)
            NUM_CLASSES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

echo "============================================"
echo "  Fiber Closure Data Preparation Pipeline"
echo "============================================"
echo ""

# Step 1: Validate input directories
echo "[Step 1/3] Validating input directories..."

if [[ ! -d "$RAW_IMAGES" ]]; then
    echo "Error: Raw images directory not found: $RAW_IMAGES"
    exit 1
fi

if [[ ! -d "$RAW_LABELS" ]]; then
    echo "Error: Raw labels directory not found: $RAW_LABELS"
    exit 1
fi

# Check for images
IMAGE_COUNT=$(find "$RAW_IMAGES" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.tif" \) 2>/dev/null | wc -l | tr -d ' ')

if [[ "$IMAGE_COUNT" -eq 0 ]]; then
    echo "Error: No images found in $RAW_IMAGES"
    exit 1
fi

echo "  Found $IMAGE_COUNT images in $RAW_IMAGES"
echo "  Labels directory: $RAW_LABELS"
echo ""

# Step 2: Slice images
echo "[Step 2/3] Slicing images into ${TILE_SIZE}x${TILE_SIZE} tiles..."
echo "  Overlap: $(echo "$OVERLAP * 100" | bc)%"
echo "  Min area ratio: $(echo "$MIN_AREA * 100" | bc)%"
echo ""

python3 "$PROJECT_ROOT/scripts/slice_dataset.py" \
    --input-images "$RAW_IMAGES" \
    --input-labels "$RAW_LABELS" \
    --output-dir "$SLICED_DIR" \
    --tile-size "$TILE_SIZE" \
    --overlap "$OVERLAP" \
    --min-area "$MIN_AREA"

echo ""

# Step 3: Generate train/valid splits
echo "[Step 3/3] Generating train.txt and valid.txt..."
echo "  Train ratio: $(echo "$TRAIN_RATIO * 100" | bc)%"
echo "  Negative ratio: $(echo "$NEGATIVE_RATIO * 100" | bc)%"
echo "  Random seed: $SEED"
echo ""

python3 "$PROJECT_ROOT/scripts/prepare_training.py" \
    --sliced-dir "$SLICED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --train-ratio "$TRAIN_RATIO" \
    --negative-ratio "$NEGATIVE_RATIO" \
    --seed "$SEED" \
    --min-valid-class-count "$MIN_VALID_CLASS_COUNT" \
    --num-classes "$NUM_CLASSES"

echo ""
echo "============================================"
echo "  Data Preparation Complete!"
echo "============================================"
echo ""
echo "Output files:"
echo "  Sliced tiles: $SLICED_DIR/"
echo "  Training list: $OUTPUT_DIR/train.txt"
echo "  Validation list: $OUTPUT_DIR/valid.txt"
echo ""
echo "Next steps:"
echo "  1. Review the generated files"
echo "  2. Update cfg/fiber.data with correct paths"
echo "  3. Start training with: ./darknet detector train cfg/fiber.data cfg/fiber.cfg"
echo ""
