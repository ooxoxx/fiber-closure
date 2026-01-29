#!/bin/bash
# YOLOv4-Tiny Training Script for Fiber Closure Detection
# Usage: ./train.sh [gpu_id]

set -e

# Configuration
DARKNET_BIN="./darknet/build/src-cli/darknet"
DATA_FILE="cfg/fiber.data"
CFG_FILE="cfg/fiber-tiny.cfg"
WEIGHTS_FILE="weights/yolov4-tiny.conv.29"
LOG_DIR="logs"
GPU_ID="${1:-0}"

# Create log directory
mkdir -p "$LOG_DIR"

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

# Check if darknet binary exists
if [ ! -f "$DARKNET_BIN" ]; then
    echo "Error: Darknet binary not found at $DARKNET_BIN"
    echo "Please build darknet first or update DARKNET_BIN path"
    exit 1
fi

# Check if pre-trained weights exist
if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "Error: Pre-trained weights not found at $WEIGHTS_FILE"
    echo "Download from: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29"
    exit 1
fi

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found at $DATA_FILE"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CFG_FILE" ]; then
    echo "Error: Config file not found at $CFG_FILE"
    exit 1
fi

echo "Starting YOLOv4-Tiny training..."
echo "  Data:    $DATA_FILE"
echo "  Config:  $CFG_FILE"
echo "  Weights: $WEIGHTS_FILE"
echo "  GPU:     $GPU_ID"
echo "  Log:     $LOG_FILE"
echo ""

# Run training
$DARKNET_BIN detector train \
    "$DATA_FILE" \
    "$CFG_FILE" \
    "$WEIGHTS_FILE" \
    -gpus "$GPU_ID" \
    -map \
    -dont_show \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training completed. Log saved to: $LOG_FILE"
