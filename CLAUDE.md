# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fiber optic closure (光缆接头盒) and cable rack (光缆余缆架) detection system using YOLOv4/Darknet framework. Designed for 4K drone inspection images with two main deliverables:
1. Training pipeline code
2. Inference service (Docker-packaged)

## Architecture

```
fiber-closure/
├── darknet/          # Symlink to hank-ai/darknet framework
├── docs/             # Technical documentation
│   └── train_inference.md  # Training & inference best practices
├── cfg/              # Model configs (.cfg), data files (.data, .names)
├── data/             # Training/validation datasets (to be created)
├── weights/          # Model weights (to be created)
└── inference/        # Inference service code (to be created)
```

## Darknet Framework (darknet/)

- **src-lib/**: Core C++ library with CUDA kernels
- **src-cli/**: CLI tool (`darknet_cli.cpp`)
- **src-python/**: Python bindings (`darknet.py`, `darknet_images.py`, `darknet_video.py`)
- **src-onnx/**: ONNX export utilities
- **cfg/**: Pre-built YOLO configurations


## Critical Training Constraints

These settings are **mandatory** for this project - posture/orientation is a classification feature:

```ini
angle=0    # NO rotation augmentation
flip=1     # NO vertical flip
```

Violating these will cause model confusion between "fallen" and "normal" states.

## Training Pipeline

Default model: **YOLOv4-Tiny** (faster training and inference, suitable for edge deployment)

1. **Slice 4K images**: 928×928 tiles, 20% overlap
2. **Label handling**: Convert global→local coords; discard if <60% area retained
3. **Anchor clustering**: Re-cluster for thin elongated targets
4. **Config**: `batch=64`, `subdivisions=16` (8 for H100), `width=height=928`
5. **Max batches**: `num_classes × 2000`

## Inference Pipeline

1. **High-overlap slicing**: 30% overlap (step=650 for 928 tiles)
2. **Edge filtering**: Mark boxes within 2px of slice edges as low-quality
3. **Containment suppression**: IoM > 0.7 → larger non-edge box wins
4. **Coordinate restoration**: Map back to 4K global coordinates

Key principle: High overlap ensures every target appears complete in at least one tile's center region.

## Detection Head Formula

For 5 classes: `filters = (5 + num_classes) × 3 = 30`

## Network Note

Use `127.0.0.1` instead of `localhost` for all network operations to avoid DNS resolution issues on this system.
