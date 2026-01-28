# Deployment Guide for H100 Server

本指南说明如何在离线 H100 服务器上部署光缆接头盒检测系统。

## Prerequisites

- Ubuntu 22.04+ 或 RHEL 8+
- NVIDIA H100 GPU with CUDA 12.x + cuDNN 8.x+
- Python 3.11+
- CMake 3.18+
- GCC 11+ 或 Clang 14+

## 1. Darknet Installation

Darknet 必须在目标服务器上编译以确保 CUDA 兼容性。

```bash
# Clone darknet repository
git clone https://github.com/hank-ai/darknet.git
cd darknet

# Build with CUDA support
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Verify installation
./darknet version
```

如果服务器无法访问外网，需提前下载 darknet 源码并传输到服务器。

## 2. Python Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Or use requirements.txt for training only
pip install -r requirements.txt
```

## 3. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with correct paths
# DARKNET_LIB_PATH: path to libdarknet.so
# DARKNET_CFG_PATH: path to model .cfg file
# DARKNET_WEIGHTS_PATH: path to model .weights file
# DARKNET_DATA_PATH: path to .data file
```

Example `.env`:
```ini
DARKNET_LIB_PATH=/path/to/darknet/build/libdarknet.so
DARKNET_CFG_PATH=./cfg/yolov4-closure.cfg
DARKNET_WEIGHTS_PATH=./weights/yolov4-closure_best.weights
DARKNET_DATA_PATH=./cfg/closure.data
```

## 4. Inference Service

启动 FastAPI 推理服务：

```bash
source venv/bin/activate
uvicorn inference.main:app --host 127.0.0.1 --port 8000
```

生产环境建议使用 gunicorn：

```bash
gunicorn inference.main:app -w 1 -k uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8000
```

API 端点：
- `POST /detect` - 检测图片中的目标
- `GET /health` - 健康检查

## 5. Training

### 5.1 Data Preparation

将原始 4K 图片和标注放入 `data/raw/` 目录：

```bash
./prepare.sh --raw-images data/raw/images --raw-labels data/raw/labels
```

这会将图片切片为 1024x1024 并转换标注坐标。

### 5.2 Start Training

```bash
./train.sh
```

训练参数已针对 H100 优化（`subdivisions=8`）。

### 5.3 Monitor Training

训练日志保存在 `logs/` 目录，loss 曲线图自动生成为 `chart.png`。

## 6. Weights Transfer

模型权重文件（`.weights`）不包含在部署包中，需单独传输：

```bash
# On source machine
scp weights/yolov4-closure_best.weights user@h100-server:/path/to/weights/

# Verify file integrity
md5sum weights/yolov4-closure_best.weights
```

## Troubleshooting

### CUDA Version Mismatch
确保 darknet 编译时使用的 CUDA 版本与运行时一致。

### libdarknet.so Not Found
检查 `DARKNET_LIB_PATH` 环境变量是否正确设置。

### Out of Memory
减小 `batch` 或增大 `subdivisions` 值。
