# Fiber Closure Detection Inference Service

基于 FastAPI 的光缆接头盒/余缆架检测推理服务，支持 4K 无人机巡检图像。

## 功能特点

- SAHI 风格的高重叠切片策略，确保目标完整检测
- 边缘检测框标记与抑制
- 全局 NMS 去重
- 支持环境变量配置

## 快速启动

```bash
# 激活虚拟环境
source .venv/bin/activate

# 启动服务
uvicorn inference.main:app --host 127.0.0.1 --port 8000
```

## 环境变量配置

所有配置项均以 `FIBER_` 为前缀：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `FIBER_MODEL_CFG` | `cfg/fiber.cfg` | 模型配置文件路径 |
| `FIBER_MODEL_WEIGHTS` | `weights/fiber_best.weights` | 模型权重文件路径 |
| `FIBER_MODEL_DATA` | `cfg/fiber.data` | 数据配置文件路径 |
| `FIBER_CONFIDENCE_THRESHOLD` | `0.5` | 检测置信度阈值 |
| `FIBER_NMS_THRESHOLD` | `0.45` | NMS 阈值 |
| `FIBER_TILE_SIZE` | `1024` | 切片尺寸 |
| `FIBER_TILE_OVERLAP` | `0.3` | 切片重叠率 (30%) |
| `FIBER_HOST` | `127.0.0.1` | 服务监听地址 |
| `FIBER_PORT` | `8000` | 服务监听端口 |

## API 接口

### 健康检查

```
GET /health
```

响应示例：
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "model_loaded": true,
  "network_size": {"width": 1024, "height": 1024}
}
```

### 目标检测

```
POST /detect
Content-Type: multipart/form-data
```

参数：
- `file`: 图像文件 (必需)
- `threshold`: 置信度阈值，0-1 (可选，默认 0.5)

响应示例：
```json
{
  "detections": [
    {
      "class_name": "closure",
      "confidence": 85.5,
      "bbox": {"x": 1024.5, "y": 512.3, "w": 120, "h": 80},
      "is_edge": false
    }
  ],
  "image_size": {"width": 3840, "height": 2160},
  "inference_time_ms": 1234.5,
  "tile_count": 12
}
```

## 使用 curl 测试

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -F "file=@test_image.jpg" \
  -F "threshold=0.5"
```

## API 文档

启动服务后访问：
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
