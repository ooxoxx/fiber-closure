# 4K 无人机巡检图像 YOLOv4 训练与推理最佳实践手册 (v2.0 优化版)

## 1. 核心设计理念 (Core Philosophy)

* **训练端 (Training):** 使用切片 (Slicing) 保证小目标分辨率，使用 H100 强算力加速，关闭姿态增强保证细粒度分类准确性。
* **推理端 (Inference):** 放弃“切断-拼接”的复杂逻辑。通过**提高切片重叠率**，确保目标至少在某一个切片中是完整的，然后通过**边缘过滤**和**包含抑制**，让“完整框”直接淘汰“残缺框”。

---

## 2. 环境构建 (Infrastructure)

省略
```



---

## 3. 数据工程 (Data Engineering)

### 3.1 训练预处理：切图 (Slicing)

严禁直接使用 4K 原图训练。

* **切片尺寸：** `1024 x 1024` (推荐)。
* **训练重叠率：** `20%`。
* **标签处理：** 全局坐标 -> 局部坐标。若目标被切断，仅当保留面积 > 60% 时保留标签，否则视为背景。
* **Anchor 重聚类：** 针对细长光缆接头盒重新聚类 (`width=1024, height=1024`)。

---

## 4. 模型配置 (`.cfg` Optimization)

基于 `yolov4-custom.cfg` 修改，针对姿态分类优化。

### 4.1 关键参数

```ini
[net]
batch=64               # 保持64，防止 Loss=NaN
subdivisions=16        # H100 可设 8
width=1024             # 切片尺寸
height=1024            # 切片尺寸
angle=0                # 【关键】设为0。姿态(倾斜/垂直)是分类依据，禁止旋转增强
flip=0                 # 【关键】设为0。禁止上下翻转，防止“掉落”与“正常”混淆
max_batches=10000      # 类别数 * 2000

```

### 4.2 检测头优化

* `filters = (5 + 5) * 3 = 30`
* `scale_x_y = 1.05` (提升网格边缘检出率)

---

## 5. 推理策略 (Inference Pipeline) - 【重点更新】

采用 **SAHI (Slicing Aided Hyper Inference)** 思想，结合**高重叠 + 边缘抑制**策略。

### 5.1 推理流程图

1. **输入：** 4K 原始图像。
2. **高重叠切片：** Overlap 提升至 **30%**。
3. **批量检测：** 仅保留置信度较高的框。
4. **坐标还原：** 映射回 4K 全局坐标系。
5. **边缘过滤 (Edge Filtering)：** 标记并降权/删除紧贴切片边缘的残缺框。
6. **包含抑制 (Containment Suppression)：** 利用完整框“吞噬”残缺框。

### 5.2 关键工程技巧详解

#### A. 增加推理重叠率 (High Inference Overlap)

* **设置：** 建议设为 **25% - 30%** (例如 1024的图，步长为 716)。
* **原理：** 当 Overlap 足够大（超过目标最大尺寸）时，任何位置的目标都极大概率会**完整地**出现在至少一个切片的中心区域。
* **效果：** 全局图上会同时出现一个“完整框”（来自中心切片）和一个或多个“边缘残缺框”（来自相邻切片）。我们只需要保留前者。

#### B. 边缘过滤 (Edge Filtering)

* **逻辑：** 检测框如果距离切片边缘小于阈值（如 2px），且该框的尺寸明显小于该类别的平均尺寸（说明被切断了），将其标记为“低质量框”。
* **操作：**
* 如果该区域只有这一个框 -> 保留（可能是真的在图像边缘）。
* 如果该区域有另一个重叠的“非边缘框” -> 直接丢弃这个边缘框。



#### C. 吞噬策略 (Containment Suppression)

* 不再尝试拼接两个残缺框，而是用完整框去抑制残缺框。
* **IoM (Intersection over Minimum) 判断：**
如果 `Intersection(A, B) / Min(Area_A, Area_B) > 0.7`，说明其中一个小框几乎完全被大框包含。
* 此时保留 **面积更大** 且 **不贴边** 的那个框。



### 5.3 推荐 Python 推理逻辑 (伪代码)

```python
def inference_optimized(full_image, net, slice_size=1024, overlap=0.30):
    """
    v2.0 优化版推理逻辑：高重叠 + 边缘过滤 + 吞噬合并
    """
    H, W = full_image.shape[:2]
    step = int(slice_size * (1 - overlap))
    
    all_detections = [] # [x1, y1, x2, y2, score, class, is_edge_box]

    # 1. 切片推理
    for y in range(0, H, step):
        for x in range(0, W, step):
            # ... (切图、推理代码同前) ...
            # ... (坐标还原为 global_x, global_y) ...
            
            # === 边缘过滤逻辑 ===
            # 判断框是否紧贴当前切片的边缘 (距离 < 2px)
            is_touching_edge = (
                local_x1 < 2 or local_y1 < 2 or 
                local_x2 > slice_size-2 or local_y2 > slice_size-2
            )
            # 如果贴边，且长宽比异常或面积过小，标记为边缘残次品
            is_edge_box = is_touching_edge # 简单起见，贴边即标记
            
            all_detections.append([gx1, gy1, gx2, gy2, score, cls, is_edge_box])

    # 2. 结果融合 (Containment Suppression)
    final_boxes = []
    # 按分数从高到低排序，优先保留高质量框
    all_detections.sort(key=lambda x: x[4], reverse=True)
    
    while len(all_detections) > 0:
        best_box = all_detections.pop(0) # 取出分数最高的
        final_boxes.append(best_box)
        
        # 将 best_box 与剩余所有框比对
        keep_indices = []
        for i, other_box in enumerate(all_detections):
            if other_box[5] != best_box[5]: # 不同类，保留
                keep_indices.append(i)
                continue
                
            # 计算 IoM (小框被大框包含的比例)
            inter_area = calculate_intersection(best_box, other_box)
            min_area = min(area(best_box), area(other_box))
            io_min = inter_area / (min_area + 1e-6)
            
            # 策略：如果重叠包含度高 (>0.7)
            if io_min > 0.7:
                # 这是一个重复或残缺的框，需要被 best_box "吞噬"
                # 但需要再次确认：如果 best_box 是边缘框，而 other_box 是完整框
                # (虽然 best_box 分数高，但可能是因为背景简单导致分数虚高)
                if best_box[6] == True and other_box[6] == False:
                    # 这是一个特殊情况：我们刚选出的 best_box 其实是个边缘残次品
                    # 此时应该 交换：保留 other_box，丢弃 best_box
                    final_boxes.pop() # 移除刚才加进去的
                    final_boxes.append(other_box) # 加这个完整的
                    # best_box 被丢弃，other_box 被选中，循环继续
                pass # 无论如何，other_box 在这里都被处理了(被吞噬或被交换)，不加入 keep_indices
            else:
                keep_indices.append(i)
        
        all_detections = [all_detections[i] for i in keep_indices]

    return final_boxes

```
