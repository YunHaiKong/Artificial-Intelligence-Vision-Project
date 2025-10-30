# 可口可乐Logo定位系统

## 项目概述

本项目是一个基于计算机视觉的可口可乐Logo定位系统，使用OpenCV和NumPy库实现。系统通过ORB特征检测和FLANN匹配算法，在视频中实时检测并定位可口可乐Logo的位置。

## 核心功能

- **实时视频处理**：支持从视频文件中读取帧并进行实时处理
- **ORB特征检测**：使用ORB算法检测Logo和视频帧中的关键点
- **FLANN特征匹配**：基于FLANN算法进行快速特征匹配
- **Lowe比率测试**：应用Lowe比率测试筛选优质匹配点
- **单应性变换**：通过RANSAC算法计算单应性矩阵进行Logo定位
- **可视化显示**：实时显示匹配结果和定位框

## 技术架构

### 核心技术
- **OpenCV**：计算机视觉处理库
- **NumPy**：数值计算库
- **ORB特征检测**：Oriented FAST and Rotated BRIEF特征检测器
- **FLANN匹配器**：Fast Library for Approximate Nearest Neighbors
- **RANSAC算法**：随机抽样一致性算法

### 处理流程
1. **特征提取**：从Logo图像和视频帧中提取ORB特征
2. **特征匹配**：使用FLANN进行KNN匹配
3. **匹配筛选**：应用Lowe比率测试筛选优质匹配
4. **单应性计算**：通过RANSAC计算变换矩阵
5. **Logo定位**：将Logo边界框变换到视频帧中
6. **可视化显示**：绘制匹配结果和定位框

## 文件结构

```
可口可乐logo定位/
├── logo.png              # 可口可乐Logo模板图像
├── video.mp4             # 测试视频文件
├── 可口可乐logo定位.py   # 主程序文件
└── README.md             # 项目说明文档
```

## 环境要求

- Python 3.6+
- OpenCV 4.x
- NumPy

### 安装依赖
```bash
pip install opencv-python numpy
```

## 使用方法

### 命令行运行
```bash
python 可口可乐logo定位.py <视频文件路径> <Logo图片路径>
```

### 示例
```bash
python 可口可乐logo定位.py video.mp4 logo.png
```

## 代码结构详解

### 初始化设置
```python
# ORB特征检测器初始化
detector = cv2.ORB_create()

# FLANN参数配置
FLANN_INDEX_LSH = 6
flann_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
search_params = dict()
matcher = cv2.FlannBasedMatcher(flann_params, search_params)
```

### Logo特征提取
```python
# 加载Logo图像并提取特征
logo = cv2.imread(logo_src, 1)
keypoints_logo, desc_logo = detector.detectAndCompute(logo, None)
```

### 视频处理循环
```python
cap = cv2.VideoCapture(video_src)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 视频帧缩放处理
    frame = cv2.resize(frame, None, fx=0.35, fy=0.35)
    frame_cur = frame.copy()
    
    # 特征检测
    kps_frame, desc_frame = detector.detectAndCompute(frame_cur, None)
```

### 特征匹配与筛选
```python
# KNN匹配
matches = matcher.knnMatch(desc_logo, desc_frame, k=2)

# Lowe比率测试筛选
MIN_COUNT = 10
good_matches_first = []
for i, m in enumerate(matches):
    if len(m) < 2:
        continue
    if m[0].distance < 0.7 * m[1].distance:
        good_matches_first.append(m[0])
```

### Logo定位与可视化
```python
# 计算单应性矩阵
H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)

# Logo边界框变换
quad_prime = cv2.perspectiveTransform(quad.reshape(1, 4, 2), H).reshape(4, 2)
cv2.polylines(frame_cur, [np.int32(quad_prime)], True, (255, 255, 0), 2)
```

## 核心算法

### ORB特征检测
ORB（Oriented FAST and Rotated BRIEF）是一种快速的特征检测和描述符算法，结合了FAST关键点检测和BRIEF描述符的优点，具有旋转不变性和尺度不变性。

### FLANN匹配
FLANN（Fast Library for Approximate Nearest Neighbors）是一种高效的近似最近邻搜索库，特别适用于高维特征向量的快速匹配。

### Lowe比率测试
通过比较最佳匹配和第二佳匹配的距离比率来筛选可靠的匹配点，提高匹配质量。

### RANSAC算法
随机抽样一致性算法用于在存在异常值的情况下稳健地估计数学模型参数，在本项目中用于计算单应性矩阵。

## 项目特色

1. **实时性能**：优化的算法确保在视频流中实时处理
2. **鲁棒性强**：通过多重筛选机制提高定位准确性
3. **可视化完善**：实时显示匹配过程和定位结果
4. **参数可调**：关键参数可根据实际场景调整

## 应用场景

- 品牌Logo检测与追踪
- 视频内容分析
- 广告效果监测
- 智能监控系统
- 增强现实应用

## 性能优化

- 视频帧缩放处理减少计算量
- FLANN快速匹配算法
- 匹配点数量阈值筛选
- 实时显示优化

## 技术细节

### 关键参数
- `MIN_COUNT = 10`：最小匹配点数量阈值
- 缩放因子：0.35（可根据视频分辨率调整）
- Lowe比率：0.7
- RANSAC阈值：3.0

### 兼容性
- 支持常见视频格式（MP4、AVI等）
- 支持多种图像格式（PNG、JPG等）
- 跨平台运行（Windows、Linux、macOS）

## 作者信息

- **项目名称**：可口可乐Logo定位系统
- **技术栈**：Python + OpenCV + NumPy
- **应用领域**：计算机视觉、图像处理

## 许可证

本项目仅供学习和研究使用。