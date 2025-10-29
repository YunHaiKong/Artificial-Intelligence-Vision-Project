# Lena图像合成系统

## 项目概述

本项目实现了一个基于OpenCV的图像合成系统，能够在经典的Lena图像上进行多种图像合成操作，包括添加OpenCV标志、佩戴花朵装饰、佩戴眼镜等。系统采用位运算和掩码技术，实现了高质量的无缝图像合成效果。

## 功能特点

- **多任务图像合成**：支持多种图像合成任务
- **交互式ROI选择**：支持鼠标交互选择感兴趣区域
- **精确掩码处理**：基于位运算的精确图像合成
- **无缝融合**：高质量的无缝图像融合效果
- **实时预览**：完整的处理过程可视化

## 技术实现流程

### 1. 基础图像合成流程
- 读取Lena原始图像
- 读取要合成的目标图像（OpenCV标志、花朵、眼镜等）
- 调整目标图像尺寸
- 创建掩码和反掩码
- 使用位运算进行图像合成
- 将合成结果放回原图像

### 2. 掩码处理技术
- 灰度图像转换
- 二值化处理
- 位运算掩码创建
- 前景和背景分离
- 图像融合

### 3. 交互式ROI选择
- 使用cv2.selectROI进行区域选择
- 自动提取ROI区域
- 动态调整合成位置

## 具体任务实现

### 任务1：给Lena图片左上角增加OpenCV标志

#### 实现步骤：
1. **读取Lena图像**：从文件系统读取经典Lena图像
2. **读取OpenCV标志**：加载OpenCV标志图像
3. **尺寸调整**：调整标志图像尺寸以匹配Lena左上角区域
4. **掩码创建**：
   - 将标志图像转换为灰度图
   - 二值化处理创建掩码
   - 创建反掩码用于背景保留
5. **前景提取**：使用掩码提取标志前景
6. **背景保留**：使用反掩码保留Lena背景
7. **图像合成**：使用cv2.add将前景和背景融合
8. **结果替换**：将合成结果放回Lena左上角

#### 核心代码：
```python
# 读取Lena图像和OpenCV标志
lena = cv2.imread('../cvimages/ch03/lena.jpg')
logo = cv2.imread('opencv_logo.png')

# 提取ROI区域
h, w = logo.shape[:2]
roi = lena[:h, :w]

# 创建掩码和反掩码
gray_logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(gray_logo, 10, 255, cv2.THRESH_BINARY)
mask_lena = cv2.bitwise_not(mask)

# 前景和背景处理
foreground = cv2.bitwise_and(logo, logo, mask=mask)
background = cv2.bitwise_and(roi, roi, mask=mask_lena)

# 图像合成
lena_logo = cv2.add(foreground, background)
lena[:h, :w] = lena_logo
```

### 任务2：给Lena头戴小花

#### 实现步骤：
1. **交互式ROI选择**：使用cv2.selectROI选择头部区域
2. **花朵图像处理**：读取并调整花朵图像尺寸
3. **掩码处理**：创建花朵掩码和反掩码
4. **图像合成**：将花朵合成到选择的ROI区域
5. **结果展示**：显示最终合成效果

#### 核心代码：
```python
# 交互式ROI选择
r = cv2.selectROI(lena)
x, y, w, h = r
roi = lena[y:y+h, x:x+w]

# 花朵图像处理
flower = cv2.imread('flower.png')
flower = cv2.resize(flower, (w, h))

# 掩码创建和图像合成
gray_flower = cv2.cvtColor(flower, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(gray_flower, 10, 255, cv2.THRESH_BINARY)
mask_lena = cv2.bitwise_not(mask)

foreground = cv2.bitwise_and(flower, flower, mask=mask)
background = cv2.bitwise_and(roi, roi, mask=mask_lena)

lena_flower = cv2.add(foreground, background)
lena[y:y+h, x:x+w] = lena_flower
```

### 任务3：给Lena戴上眼镜

#### 实现步骤：
1. **ROI区域选择**：选择眼部区域
2. **眼镜图像处理**：读取并调整眼镜图像
3. **掩码合成**：使用相同的掩码技术进行合成
4. **结果融合**：将眼镜无缝融合到眼部区域

## 文件结构

```
给lena贴图/
├── 给lena贴图.html              # 项目报告文档
├── markdown/                    # 处理结果图像
│   ├── addlogo/
│   │   ├── overall.png
│   │   ├── step4.png
│   │   └── step8.png
│   └── ...
├── cvimages/ch03/              # 原始图像文件
│   ├── lena.jpg
│   ├── lena02.png
│   └── ...
├── opencv_logo.png             # OpenCV标志图像
├── flower.png                  # 花朵图像
├── glass.png                   # 眼镜图像
└── README.md                   # 项目说明文档
```

## 环境要求

### 必备依赖
```bash
pip install opencv-python
pip install numpy
pip install matplotlib
```

### 推荐版本
- Python 3.7+
- OpenCV 4.5+
- NumPy 1.19+
- Matplotlib 3.3+

## 使用方法

### 运行方式

1. **HTML文档查看：**
   ```bash
   # 直接在浏览器中打开 给lena贴图.html
   # 查看完整的项目报告和处理结果
   ```

2. **代码执行：**
   - 确保所有图像文件路径正确
   - 按顺序执行各个任务代码块
   - 对于交互式任务，使用鼠标选择ROI区域

### 交互式操作

1. **ROI区域选择：**
   - 运行包含cv2.selectROI的代码
   - 在图像窗口中使用鼠标拖拽选择区域
   - 按空格或回车确认选择

2. **结果查看：**
   - 每个处理步骤都有对应的图像显示
   - 最终合成结果会自动显示

## 核心算法

### 掩码技术原理
- **灰度转换**：将彩色图像转换为灰度图像
- **二值化**：使用阈值处理创建二值掩码
- **位运算**：
  - cv2.bitwise_and：按位与运算
  - cv2.bitwise_not：按位非运算
  - cv2.add：图像加法融合

### 图像合成算法
1. **前景提取**：使用掩码提取目标图像的有效区域
2. **背景保留**：使用反掩码保留原始图像的背景
3. **图像融合**：将前景和背景相加得到合成结果

### ROI选择算法
- cv2.selectROI提供交互式区域选择
- 返回选择的矩形区域坐标(x, y, w, h)
- 支持动态调整选择区域

## 项目亮点

1. **多任务支持**：一个系统支持多种图像合成任务
2. **交互式操作**：支持鼠标交互选择合成位置
3. **无缝融合**：基于掩码技术的高质量图像合成
4. **技术全面**：涵盖了OpenCV图像处理的核心技术
5. **教育价值**：适合学习图像处理和计算机视觉基础

## 扩展应用

- 可扩展支持更多图像合成任务
- 可集成人脸检测自动定位合成位置
- 可添加图像滤镜和特效
- 可支持批量图像处理
- 可开发为图像编辑工具

## 技术细节

### 图像处理参数
- 二值化阈值：10
- 掩码类型：THRESH_BINARY
- 图像融合：cv2.add线性融合

### 文件路径管理
- 使用相对路径引用图像文件
- 支持多种图像格式（jpg, png等）
- 自动处理图像尺寸匹配

### 可视化设置
- 使用Matplotlib进行图像显示
- BGR到RGB颜色空间转换
- 自动调整图像显示尺寸

## 作者信息

- **学号：** 230200821
- **姓名：** 康嘉祥
- **课程：** 人工智能视觉
- **指导教师：** 李杰
- **完成时间：** 2025年6月

## 许可证

本项目仅用于学习和教育目的。