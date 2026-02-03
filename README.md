# 特定猫咪追踪与距离测量系统

基于 YOLOv8 + ResNet50 深度学习特征匹配的智能猫咪识别与追踪系统。

## 🎯 功能特点

- ✅ **YOLOv8 目标检测** - 使用最新的 YOLOv8 模型进行猫咪检测
- ✅ **ResNet50 深度特征提取** - 提取 2048 维深度特征向量进行精确匹配
- ✅ **余弦相似度匹配** - 在深度特征空间进行精确比对，区分不同的猫
- ✅ **实时距离测量** - 使用 Intel RealSense 深度相机测量目标猫咪的实时距离
- ✅ **多猫区分** - 可以在多只猫中识别并追踪特定的一只
- ✅ **可视化标注** - 绿色框标注目标猫，灰色框显示其他猫

## 📋 系统要求

### 硬件要求
- Intel RealSense 深度相机（D400 系列，如 D435）
- CPU: 支持 AVX 指令集的现代处理器
- 内存: 至少 4GB RAM

### 软件要求
- Ubuntu 20.04/22.04 或其他 Linux 发行版
- Python 3.8 - 3.11
- Intel RealSense SDK

## 🚀 安装步骤

### 1. 安装系统依赖

```bash
# 安装 RealSense SDK
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main"
sudo apt update
sudo apt install librealsense2-devel librealsense2-utils
```

### 2. 安装 Python 依赖

```bash
# 进入项目目录
cd Yolo-Object-Detection-and-Distance-Measurement-With-Intel-Realsense-Camera

# 安装 PyTorch CPU 版本（适合 AMD 显卡或无独显的系统）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --user

# 安装其他依赖
pip install ultralytics scipy pyrealsense2 opencv-python numpy pillow --user
```

## 📖 使用方法

### 1. 准备参考图片

准备一张你想追踪的猫的清晰照片，放在项目目录下。

```bash
# 例如：my_cat.jpg
```

### 2. 运行追踪程序

```bash
python3 track_specific_cat.py my_cat.jpg
```

### 3. 操作说明

- **'q' 键** - 退出程序
- **'s' 键** - 保存当前帧到文件

## ⚙️ 工作原理

### 1. YOLOv8 猫咪检测
- 使用 YOLOv8n（nano版本）实时检测视频流中的所有猫
- 检测置信度阈值：0.5
- 处理速度：15-20 FPS（CPU模式）

### 2. ResNet50 特征提取
- 从参考图片中提取 2048 维深度特征向量
- 对每只检测到的猫提取特征向量
- 特征归一化确保尺度一致性

### 3. 余弦相似度匹配
- 计算检测到的猫与参考猫的余弦相似度
- 相似度范围：0-1（1表示完全相同，0表示完全不同）
- **默认阈值：0.75** - 只有相似度 ≥ 0.75 才认为是目标猫
- **差距要求：0.05** - 目标猫必须比第二名高出至少 0.05

### 4. 距离测量
- 使用 RealSense 深度传感器获取深度信息
- 计算猫咪中心点的实时距离（米）
- 显示在画面标注上

## 🎨 显示说明

### 目标猫（绿色粗框）
```
TARGET Distance: 1.23m
Similarity: 0.856
```

### 其他猫（灰色细框）
```
Other cat (0.623)
```

### 底部信息
```
Max similarity: 0.856  # 当前帧最高相似度
```

## 🔧 参数调整

### 修改相似度阈值

编辑 `track_specific_cat.py` 第 211 行：

```python
if target_cat['match_score'] < 0.75:  # 修改这个值
```

推荐设置：
- **0.75** - 默认，平衡准确度和识别率
- **0.80** - 更严格，减少误识别
- **0.70** - 更宽松，提高识别率

### 修改检测帧率

编辑 `track_specific_cat.py` 第 173 行：

```python
if frame_count % 2 == 0:  # 每2帧检测一次，改为1可以每帧都检测
```

## 📊 性能优化

### CPU 模式（默认）
- 使用场景：AMD 显卡或无独显系统
- 处理速度：15-20 FPS
- 内存占用：约 2GB

### GPU 模式（可选）
如果有 NVIDIA GPU，可以安装 CUDA 版本的 PyTorch：

```bash
# 卸载 CPU 版本
pip uninstall torch torchvision

# 安装 CUDA 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🐛 常见问题

### Q1: 找不到相机
**A:** 检查相机连接并确认 RealSense SDK 已安装：
```bash
rs-enumerate-devices
```

### Q2: 识别不准确
**A:** 可能原因：
- 参考图片质量不佳 → 使用清晰的正面照片
- 光照差异太大 → 在相似光照条件下使用
- 阈值设置不当 → 调整相似度阈值

### Q3: 程序运行缓慢
**A:** 优化方法：
- 降低检测频率（改为每 3-4 帧检测一次）
- 使用 GPU 加速
- 降低相机分辨率

### Q4: 所有猫相似度都很低
**A:** 正常现象：
- 手机屏幕展示图片与真实猫差异大
- 角度、光照、距离变化都会影响相似度
- 真实场景中目标猫通常能达到 0.7-0.9 的相似度

## 📝 技术栈

- **目标检测**: YOLOv8 (Ultralytics)
- **特征提取**: ResNet50 (torchvision)
- **深度相机**: Intel RealSense D400 系列
- **深度学习框架**: PyTorch (CPU/CUDA)
- **计算机视觉**: OpenCV
- **数值计算**: NumPy, SciPy

## 🙏 致谢

本项目基于以下开源项目改进：
- [Yolo-Object-Detection-and-Distance-Measurement-with-Zed-camera](https://github.com/MehmetOKUYAR/Yolo-Object-Detection-and-Distance-Measurement-with-Zed-camera)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

## 📄 许可证

本项目遵循原项目的开源许可证。

---

**Happy Tracking! 🐱**
