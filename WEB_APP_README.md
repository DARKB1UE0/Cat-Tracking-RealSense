# 猫咪追踪Web应用使用指南

## 📋 项目说明

这是一个基于YOLOv8和ResNet50的猫咪追踪系统的Web界面。通过上传参考猫照片，系统可以使用RealSense相机实时追踪特定的猫。

## 🚀 快速开始

### 1. 安装依赖

首先确保你已经安装了所有必要的依赖包：

```bash
pip install -r requirements.txt
```

### 2. 启动Web服务器

运行以下命令启动Flask Web服务器：

```bash
python web_app.py
```

服务器将在 `http://localhost:5000` 启动。

### 3. 使用Web界面

1. 在浏览器中打开 `http://localhost:5000`
2. 点击上传区域或拖拽一张参考猫的照片
3. 点击"上传照片"按钮
4. 上传成功后，点击"启动追踪"按钮
5. 系统会弹出一个窗口显示实时追踪画面
6. 在追踪窗口中：
   - 按 `q` 键退出追踪
   - 按 `s` 键保存当前帧

## 📁 项目结构

```
Cat-Tracking-RealSense/
├── web_app.py              # Flask Web应用主程序
├── track_specific_cat.py   # 猫追踪核心代码
├── requirements.txt        # Python依赖包
├── yolov8n.pt             # YOLOv8模型文件
├── templates/
│   └── index.html         # Web界面HTML
├── static/
│   ├── style.css          # 样式文件
│   └── script.js          # JavaScript脚本
└── uploads/               # 上传的照片存储目录（自动创建）
```

## 🔧 技术栈

- **后端**: Flask (Python Web框架)
- **前端**: HTML5 + CSS3 + JavaScript
- **AI模型**: 
  - YOLOv8n (猫检测)
  - ResNet50 (特征提取)
- **相机**: Intel RealSense (深度相机)
- **图像处理**: OpenCV, PyTorch

## ⚙️ 功能特性

✅ 拖拽/点击上传参考猫照片  
✅ 实时预览上传的照片  
✅ 自动验证图片格式和大小  
✅ 实时状态监控  
✅ 友好的用户界面  
✅ 支持同时追踪多只猫并识别目标猫  
✅ 显示距离信息（通过RealSense深度相机）  

## 📝 使用说明

### 准备工作

1. **连接RealSense相机**: 确保Intel RealSense深度相机已正确连接到电脑
2. **准备参考照片**: 拍摄一张清晰的目标猫照片（最好是正面或侧面特写）

### 操作步骤

1. **启动服务器**:
   ```bash
   python web_app.py
   ```

2. **访问Web界面**: 浏览器打开 http://localhost:5000

3. **上传照片**: 
   - 点击上传区域选择照片
   - 或直接拖拽照片到上传区域
   - 支持格式：JPG, PNG, GIF, BMP
   - 最大文件大小：16MB

4. **启动追踪**: 点击"启动追踪"按钮后会弹出实时追踪窗口

5. **查看结果**: 
   - 绿色框标识目标猫
   - 灰色框标识其他猫
   - 显示相似度分数和距离信息

### 键盘快捷键（在追踪窗口中）

- `q`: 退出追踪
- `s`: 保存当前帧到文件

## 🎯 追踪参数

- **YOLOv8置信度阈值**: 0.5
- **ResNet相似度阈值**: 0.75
- **相似度差距要求**: 0.05
- **最小检测框尺寸**: 30x30 像素
- **相机分辨率**: 640x480 @ 30fps

## 🐛 故障排除

### 问题1: 无法启动相机
- 确保RealSense相机已正确连接
- 检查是否安装了pyrealsense2库
- 尝试重新插拔相机

### 问题2: 上传失败
- 检查文件格式是否正确
- 确保文件大小不超过16MB
- 检查uploads目录是否有写入权限

### 问题3: 追踪不准确
- 使用更清晰的参考照片
- 确保光线充足
- 参考照片应该包含猫的主要特征

### 问题4: 端口已被占用
如果5000端口已被占用，可以修改 `web_app.py` 中的端口号：
```python
app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
```

## 🌐 网络访问

如果需要从局域网其他设备访问，可以：

1. 确保防火墙允许5000端口
2. 使用服务器的IP地址访问，例如: `http://192.168.1.100:5000`

## 📊 性能优化

- 系统每2帧处理一次以提高性能
- 使用YOLOv8 nano版本（yolov8n.pt）保证速度
- ResNet特征提取使用GPU加速（如果可用）

## ⚠️ 注意事项

1. 追踪过程中请勿关闭浏览器，但可以最小化
2. 同一时间只能运行一个追踪任务
3. 确保RealSense相机驱动已正确安装
4. 建议使用性能较好的电脑以获得流畅体验

## 📞 技术支持

如有问题，请查看：
- RealSense官方文档: https://www.intelrealsense.com/
- YOLOv8文档: https://docs.ultralytics.com/
- Flask文档: https://flask.palletsprojects.com/

## 📄 许可证

请参考项目根目录下的LICENSE文件。
