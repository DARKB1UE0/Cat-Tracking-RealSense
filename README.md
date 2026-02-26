# 🐱 猫咪追踪与导航控制系统 (Cat Tracking & Nav System)

本项目是一个集成了 **深度学习猫咪追踪** (YOLOv8 + ResNet50) 与 **ROS2 机器人导航** 的综合系统。通过一个统一的 Web 界面，你可以实时监控猫咪状态、查看视频流，并通过远程桌面 (NoVNC) 直接控制机器人 RViz 进行导航。

---

## ✨ 功能特性

### 1. 🎯 智能追踪 (AI Tracking)
- **目标检测**: 使用 YOLOv8n 实时检测视频流中的所有猫。
- **特征匹配**: 使用 ResNet50 提取特征，通过余弦相似度（Default > 0.75）识别特定的目标猫。
- **距离测量**: 利用 Intel RealSense 深度相机实时测量目标距离。

### 2. 🌐 Web 控制台 (Web Interface)
- **横向双栏布局**: 左侧视频流（完整画面），右侧 RViz 远程桌面 + 追踪控制。
- **目标管理**: 支持上传参考图片，一键启动/停止 AI 追踪。
- **局域网访问**: 支持从同一网络内的其他设备访问 (`http://<机器人IP>:5000`)。
- **暗色主题**: 现代化深色 UI，适合长时间操作。

### 3. 🖥️ 远程导航 (Remote Navigation)
- **原生 RViz 集成**: 通过 NoVNC 技术，将机器人的 RViz 窗口直接嵌入网页。
- **无缝控制**: 在网页中直接操作 RViz（设置导航点、查看雷达图、调整参数），体验与本地一致。
- **Wayland 兼容**: 支持 Wayland 桌面环境（通过 XWayland 捕获 RViz）。
- **智能窗口识别**: 精确匹配 RViz 窗口（`- RViz` 标题匹配），避免误捕其他窗口。

---

## 🛠️ 环境要求

- **硬件**:
    - Intel RealSense 深度相机 (D400 系列)
    - 运行 Linux 的机器人底盘 (支持 ROS2 Humble)
- **软件**:
    - Ubuntu 22.04 + ROS2 Humble
    - Python 3.10+
    - 依赖库: `wmctrl`, `x11vnc`, `websockify`, `rosbridge_suite`
    - Wayland 桌面环境需要 XWayland 支持

---

## 🚀 快速开始

### 1. 启动机器人基础 (ROS 端)
在机器人/远程主机上，启动底层的导航栈和 RViz。
**注意：必须确保 RViz 窗口在屏幕上打开。**

```bash
ros2 launch wheeltec_bringup navigation.launch.py
```

### 2. 启动 Web 控制系统
新建一个终端，运行一键启动脚本：

```bash
cd ~/nav_ws/src/Cat-Tracking-RealSense
./launch_web_nav.sh
```

此脚本会自动：
1.  启动 `rosbridge_server`。
2.  启动 Flask Web 服务器 (Port 5000)。
3.  检测 RViz 窗口并启动 VNC 串流 (Port 6080)。

### 3. 打开浏览器控制
访问: `http://localhost:5000`

- **左侧**: 实时视频流（完整画面显示）。
- **右侧**: RViz 远程桌面 + 追踪控制面板。
    - 上传猫咪照片 → 点击 "启动追踪"。
    - 直接在 RViz 中点击设置 `2D Nav Goal` 导航。

---

## 📂 项目结构

```
Cat-Tracking-RealSense/
├── launch_web_nav.sh       # [入口] 主启动脚本
├── launch_rviz_web.sh      # [子脚本] VNC 与 RViz 窗口管理
├── web_app.py              # Flask 后端服务器
├── track_specific_cat.py   # AI 追踪核心逻辑
├── templates/
│   └── index.html          # Web 前端界面
├── static/
│   ├── style.css           # 样式表
│   ├── script.js           # 前端交互逻辑
│   ├── nav-script.js       # 导航脚本 (备用)
│   └── novnc/              # NoVNC 客户端库
└── uploads/                # 上传的猫咪图片存放目录
```

---

## 🐛 常见问题 (Troubleshooting)

### Q: VNC 显示 "Disconnect" 或无法连接？
- **A**: 
  1. 确保已在桌面上**打开了 RViz**（脚本需要捕捉 RViz 窗口）。
  2. 如在 Wayland 环境下，确保 RViz 通过 XWayland 运行。
  3. 检查 `/tmp/x11vnc.log` 查看错误日志。

### Q: VNC 显示了错误的窗口？
- **A**: 脚本通过匹配窗口标题中的 `- RViz` 来识别 RViz。确保 RViz 窗口标题包含此字符串。可用 `wmctrl -l` 查看所有窗口。

### Q: 无法在网页中点击/控制 RViz？
- **A**: 确保安装了 `wmctrl`: `sudo apt install wmctrl`。

### Q: 视频流卡顿？
- **A**: 检查网络带宽。VNC 配置了 `-noxdamage -noshm` 优化参数，主要瓶颈可能在 WiFi 信号。

### Q: AI 无法识别我的猫？
- **A**: 请上传一张光线充足、特征清晰的**正面**照片。

---

## 📄 许可证
本项目遵循开源许可证。
