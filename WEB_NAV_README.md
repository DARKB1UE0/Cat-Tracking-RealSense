# 🌐 网页导航控制系统 (Web Navigation Control)

这个项目整合了 **RealSense 猫咪追踪** 和 **ROS2 导航控制**。你可以在同一个网页中看到 RealSense 相机的视频流，同时通过嵌入的远程桌面 (NoVNC) 直接控制 RViz 进行导航。

## ✨ 功能特性 (v4.0 横向双栏版)

- **📷 实时视频流**: 左侧显示 RealSense 相机完整 RGB 画面。
- **🖥️ 远程桌面 (RViz)**: 右侧嵌入 RViz 窗口，支持缩放显示。
    - **原生体验**: 100% 还原 RViz 所有功能。
    - **无缝操作**: 支持鼠标与键盘直接控制。
    - **Wayland 兼容**: 支持 Wayland 桌面环境 (通过 XWayland)。
- **🎯 追踪控制**: 右侧面板上传猫咪照片，一键启动/停止追踪。
- **🌍 局域网访问**: 支持从同一网络的其他设备访问 (`http://<机器人IP>:5000`)。
- **🌙 暗色主题**: 现代深色 UI，适合长时间操作。

## 🚀 快速开始

### 1. 启动机器人的基础导航 & RViz
确保你在机器人/远程电脑上已经启动了 Nav2 和 RViz 界面。
```bash
ros2 launch wheeltec_bringup navigation.launch.py
```
*注意：必须保证屏幕上已有 RViz 窗口。*

### 2. 启动网页控制系统
打开一个新的终端，进入项目目录并运行启动脚本：

```bash
cd ~/nav_ws/src/Cat-Tracking-RealSense
./launch_web_nav.sh
```

这个脚本会自动执行以下操作：
1.  启动 `rosbridge_server` (用于网页与 ROS 通信，端口 9090)。
2.  检测 RViz 窗口并启动 VNC 串流 (端口 6080)。
3.  启动 Flask Web 服务器 (端口 5000)。

### 3. 使用网页控制
1.  **打开浏览器**: 访问 `http://localhost:5000`（或从其他设备访问 `http://<机器人IP>:5000`）。
2.  **查看视频**: 左侧显示 RealSense 实时完整画面。
3.  **操作 RViz**: 右侧 "远程桌面" 显示 RViz 缩放画面，可直接交互。
4.  **追踪控制**: 右侧上传猫咪照片，点击 "启动追踪"。

## 📁 文件说明
- `launch_web_nav.sh`: 一键启动脚本。
- `launch_rviz_web.sh`: VNC 与 RViz 窗口管理脚本（支持 Wayland）。
- `templates/index.html`: 网页前端，横向双栏布局。
- `static/style.css`: 暗色主题样式表。
- `static/script.js`: 前端交互逻辑（上传、追踪控制）。
- `static/novnc/`: NoVNC 客户端库。

## 🛠️ 常见问题
- **VNC 无法连接?**
    - 确保桌面上已打开 RViz 窗口。
    - Wayland 环境下检查 `/tmp/x11vnc.log` 日志。
    - 确保安装了 `wmctrl` 和 `x11vnc`。
- **VNC 显示了错误窗口?**
    - 脚本匹配标题含 `- RViz` 的窗口，用 `wmctrl -l` 检查。
- **无法连接 ROS?**
    - 确保 `rosbridge_server` 启动成功。
    - 确保浏览器和机器人处于同一局域网。
