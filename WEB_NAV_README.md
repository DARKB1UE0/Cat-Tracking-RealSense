# 🌐 网页导航控制系统 (Web Navigation Control)

这个项目整合了 **RealSense 猫咪追踪** 和 **ROS2 导航控制**。你可以在同一个网页中看到 RealSense 相机的视频流，同时通过嵌入的远程桌面 (NoVNC) 直接控制 RViz 进行导航。

首次使用请先阅读 [网页使用说明](README.md)；硬件配置、建图保存及 AMCL 导航见 [工作空间 README](../README.md)。本页补充网页导航与键盘通信细节。

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
在已配置好硬件与工作空间的机器人电脑上，启动包含底盘、雷达、SLAM、Nav2 和 RViz 的完整系统：

```bash
source /opt/ros/humble/setup.bash
source ~/nav_ws/install/setup.bash
ros2 launch wheeltec_bringup slam_navigation.launch.py
```

默认串口为 `/dev/ttyACM0`，可通过 `serial_port:=/dev/ttyUSB0` 覆盖。若使用已有地图，改用 `bringup.launch.py mode:=nav map:=...`，具体命令见工作空间 README。单独的 `navigation.launch.py` 不会启动底盘、雷达或 RViz。

启动 VNC 前必须保证本机图形桌面上已有 RViz 窗口。

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

### 4. 键盘驾驶

在右侧「键盘驾驶」面板等待通信连接成功，取消 RViz 中正在执行的导航任务，再点击「启用键盘驾驶」。

| 按键 | 动作 |
|------|------|
| W / S | 前进 / 后退 |
| A / D | 向左 / 向右平移（麦轮） |
| J / K | 左转 / 右转 |
| 空格 / Esc | 停止并关闭键盘驾驶 |

- 按住持续移动，松开停止；支持 W+A、W+J 等组合，相反方向按键互相抵消。
- 默认移动速度为 **0.20 m/s**，转向速度为 **0.50 rad/s**，可通过滑块调节。斜向移动的合速度不超过选定移动速度。
- 切换窗口、隐藏页面、进入输入框或点击 RViz 远程桌面会关闭键盘驾驶；返回后需重新启用。RViz 内的按键由远程桌面处理。
- 页面通过当前主机的 **9090** 端口连接 rosbridge，以 **10 Hz** 向 `/cmd_vel` 发布 `geometry_msgs/msg/Twist`。仅查看页面或未按键时不会持续发送零速度干扰导航。
- 断线时清除按键状态，自动重连后不会恢复移动。网络中断导致停止消息无法送达时，由底盘驱动的 `cmd_vel_timeout` 兜底（本项目默认 **0.5 秒**，前提是没有其他节点继续发布速度）。
- 手动驾驶期间不要同时运行其他遥控或自动导航速度发布者；当前系统没有速度指令仲裁。
- 若只运行 `python3 web_app.py`，需另行启动 `ros2 launch rosbridge_server rosbridge_websocket_launch.xml`。推荐使用上面的 `./launch_web_nav.sh`。当前启动脚本使用 HTTP/WS；HTTPS 页面需要为 rosbridge 配置 WSS。

## 📁 文件说明
- `launch_web_nav.sh`: 一键启动脚本。
- `launch_rviz_web.sh`: VNC 与 RViz 窗口管理脚本（支持 Wayland）。
- `templates/index.html`: 网页前端，横向双栏布局。
- `static/style.css`: 暗色主题样式表。
- `static/script.js`: 前端交互逻辑（上传、追踪控制）。
- `static/teleop.js`: 键盘驾驶、速度发布、停止及断线重连。
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
