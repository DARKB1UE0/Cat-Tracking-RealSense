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

默认串口为 `/dev/serial/by-id/usb-WCH.CN_USB_Single_Serial_0002-if00`，可通过 `serial_port:=/dev/ttyUSB0` 覆盖。若使用已有地图，改用 `bringup.launch.py mode:=nav map:=...`，具体命令见工作空间 README。单独的 `navigation.launch.py` 不会启动底盘、雷达或 RViz。

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
- 默认移动速度为 **0.20 m/s**，转向速度为 **0.50 rad/s**；滑块上限分别为 **1.00 m/s**、**2.00 rad/s**。斜向移动的合速度不超过选定移动速度。
- 切换窗口、隐藏页面、进入输入框或点击 RViz 远程桌面会关闭键盘驾驶；返回后需重新启用。RViz 内的按键由远程桌面处理。
- 页面通过当前主机的 **9090** 端口连接 rosbridge，以 **10 Hz** 向 `/cmd_vel_manual` 发布 `geometry_msgs/msg/Twist`。仅查看页面或未按键时不会持续发送零速度干扰导航。
- 断线时清除按键状态，自动重连后不会恢复移动。网络中断导致停止消息无法送达时，由底盘驱动的 `cmd_vel_timeout` 兜底（本项目默认 **0.5 秒**，前提是没有其他节点继续发布速度）。
- 网页后端在无自动追踪、无待取消目标、无其他 Nav2 任务时将键盘速度转发至 `/cmd_vel`，350 ms 未收到新手动速度时停车。其他直接发布 `/cmd_vel` 的节点不受该仲裁，应先关闭。
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

## RViz 目标猫位置

识别参考图对应的目标猫后，网页发布 `/cat/position` 和 `/cat/markers`；RViz 的 **Target Cat** 显示绿色位置球和坐标文字。相机近似位于车体中心和车顶高度（相对 base_link 高 0.15 m），位置叠加云台实测 Yaw 和 Roll，再转换到地图；Yaw=0 时朝前。第二轴物理上为 Roll，USB 协议字段仍名为 pitch。目标丢失或停止追踪会清除标注。

首次使用需重启网页服务；通过建图或导航 launch 启动的 RViz 会自动添加并启用 `/cat/markers`（Target Cat），无需手动 Add。已经打开的 RViz 可通过 File → Open Config 重新加载对应配置。配置和标注精度说明见 [目标猫位置标注](docs/cat_rviz_markers.md)。

## 自动跟随目标猫

先启动导航、完成地图定位，再上传照片并「启动追踪」，保持视频流打开。RViz 出现目标猫标注后，点击「自动追踪」。

- 通过 Nav2 规划到猫前方的停车点，目标距离为 **车体 base_link 中心到猫标注点的水平距离 1 米**，不是车头到猫的距离。接近到 1.1 米内取消前进，超过 1.3 米再继续；Nav2 定位、目标检测及停车误差会影响实际距离，不会主动倒车拉开距离。
- 跟随期间向 Nav2 发布 0.20 m/s 平移限速；猫位置变化时最多每秒更新一次目标，先等待旧目标取消完成。
- Yaw 保持水平朝向目标（±90°），Roll 逐步回到 0° 保持画面水平（手动仍限 ±30°）。当前没有物理 Pitch 俯仰轴，无法主动上下转动镜头；超出视野会停止跟随。
- 目标数据超过 1 秒、定位失效、云台故障、导航失败会取消跟随。网页心跳超过 0.8 秒、页面失焦/隐藏/关闭也会停止；恢复后必须重新点击启用。
- 「停止自动追踪」、空格/Esc、停止云台、停止相机或停止识别都会结束自动跟随。开始前先取消 RViz 导航；跟随期间不要从其他程序发起驾驶。

后端接口为 `/api/follow/status` 与 `/api/follow/control`。网页后端需 ROS Humble 的 `rclpy`、`nav2_msgs`、`tf2_ros`；使用 `bash launch_web_nav.sh` 会加载 ROS 环境。Python 更新后重启服务，浏览器按 Ctrl+F5 刷新。当前仅完成隔离 ROS/浏览器测试，尚未进行实车自动接近猫的测试。

### 从导航切换到键盘驾驶

点击「启用键盘驾驶」会停止自动追踪，并请求取消 Nav2 的单点导航、多点导航及航点任务（包括 RViz 发起的任务）。页面显示“正在取消导航”，等到取消任务达到终态、导航速度平滑器的旧输入消退后才启用按键，一般约 1～2 秒；切换期间按键不发送移动速度。无需导航到达目标朝向，也无需有效的 map 定位才能接管。

若 Nav2 拒绝取消、接口不可用或 6 秒内未确认停止，页面显示原因并保持键盘关闭；恢复后可重新点击。切换期间失焦、关闭页面、空格或 Esc 会撤销本次按键启用，迟到的成功响应不会重新启用驾驶。后端取消操作仍会完成。

键盘控制通过 `/api/teleop/enable` 请求接管，再经 `/cmd_vel_manual` 转发速度；平移滑块最大 1.00 m/s、转向最大 2.00 rad/s，默认仍为 0.20 m/s、0.50 rad/s。其他直接发布 `/cmd_vel` 的遥控程序不在取消范围内。
