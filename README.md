# 猫车网页使用说明

网页将 RealSense 视频、猫咪识别、RViz 远程桌面和键盘驾驶放在同一页面。完整硬件配置、ROS 安装、建图和保存地图步骤见 [工作空间使用说明](../README.md)。

## 1. 准备环境

- 主机：Ubuntu 22.04、ROS2 Humble，工作空间已编译并加载环境。
- 相机：连接 Intel RealSense D400 系列相机；避免同时被其他程序占用。
- 底盘与导航：先按下一节启动完整 ROS 系统。
- RViz 串流：本机图形桌面应已登录，存在可被 `wmctrl -l` 识别的 RViz 窗口；Wayland 下依赖 XWayland。
- 浏览器：Chrome、Firefox 或 Edge，同机或局域网访问。

首次安装网页依赖：

```bash
sudo apt install ros-humble-rosbridge-server wmctrl x11vnc websockify python3-pip
cd ~/nav_ws/src/Cat-Tracking-RealSense
python3 -m pip install -r requirements.txt
```

首次猫咪识别可能需要下载 ResNet50 等模型权重。若使用 Python 虚拟环境，先激活再启动服务。

## 2. 启动服务

### 终端 1：启动机器人与 RViz

每个新的 ROS 终端都先加载环境，然后从以下两个模式中选择一个。

```bash
source /opt/ros/humble/setup.bash
source ~/nav_ws/install/setup.bash
```

**实时建图与导航：**

```bash
ros2 launch wheeltec_bringup slam_navigation.launch.py
```

**已有地图导航：**

```bash
ros2 launch wheeltec_bringup bringup.launch.py mode:=nav \
  map:="$HOME/nav_ws/maps/testroom.yaml"
```

完整启动默认串口为 `/dev/serial/by-id/usb-WCH.CN_USB_Single_Serial_0002-if00`，必要时追加 `serial_port:=/dev/ttyUSB0`。已有地图模式使用 AMCL，需在 RViz 中确认或设置初始位姿。单独运行 `navigation.launch.py` 不会启动底盘、雷达或 RViz，不能替代上述完整入口。

### 终端 2：启动网页

```bash
cd ~/nav_ws/src/Cat-Tracking-RealSense
./launch_web_nav.sh
```

该脚本自动加载 ROS 环境，同时启动 Flask、rosbridge 和 RViz 的 VNC 转发。请保持终端运行；必须从项目目录启动，因为脚本和模型使用相对路径。

| 服务 | 默认端口 | 用途 |
|---|---|---|
| Flask | 5000 | 主页、视频、识别控制 |
| rosbridge | 9090 | 网页键盘驾驶与 ROS 通信 |
| websockify / noVNC | 6080 | 浏览器内的 RViz 远程桌面 |
| x11vnc | 5900 | RViz 窗口的 VNC 服务 |

浏览器访问 `http://localhost:5000`；其他设备访问 `http://机器人电脑IP:5000`。使用同一可达网络，不要将雷达 IP 当作网页地址。

当前 RViz 脚本绑定桌面 `:0`，`NOVNC_DIR` 写为 `/home/bigtruck/nav_ws/src/Cat-Tracking-RealSense/static/novnc`。更换用户名、目录或显示编号时修改 [launch_rviz_web.sh](launch_rviz_web.sh)。脚本启动时会终止已有 x11vnc/websockify 进程，不要与其他远程桌面服务共用。

## 3. 网页键盘驾驶

先取消 RViz 导航任务，退出终端遥控，再等待面板显示「通信已连接」，点击 **启用键盘驾驶**。

| 按键 | 动作 |
|---|---|
| W / S | 前进 / 后退 |
| A / D | 左移 / 右移（麦轮平移） |
| J / K | 左转 / 右转 |
| 空格 / Esc | 停止并关闭驾驶 |

按住移动、松开停止。可以组合 W+A 或 W+J，相反方向相互抵消。默认移动速度为 **0.20 m/s**，转向速度为 **0.50 rad/s**，使用滑块调节。

切换窗口、进入输入框、隐藏页面或点击 RViz 远程桌面后会关闭驾驶，返回需重新启用。断线会清除按键状态，重连不会自动继续移动。

当前没有导航与手动驾驶的速度仲裁，不要让多个来源同时控制底盘。底盘默认 0.5 秒指令超时仅在没有其他节点持续发送速度时生效。网页键盘驾驶已通过 28 项浏览器模拟检查，实车验证仍待完成，见 [工作日志](WORK_LOG.md)。

## 4. 查看视频与识别猫咪

1. 打开页面后等待视频；未显示时点击「启动相机」。
2. 点击或拖拽上传区域选择清晰的目标猫照片，再点击「上传照片」。支持 JPG、PNG、GIF、BMP，最大 16 MB。
3. 点击「启动追踪」，等待模型初始化。
4. 查看视频中的绿色目标框、距离和匹配结果；其他检测到的猫以灰框标注。
5. 点击「停止追踪」结束识别，视频与驾驶控制独立运行。

这里的「追踪」是视觉识别和距离显示，尚未实现底盘自动跟随猫咪。「停止追踪」也不等于停车。

## 5. 在网页中使用 RViz

右侧远程桌面是主机上已有 RViz 窗口的实时投屏，可直接操作鼠标。

- 已有地图模式：用 **2D Pose Estimate** 设置实际位置和朝向，确认扫描与地图对齐。
- 目标导航：使用 **Nav2 Goal** 设置目标；若现有 `2D Goal Pose` 不触发导航，添加 `nav2_rviz_plugins/GoalTool`。
- 回到键盘驾驶：先取消导航，点击网页「启用键盘驾驶」。RViz 获得焦点时，键盘输入由远程桌面处理。

## 6. 停止与重新启动

1. 取消导航，点击网页「停止」，确认车体停稳；如有终端遥控，按 Q 退出。
2. 点击「停止追踪」。需要保存地图时，在停止 ROS 前完成保存。
3. 在网页启动终端按 `Ctrl+C`；需要关闭整套系统时，再停止 ROS 终端。
4. 重新启动时仍按「机器人 → 网页 → 浏览器」顺序。

前端文件改动后按 `Ctrl+F5` 强制刷新。Python 后端改动后需重启网页服务；当前启动关闭了自动重载。

## 7. 常见问题

| 现象 | 检查方法 |
|---|---|
| 页面打不开 | 检查 Flask 终端，确认端口 5000、主机 IP 和网络可达性 |
| 键盘驾驶按钮不可用 | 检查 rosbridge 终端与 9090 端口；只运行 `web_app.py` 不会启动 rosbridge |
| 按 WASD 没反应 | 确认已启用驾驶、焦点不在 RViz/输入框，底盘驱动已启动 |
| VNC 一直等待 RViz | 用 `wmctrl -l` 查看标题是否以 `- RViz` 结尾；核对显示编号与桌面会话 |
| VNC 黑屏或断开 | 检查 `/tmp/x11vnc.log`、5900/6080 端口及是否重复启动 |
| 无视频或相机初始化失败 | 检查 USB、设备权限和占用情况，关闭 RealSense Viewer 后重试 |
| 识别启动慢或失败 | 检查依赖、权重缓存/下载和 Flask 错误输出 |
| HTTPS 页面无法驾驶 | 当前部署使用 HTTP/WS；HTTPS 需要为 rosbridge 配置 WSS |

只需视频和识别时，可在项目目录执行 `python3 web_app.py`；该模式不自动提供键盘驾驶所需的 rosbridge 或 RViz 串流。

## 8. 文件与开发记录

| 文件 | 用途 |
|---|---|
| [launch_web_nav.sh](launch_web_nav.sh) | 网页综合启动入口 |
| [launch_rviz_web.sh](launch_rviz_web.sh) | RViz 窗口捕获与 VNC 服务 |
| [web_app.py](web_app.py) | Flask、视频流和识别接口 |
| [track_specific_cat.py](track_specific_cat.py) | YOLOv8 与 ResNet50 猫咪匹配 |
| [templates/index.html](templates/index.html) | 页面布局 |
| [static/script.js](static/script.js) | 相机、上传和追踪交互 |
| [static/teleop.js](static/teleop.js) | 键盘驾驶与 ROS 通信 |
| [static/style.css](static/style.css) | 页面样式 |
| [WEB_NAV_README.md](WEB_NAV_README.md) | 网页导航及键盘通信细节 |
| [BROWSER_GUIDE.md](BROWSER_GUIDE.md) | 视频流操作、接口说明 |
| [WORK_LOG.md](WORK_LOG.md) | 日志规范、开发与验证记录 |

浏览器模拟测试（需 Chrome/Chromium 与 Jinja2，不连接实车）：

```bash
cd ~/nav_ws/src/Cat-Tracking-RealSense
python3 -m unittest discover -s tests -v
```

阶段性开发、修复或联调完成后，按工作日志中的模板补充验证结果和待办事项。项目许可证见 [LICENSE](LICENSE)，内嵌组件遵循各自许可声明。

## RViz 目标猫位置

识别参考图对应的目标猫后，网页发布 `/cat/position` 和 `/cat/markers`；RViz 的 **Target Cat** 显示绿色位置球和坐标文字。相机近似位于车体中心和车顶高度（相对 base_link 高 0.15 m），位置叠加云台实测 yaw，再转换到地图；yaw=0 时朝前，Pitch 暂按水平近似。目标丢失或停止追踪会清除标注。

首次使用需重启网页服务；通过建图或导航 launch 启动的 RViz 会自动添加并启用 `/cat/markers`（Target Cat），无需手动 Add。已经打开的 RViz 可通过 File → Open Config 重新加载对应配置。配置和标注精度说明见 [目标猫位置标注](docs/cat_rviz_markers.md)。
