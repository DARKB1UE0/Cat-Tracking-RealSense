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

退出终端遥控，等待面板显示「通信已连接」，点击 **启用键盘驾驶**；网页会取消当前 Nav2 导航并在确认停止后接管。

| 按键 | 动作 |
|---|---|
| W / S | 前进 / 后退 |
| A / D | 左移 / 右移（麦轮平移） |
| J / K | 左转 / 右转 |
| 空格 / Esc | 停止并关闭驾驶 |

按住移动、松开停止。可以组合 W+A 或 W+J，相反方向相互抵消。默认移动速度为 **0.20 m/s**，转向速度为 **0.50 rad/s**；滑块上限分别提高至 **1.00 m/s**、**2.00 rad/s**。

切换窗口、进入输入框、隐藏页面或点击 RViz 远程桌面后会关闭驾驶，返回需重新启用。断线会清除按键状态，重连不会自动继续移动。

网页键盘速度经 `/cmd_vel_manual` 由网页后端转发至 `/cmd_vel`；自动追踪、等待导航取消或检测到其他 Nav2 任务时，后端拒绝转发键盘速度。需在已加载 ROS 环境的终端启动网页。其他直接发布 `/cmd_vel` 的终端遥控不受此仲裁，应先关闭。

## 4. 查看视频与识别猫咪

1. 打开页面后等待视频；未显示时点击「启动相机」。
2. 点击或拖拽上传区域选择清晰的目标猫照片，再点击「上传照片」。支持 JPG、PNG、GIF、BMP，最大 16 MB。
3. 点击「启动追踪」，等待模型初始化。
4. 查看视频中的绿色目标框、距离和匹配结果；其他检测到的猫以灰框标注。
5. 点击「停止追踪」结束识别，视频与驾驶控制独立运行。

「启动追踪」开启视觉识别；出现地图标注后，点击「自动追踪」才会驱动车辆。「停止追踪」会同时取消本页面服务发起的自动跟随任务。具体行为见下文。

## 5. 在网页中使用 RViz

右侧远程桌面是主机上已有 RViz 窗口的实时投屏，可直接操作鼠标。

- 已有地图模式：用 **2D Pose Estimate** 设置实际位置和朝向，确认扫描与地图对齐。
- 目标导航：使用 **Nav2 Goal** 设置目标；若现有 `2D Goal Pose` 不触发导航，添加 `nav2_rviz_plugins/GoalTool`。
- 回到键盘驾驶：点击网页「启用键盘驾驶」，后端自动取消当前导航，确认停止后接管。RViz 获得焦点时，键盘输入由远程桌面处理。

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
