# 目标猫位置标注

网页识别到参考图匹配的目标猫后，从对齐到彩色画面的深度帧中取检测框中心 5×5 邻域有效深度中位数，使用该帧内参反投影成三维光学坐标，再按图像接收时刻的 TF 转换到 `map`。

- `/cat/position`：`geometry_msgs/msg/PointStamped`，目标的地图三维位置，单位米。
- `/cat/markers`：`visualization_msgs/msg/MarkerArray`，绿色球体和 `Target cat (x, y) m` 文字。
- 建图和导航的 RViz 配置已添加 **Target Cat**。已打开的 RViz 可通过 Add → By topic → `/cat/markers` → MarkerArray 添加，或重新加载配置文件。
- 只标注参考图匹配的目标，不标注所有猫。目标丢失、停止追踪或深度无效会清除；没有新结果时最迟在图像接收 2 秒后清除，避免旧位置被误认成当前目标。识别耗时超过 2 秒的结果不发布。
- 标注是猫检测框中心对应的可见表面估计，不是猫的脚底。单纯标注不会驱动车辆；点击「自动追踪」后，跟随模块才基于标注生成距猫 1 米的导航目标。

## 相机坐标和安装配置

已按用户要求在 `camera_mount.json` 中配置：相机安装在云台上，近似位于车体中心、车顶高度，相对 base_link 高 0.15 m（离地约 0.225 m）。yaw=0 时朝车头、图像上方朝上；正 yaw 向左。相机与转轴偏移暂按零估计。

地图位置按以下姿态转换：**云台实测 Roll 与 Yaw → 车体在 map 中的姿态**。使用图像接收时采样的云台 USB 反馈角度，不使用滑条设定角；云台断开、反馈过期或有阻断故障时不标注。当前 `second_axis: roll`、`follow_secondary: true`，第二轴绕机械 +X 旋转；协议仍使用 pitch 字段。`pitch_sign` 暂按右手正方向 +1，实际编码器方向仍需标定。旧配置未填写 second_axis 时兼容原 Pitch 轴，follow_pitch 作为旧开关仍被支持。

这个模式直接利用现有 `map → odom → base_footprint → base_link` TF，无需新增相机 TF。若移除安装配置，才改用外部相机 TF；缺失时页面提示等待，不会把光学坐标当作地图坐标。

没有安装配置文件时使用 TF 模式，查询 `map → camera_color_optical_frame`。RealSense 光学坐标为 **X 右、Y 下、Z 前**。固定在车体上的相机可使用经过测量的静态 TF；装在云台上的相机须使用随实测角度变化的 TF。

也可以由网页按测量的安装参数计算。将 `camera_mount.example.json` 复制为 `camera_mount.json` 并填入参数，或通过 `CAT_CAMERA_CONFIG` 指定其他配置文件。示例中的 null 是待测参数，不能直接用于标注。修改后重启网页服务。

云台模式 `mode: gimbal`：

| 参数 | 定义 |
| --- | --- |
| `mount_xyz_m` | yaw 轴心相对 base_link 的前、左、上偏移（米） |
| `mount_rpy_deg` | 云台零位基准相对 base_link 的 roll/pitch/yaw（度），包含电机零位与车头方向的偏差 |
| `pitch_xyz_m` | 第二轴轴心相对 yaw 轴心的偏移，在 yaw 零位坐标系中表示 |
| `camera_xyz_m` | 相机光学中心相对第二轴轴心的偏移，在第二轴零位机械坐标系中表示 |
| `camera_rpy_deg` | 相机光学坐标系相对第二轴零位机械坐标系的旋转；若镜头正前、图像上方朝上，常见值为 [-90, 0, -90]，须按实物确认 |
| `yaw_sign` | 编码器正向与绕 +Z 右手旋转的关系：+1 或 -1 |
| `pitch_sign` | 第二轴编码器符号：+1 或 -1；Roll 绕 +X，Pitch 绕 +Y，按右手规则 |
| `second_axis` | 当前 roll；兼容 pitch，决定第二轴旋转方向 |
| `follow_secondary` | 是否叠加第二轴实测角度，当前 true；Yaw 始终叠加 |
| `map_frame` | 默认 map，需要导航或 SLAM 发布地图 TF |

变换顺序为：安装基准 → yaw 旋转 → 第二轴轴心平移 → 第二轴旋转 → 相机平移与光学旋转。使用接收图像时刻采样的云台实测角度，不用滑条目标代替反馈。云台反馈断开或有电机故障时停止标注。

固定相机可用 `mode: fixed`，仅需 `camera_xyz_m` 和 `camera_rpy_deg`，它们直接表示相机光学坐标系到 base_link 的安装变换。TF 模式可用 `mode: tf` 并设置 `camera_frame`、`map_frame`。

时间戳使用接收 RealSense 帧时的 ROS 系统时钟，在识别推理之前记录；未做硬件时钟同步，因此快速运动时仍有采集与 USB 传输时延误差。

## 启动与验证

```bash
source /opt/ros/humble/setup.bash
source ~/nav_ws/install/setup.bash
# 当前已配置车体中心、车顶高度近似，并叠加云台实测 Yaw
bash launch_web_nav.sh
```

开启导航/SLAM、上传目标猫图片并启动网页追踪，保持视频流打开。页面“地图标注”会显示目标状态和坐标。仅用 `python3 web_app.py` 时也需要先 source ROS 环境；未加载 ROS 时相机功能仍可用，标注会显示依赖错误。

```bash
ros2 topic echo /cat/position --once
ros2 topic echo /cat/markers --once
```

需要 ROS Humble 的 rclpy、tf2_ros、tf2_geometry_msgs、visualization_msgs；当前系统已具备。
