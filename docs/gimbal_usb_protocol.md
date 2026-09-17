# 云台 USB 控制协议与联调

大疆 C 板通过 USB CDC 虚拟串口连接上位机。波特率参数用于串口 API 兼容，不决定 USB 传输速率。

截至 2026-09-17，已按用户要求撤回 yaw 顺时针 90° 安装偏移，恢复 GM6020 编码器原点及 DM4310P 反馈 0 rad 为零点。无上位机指令时回零保持功能保留；回退固件已通过完整编译、软件模拟及 ST-Link 烧录校验，并复位启动。真实 ROS/USB 完整控制链路仍需联调。

## 控制路径

`/gimbal/command → ros2_usb_gimbal.py → tools/gimbal_usb.py → USB CDC → GimbalUsbRxCallback → RobotCMDTask → GimbalTask → DJI/DM 驱动 → CAN1`

反馈路径为 `CAN1 电机反馈 → 电机解码 → GimbalTask → USB 状态帧 → /gimbal/status`。反馈角度来自编码器，不能用目标角度代替。两轴均使用 CAN1，当前波特率为 1 Mbit/s。

## 帧格式

所有多字节字段小端，浮点数为 IEEE754 float32：

`AA 55 | version:u8 | type:u8 | seq:u8 | length:u16 | payload | crc16:u16`

- version 固定为 1；length 是 payload 字节数，最大 32。
- CRC16/IBM（Modbus 参数），初值 `0xFFFF`，反射多项式 `0xA001`，覆盖 version 至 payload；CRC 本身低字节在前。
- seq 为滚动 8 位序号；当前版本不提供 ACK、去重或重传机制。
- 双端支持跨 USB 分包、粘包、噪声和 CRC 错误恢复；错误帧不刷新控制看门狗。

| 类型 | payload | 行为 |
|------|---------|------|
| `0x01` SETPOINT | `<ffBBH>`，12 字节 | 相对电机零点的 yaw、pitch（度），mode（0 锁存停止 / 1 位置控制），flags=0，reserved=0 |
| `0x02` ENABLE | 无 | 仅在最近 100 ms 内已有有效目标时使能；STOP 或超时后不能恢复旧目标 |
| `0x03` STOP | 无 | 锁存停止两轴并撤销目标，无指令时仍停止；恢复须发送 mode=1 的 SETPOINT |
| `0x04` PING | 无 | 请求实际状态；不续期运动指令 |
| `0x81` STATUS | `<ffBBHI>`，16 字节 | 实测 yaw、pitch，enabled，fault，reserved=0，age_ms |

mode 只允许 0/1；NaN、Inf、非零保留字段、错误长度或版本均被拒绝。有效 SETPOINT、STOP 和被接受的 ENABLE 更新 age_ms；连续正常使用建议直接发送 SETPOINT。

STATUS 正常以约 50 Hz 发布，USB 忙时稍后重试。age_ms 是距最近有效控制帧的毫秒数；enabled=1 表示两轴软件允许输出且达妙反馈为使能状态，不代表已经到达目标角度。

fault 是位掩码，可同时包含多项：

| 位值 | 含义 |
|------|------|
| `0x01` | 上位机控制指令超时或尚未收到指令；仅诊断，不阻止默认零位保持 |
| `0x02` | yaw 无反馈或反馈超过 50 ms |
| `0x04` | pitch 无反馈或反馈超过 50 ms |
| `0x08` | 达妙电机上报错误状态 |
| `0x10` | 目标角度无效或超出软件范围 |
| `0x20` | 尚未建立初始角度基准 |

开机未收到上位机指令，或最近的运动指令超过 100 ms 未更新时，下位机默认以 yaw=0°、pitch=0° 进行位置闭环，持续向两轴发送 CAN 控制帧，回到电机零点并保持。失效的非零目标不会继续执行。默认回零沿用位置/速度环的输出限幅，实际到位精度和负载保持能力需调试 PID 验证。

显式 STOP、SETPOINT mode=0、任一电机反馈超时、达妙报错、目标越界或尚未建立有效电机坐标基准时停止两轴。STOP/mode=0 的停止状态会保留到后续使能指令或固件重启，不会因 USB 超时自动变成回零。mode=0 的预置目标仅能在 100 ms 内通过显式 ENABLE 激活；STOP 撤销目标后单独 ENABLE 无效。故障存在时禁止默认回零，反馈恢复且故障消失后重新采用当前有效目标或默认零位；固件不会自动发达妙清错命令。

停止时 6020 发送零输出，4310P 发送失能帧；默认零位保持则继续输出位置闭环所需的电压/力矩，两者不同。实际响应受任务调度和 CAN 传输影响，DM 模式帧间隔为 20 ms。状态 `enabled=1, fault=0x01` 表示上位机指令缺失但两轴允许保持零位，不能将 `fault!=0` 一律解释为电机已失能。

## 电机配置与角度基准

集中配置见下位机 `application/gimbal/gimbal_config.h`。2026-09-16 用户确认 GM6020 ID=1，DM 参数已按用户提供的调试工具截图配置：

| 参数 | 当前配置 |
|------|--------|
| GM6020 拨码 ID | 1；命令 ID `0x1FF` 前两个字节，反馈 ID `0x205` |
| GM6020 控制方式 | 原版 GM6020 电压命令协议；不能直接用于新版电流命令协议 |
| DM4310P Motor CAN ID | `0x01`，MIT 模式 |
| DM4310P Master ID | `0x00`，反馈帧使用标准 CAN ID 0，属于有效 ID |
| DM P_MAX / V_MAX / T_MAX | `12.5 rad / 30 rad/s / 10 Nm`，与截图一致 |
| CAN 总线与速率 | 两轴 CAN1，1 Mbit/s（截图 CAN Baud=1M） |
| 软件角度范围 | yaw ±90°，pitch ±30° |
| 调试输出上限 | yaw 电压命令 ±3000，pitch 力矩 ±1 Nm |

P/V/T 是协议编解码范围，T_MAX=10 Nm 不代表要求电机输出 10 Nm；应用仍限制 pitch 力矩为 ±1 Nm。电机 ID=1 与 GM6020 拨码 ID=1 不冲突：DM 命令/反馈为 0x001/0x000，GM6020 命令/反馈为 0x1FF/0x205。上位机 USB 仍发送角度，CAN ID 和量程由下位机处理。

方向、机械范围和 MCU PID 仍需实机标定。截图中的电机内部环路参数与 MCU 串级 PID 是不同层级；代码不会远程修改电机持久参数。截图 CAN Timeout=0，电机端通信超时保护不能视为已启用；下位机的 100 ms 指令失效回零和 50 ms 电机反馈异常停机依赖 C 板正常运行及 CAN 命令可达。

零点使用电机自身的零点：GM6020 编码器值 0，DM4310P 反馈位置 0 rad。首次有效反馈仅用于建立坐标基准，绝不把开机姿态作为新零位。GM6020 在初始化时选取距当前角度最近的整圈零点，例如初始 355° 对应 -5°，回零目标为同一物理零点的 360°；随后固定该圈基准并跟踪跨圈变化。达妙保持它自身的反馈零位，不发送写编码器零点命令。

USB 指令和状态角度均相对上述电机零点，内部闭环统一使用度和度/秒。电机零点与车体正前方可能不同，相机 TF 的安装偏角必须重新核对；开机摆放朝向不会改变电机零位。`GIMBAL_YAW_SIGN`、`GIMBAL_PITCH_SIGN` 取 ±1，分别校准左转为正、抬头为正。`GimbalUsbSetInitialPose()` 是固件内部坐标接口，当前 USB 协议没有远程写零位命令。

当前达妙驱动每 20 ms 发送停止/使能模式帧并等待反馈，以获取停用状态下的角度。需验证该型号固件是否对这些帧回传状态；若不回传，应按实际电机协议增加状态查询，不能跳过反馈检查。

## ROS 2 使用

在已安装 ROS 2 Humble 的 Linux 环境，从 `Cat-Tracking-RealSense` 根目录启动，当前节点是独立脚本：

```bash
source /opt/ros/humble/setup.bash
python3 -m pip install pyserial
python3 ros2_usb_gimbal.py --ros-args -p port:=/dev/ttyACM0
```

串口设备名及权限以实机为准。节点启动、空闲和退出均不自动发送控制帧，由下位机执行默认零位保持或已锁存的 STOP。订阅采用 volatile、depth=1，不自动重放旧目标，不自动重连。

确认配置和机械零位后，在另一终端持续发送小角度目标，并观察状态：

```bash
ros2 topic pub -r 50 /gimbal/command std_msgs/msg/Float32MultiArray "{data: [5.0, 0.0, 1.0]}"
ros2 topic echo /gimbal/status
```

`/gimbal/status` 数组顺序是 `[yaw_deg, pitch_deg, enabled, fault, age_ms]`。停止目标发布进程后，超过 100 ms 固件回到电机零点并保持。若需要失能停机，先停止持续发布，再明确发送：

```bash
ros2 topic pub --once /gimbal/command std_msgs/msg/Float32MultiArray "{data: [0.0, 0.0, 0.0]}"
```

一个运动控制来源应独占 `/gimbal/command`；其他进程继续发送目标会在 STOP 后重新使能。一次性发送非零角度只有效约 100 ms，随后默认回零。要解除锁存 STOP 并恢复零位保持，可发送一次 `{data: [0.0, 0.0, 1.0]}`，之后即使没有新指令也继续保持零位。

认定完整实物链路通过，需要观察到：USB 实际状态回传、CAN1 命令及两轴反馈、实测角度随小目标变化、未连接上位机时回到电机零点、停止发布后回零保持、显式 STOP 后持续失能，以及断开任一电机反馈后两轴停止。已完成固件烧录并收到用户关于零位朝向的现场反馈，上述完整联调项目尚未全部验证。

本次控制链路修复不等于完成 RViz 目标标注：仍缺少检测结果到 `/cat_target_camera` 的实际发布接入，相机安装偏移、TF 方向和时间同步也需实机验证。

## 软件验证

上位机协议及 ROS 回调模拟回归：

```bash
python3 -m unittest discover -s tests -p test_gimbal_usb.py -v
```

下位机完整 ARM 编译和运行实际 C 代码的控制链路模拟，参见下位机 `tests/README.md`。模拟替换外设和时钟，不能验证电气接线、电机固件、控制稳定性或真实 RTOS 时序。
