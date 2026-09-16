# 云台 USB 控制协议与联调

大疆 C 板通过 USB CDC 虚拟串口连接上位机。波特率参数用于串口 API 兼容，不决定 USB 传输速率。

截至 2026-09-16，已修复并接通代码中的指令和反馈路径，ARM 固件编译通过；软件模拟不能证明电机实物已经可控。尚未烧录、运行真实 ROS 2 或连接 USB/CAN 电机联调。

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
| `0x01` SETPOINT | `<ffBBH>`，12 字节 | yaw、pitch（度），mode（0 停止 / 1 位置控制），flags=0，reserved=0 |
| `0x02` ENABLE | 无 | 仅在最近 100 ms 内已有有效目标时使能；STOP 或超时后不能恢复旧目标 |
| `0x03` STOP | 无 | 停止两轴并撤销目标；恢复必须重新发送 SETPOINT |
| `0x04` PING | 无 | 请求实际状态；不续期运动指令 |
| `0x81` STATUS | `<ffBBHI>`，16 字节 | 实测 yaw、pitch，enabled，fault，reserved=0，age_ms |

mode 只允许 0/1；NaN、Inf、非零保留字段、错误长度或版本均被拒绝。有效 SETPOINT、STOP 和被接受的 ENABLE 更新 age_ms；连续正常使用建议直接发送 SETPOINT。

STATUS 正常以约 50 Hz 发布，USB 忙时稍后重试。age_ms 是距最近有效控制帧的毫秒数；enabled=1 表示两轴软件允许输出且达妙反馈为使能状态，不代表已经到达目标角度。

fault 是位掩码，可同时包含多项：

| 位值 | 含义 |
|------|------|
| `0x01` | 上位机控制指令超时或尚未收到指令 |
| `0x02` | yaw 无反馈或反馈超过 50 ms |
| `0x04` | pitch 无反馈或反馈超过 50 ms |
| `0x08` | 达妙电机上报错误状态 |
| `0x10` | 目标角度无效或超出软件范围 |
| `0x20` | 尚未建立初始角度基准 |

超过 100 ms 没有有效控制帧，两轴进入停止处理。任一电机反馈超时或达妙报错也停止两轴。6020 发送零输出，4310P 发送失能帧；这是软件停止，不是机械锁止。实际响应还受任务调度和 CAN 传输影响，DM 模式帧间隔为 20 ms。

## 电机配置与角度基准

集中配置见下位机 `application/gimbal/gimbal_config.h`。以下是待实物确认的默认值：

| 参数 | 默认值 |
|------|--------|
| GM6020 拨码 ID | 1；命令 ID `0x1FF` 前两个字节，反馈 ID `0x205` |
| GM6020 控制方式 | 原版 GM6020 电压命令协议；不能直接用于新版电流命令协议 |
| DM4310P Motor CAN ID | `0x01`，MIT 模式 |
| DM4310P Master ID | `0x11`，这是反馈帧 ID，不等于 Motor CAN ID |
| DM P_MAX / V_MAX / T_MAX | `12.56637 rad / 45 rad/s / 18 Nm`，沿用旧驱动值，尚未核实 |
| 软件角度范围 | yaw ±90°，pitch ±30° |
| 调试输出上限 | yaw 电压命令 ±3000，pitch 力矩 ±1 Nm |

必须把 ID、模式、量程、方向、机械范围和 PID 与实物配置对应。尤其是达妙量程若不一致，角度和力矩编解码都会错误，代码中的 1 Nm 限制也不代表真实 1 Nm。代码不会远程修改电机的工作模式或量程。

固件在开机后两轴首次收到有效反馈时建立零位，一次开机只记录一次。指令和状态均是相对该零位的关节角；闭环内部统一使用度和度/秒。若要把它解释为车体朝向，开机前须把云台视线对齐车头并调平，或完成专门的零位标定。`GIMBAL_YAW_SIGN`、`GIMBAL_PITCH_SIGN` 取 ±1，分别校准左转为正、抬头为正。`GimbalUsbSetInitialPose()` 是固件内部接口，当前 USB 协议没有远程写零位命令。

当前达妙驱动每 20 ms 发送停止/使能模式帧并等待反馈，以获取停用状态下的角度。需验证该型号固件是否对这些帧回传状态；若不回传，应按实际电机协议增加状态查询，不能跳过反馈检查。

## ROS 2 使用

在已安装 ROS 2 Humble 的 Linux 环境，从 `Cat-Tracking-RealSense` 根目录启动，当前节点是独立脚本：

```bash
source /opt/ros/humble/setup.bash
python3 -m pip install pyserial
python3 ros2_usb_gimbal.py --ros-args -p port:=/dev/ttyACM0
```

串口设备名及权限以实机为准。节点连接时发送 STOP，订阅采用 volatile、depth=1；不自动重放目标，不自动重连恢复运动。

确认配置和机械零位后，在另一终端持续发送小角度目标，并观察状态：

```bash
ros2 topic pub -r 50 /gimbal/command std_msgs/msg/Float32MultiArray "{data: [5.0, 0.0, 1.0]}"
ros2 topic echo /gimbal/status
```

`/gimbal/status` 数组顺序是 `[yaw_deg, pitch_deg, enabled, fault, age_ms]`。停止目标发布进程后，超过 100 ms 固件触发停机；也可停止持续发布后发送：

```bash
ros2 topic pub --once /gimbal/command std_msgs/msg/Float32MultiArray "{data: [0.0, 0.0, 0.0]}"
```

一个运动控制来源应独占 `/gimbal/command`；其他进程继续发送目标会在 STOP 后重新使能。一次性发送角度只有效约 100 ms，不能用来要求长时间保持位置。

认定实物链路通过，需要同时观察到：USB 实际状态回传、CAN1 命令及两轴反馈、实测角度随小目标变化、停止发布后输出归零/失能，以及断开任一电机反馈后两轴停止。目前这些硬件步骤尚未执行。

本次控制链路修复不等于完成 RViz 目标标注：仍缺少检测结果到 `/cat_target_camera` 的实际发布接入，相机安装偏移、TF 方向和时间同步也需实机验证。

## 软件验证

上位机协议及 ROS 回调模拟回归：

```bash
python3 -m unittest discover -s tests -p test_gimbal_usb.py -v
```

下位机完整 ARM 编译和运行实际 C 代码的控制链路模拟，参见下位机 `tests/README.md`。模拟替换外设和时钟，不能验证电气接线、电机固件、控制稳定性或真实 RTOS 时序。
