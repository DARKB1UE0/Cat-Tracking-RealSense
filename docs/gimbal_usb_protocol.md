# 云台 USB 控制协议

大疆 C 板以 USB CDC 虚拟串口连接上位机。串口波特率字段仅用于枚举兼容，实际 USB 链路无波特率限制。

帧格式（小端）：`AA 55 | version:u8 | type:u8 | seq:u8 | length:u16 | payload | crc16:u16`。CRC16 为 IBM/Modbus（初值 `FFFF`，多项式 `A001`），覆盖 `version` 到 payload。

`0x01` 设定点 payload 为 `<ffBBH>`：yaw、pitch（度，float32）、enable/mode（u8）、flags（u8）、保留（u16）。`0x02` 使能，`0x03` 急停，`0x04` ping；`0x81` 为状态帧，payload `<ffBBHI>`：当前 yaw/pitch、enabled、fault、保留、指令超时毫秒。

固件超过 100 ms 未收到设定帧会自动进入停止状态。Python 双端接口见 `tools/gimbal_usb.py`。
