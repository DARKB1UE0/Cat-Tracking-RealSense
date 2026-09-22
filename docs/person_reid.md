# 人物重识别模型

人物模式使用 Intel Open Model Zoo 的 `person-reidentification-retail-0287`。它是面向人体外观匹配的 OmniScaleNet + Linear Context Transform 模型，输出 256 维特征，通过归一化余弦相似度比较参考人物与实时检测框。

- [官方模型说明](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/person-reidentification-retail-0287/README.md)
- [官方模型文件清单、大小及 SHA-384](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/person-reidentification-retail-0287/model.yml)
- [Apache-2.0 许可](https://github.com/openvinotoolkit/open_model_zoo/blob/master/LICENSE)
- 固定下载源：`https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-reidentification-retail-0287/FP32/`
- FP32 文件：`person-reidentification-retail-0287.xml`（451147 字节）及 `.bin`（2362128 字节）。校验值固定在 `person_reid.py`，不从运行时网络响应取得。

权重下载至用户缓存，不纳入 Git，不上传参考照片。下载失败或校验不通过时初始化报错；缓存完好时不需要网络。使用 `python3 person_reid.py` 下载并执行一次 CPU 推理自检。依赖固定为 OpenVINO 2024.6.0，本机 Python 3.10 / x86_64 已验证；其他架构需先确认运行时支持。

输入为完整人体框，BGR、float32、0–255、NCHW `[1,3,256,128]`；IR 已含归一化，不重复使用 ImageNet 预处理。辅助颜色项仍只取衣着区域。人物不加载猫用的 ResNet50，也不使用同一套特征。

网页综合分为 `0.90 * ReID + 0.10 * Color`，负的 ReID 余弦相似度按 0 显示。综合阈值 0.15、与次高分差至少 0.08、连续 3 次空间相邻检测确认；ReID 和 Color 均须数值有效，两项均无单独最低分。颜色或特征输出非法、推理异常、候选丢失均不能确认目标。异常会重置连续确认计数，后续必须重新积累。

这些分数不是身份概率，旧 ResNet 的阈值经验不能直接套用。0.15 按用户要求设置，仍需在现场用同一人的不同角度及其他人验证区分效果。参考照应为当前衣着的完整单人照；换衣、相似服装、强遮挡和极端姿态仍可能失败。自动跟随保持 1 米停车距离。识别闪烁时，以最后确认目标的采集时间计：0.5 秒内保留已有导航但不新发目标，超过后请求暂停导航；之后持续等待、不会因目标丢失时间过长退出；重获确认目标可继续，需等旧导航取消完成。无目标时也可先启用等待。手动停止、模式切换、页面失联或设备/导航故障仍会结束，不自动恢复。
