# Omni-Tuner

**Omni-Tuner: Parameter-Efficient Adaptation for Source-Free Few-Shot Object Detection in Remote Sensing**

<p align="center">
  <img src="img/1.jpg" alt="Omni-Tuner main figure" width="450">
</p>

## 📌 项目简介

本项目实现了遥感场景下的 **Source-Free Few-Shot Domain Adaptive Object Detection (SF-FSDAOD)**。
在 Swin-Large 骨干上，仅需 **<3% 的可训练参数**，即可在目标域 **3 分钟以内完成适配**，且性能显著优于全参数微调。

## ✨ 亮点

- 面向遥感目标检测的 SF-FSDAOD
- 基于 **MMDetection** 的实现
- Swin-Large 上 <3% 参数高效适配
- 目标域 3 分钟内完成适配，性能远超全调

## 🧩 环境配置

本项目依托 MMDetection 框架，环境配置请**严格参考** [`env/begin.md`](env/begin.md)。

> 说明：该文件包含与 MMDetection/MMCV 版本匹配的完整环境安装步骤。

## 🗂️ 数据集准备

数据集统一放置在 `dataset` 目录，实验所用 **6 个数据集已注册**。

目录结构示例：

```
dataset
├── xview3c
│   ├── VOC2007
│   ├── ├── JPEGImages
│   ├── ├── Annotations
│   ├── └── ImageSets
├── hrrsd1c
├── ssdd1c_fs
└── ...
```

## 🚀 快速开始（以 xView → DOTA 为例）

实验快捷启动脚本已存放在 `tool_sh` 中。

### 1) 运行源域训练

```bash
bash tool_sh\train_source_xview.sh
```

### 2) 源域信息离线抽取

```bash
bash tool_sh\extract_xview_all_in_one.sh
```

### 3) 目标域微调

```bash
bash tool_sh\run_target_dota.sh
```

## 📁 主要目录说明

- `Omni_Tuner_configs/`：Omni-Tuner 相关配置
- `prototype_tools/`：原型与统计信息抽取工具
- `tool_sh/`：实验一键启动脚本
- `dataset/`：数据集目录

## 📄 许可证

本项目采用 [MIT License](LICENSE)。

## 🙏 致谢

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [MONA](https://github.com/Leiyi-HU/mona)
