# GS-Playground

语言：[English](README.md) | 简体中文

<p align="center">
  <strong>GS-Playground: A High-Throughput Photorealistic Simulator for Vision-Informed Robot Learning</strong>
</p>

<p align="center">
  <a href="https://gsplayground.github.io"><img src="https://img.shields.io/badge/project-page-brightgreen" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/paper-arXiv-red" alt="arXiv"></a>
  <a href="https://huggingface.co/gsplayground"><img src="https://img.shields.io/badge/assets-HuggingFace-yellow" alt="Hugging Face"></a>
  <img src="https://img.shields.io/badge/RSS-2026-blueviolet" alt="RSS 2026">
</p>

<p align="center">
  <strong>🎉 已被 RSS 2026 接收 🎉</strong>
</p>

<p align="center">
  <img src="media/teaser.png" alt="GS-Playground teaser" width="95%">
</p>

GS-Playground 是面向视觉机器人学习的高吞吐、高保真仿真框架。系统将并行机器人物理引擎与批量 3D Gaussian Splatting (3DGS) 渲染结合，为视觉强化学习、导航、操作和运动控制提供接近真实外观的观测、刚体同步的视觉资产，以及可用于训练的 sim-ready 场景。

当前仓库是早期公开预览版本，只包含一个最小 batch rendering benchmark 和两个最小 demo。完整 simulator、资产、数据集、训练代码和论文实验复现脚本会分阶段发布。

## 📰 新闻

- **2026-04-28:** GS-Playground 已被 **RSS 2026** 接收。
- **2026-04-29:** arXiv badge/link 已占位，预印本公开后会更新。

## ✨ 亮点

- **高保真视觉仿真：** 使用批量 3DGS 渲染为机器人学习环路提供 RGB 和 depth 观测。
- **高吞吐感知：** 论文中报告系统在 `640 x 480` 分辨率下通过批量渲染和内存高效 3DGS 资产达到最高 `10^4` FPS。
- **Rigid-Link Gaussian Kinematics：** 将 3DGS 点云簇绑定到仿真刚体，保证机器人和物体运动时的时间一致性。
- **并行物理引擎：** 基于 velocity-impulse formulation 的稳定接触求解器，支持 contact-rich 任务和大步长仿真。
- **Real2Sim 资产流程：** 从真实采集生成照片级、物理一致、内存高效的仿真资产。
- **多构型机器人覆盖：** 论文实验覆盖 locomotion、navigation 和 manipulation，包括四足、人形和机械臂。

## 📦 当前 Release

当前仓库保持轻量，主要用于提前开放渲染接口、demo 资产和最小示例：

- [x] `benchmark/`：最小 batch rendering notebook 和辅助脚本。
- [x] `demo/live_demo/`：最小 replay demo，包含 Franka/Robotiq 本地资产和 replay 数据。
- [x] `demo/navigation/`：最小机器人导航 demo，包含 Go1、Go2、G1 策略资产。
- [x] `pyproject.toml` 和 `uv.lock`：当前 benchmark 和 demo 共用的环境定义。

大规模训练流水线、完整 benchmark、生成式 3DGS 资产集合、Real2Sim 工具和论文实验配置尚未包含在当前预览版本中。

## 🛠️ 安装

以下命令均在仓库根目录执行。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/discoverse-dev/gs_playground.git
cd gs_playground
UV_CACHE_DIR=.uv-cache uv sync --reinstall-package motrixsim-core
```

依赖版本和平台标记由 `pyproject.toml` 与 `uv.lock` 维护。

## 🚀 快速开始

### Live Replay Demo

```bash
UV_CACHE_DIR=.uv-cache uv run python demo/live_demo/replay.py
```

### Navigation Demo

```bash
UV_CACHE_DIR=.uv-cache uv run python demo/navigation/robot_locomotion.py --robot go2 --scene nav_scene_1/mjcf/scene.xml --gs_ply nav_scene_1/3dgs/point_cloud.ply
```

### Batch Rendering Benchmark

```bash
UV_CACHE_DIR=.uv-cache uv run jupyter nbconvert \
  --to notebook \
  --execute benchmark/mtx_batch_minimal.ipynb \
  --ExecutePreprocessor.cwd=benchmark \
  --output mtx_batch_minimal.executed.ipynb
```

### 可选 Jupyter Kernel

```bash
UV_CACHE_DIR=.uv-cache uv run python -m ipykernel install \
  --user \
  --name gsp-render-dev \
  --display-name "gsp-render-dev"
```

## 🗺️ Release 计划

论文系统比当前预览仓库更完整，后续计划发布：

- [x] 最小 batch rendering benchmark。
- [x] 最小 live replay demo。
- [x] 最小 navigation demo。
- [x] README teaser 和项目链接。
- [ ] 预印本在 **2026-04-29** 公开后更新 arXiv 链接。
- [ ] arXiv 记录稳定后补充 citation metadata。
- [ ] 核心 simulator API：batched robot simulation、同步 3DGS 观测、RGB/depth camera、contact 和 MJCF 兼容资产接口。
- [ ] Batch 3DGS renderer：优化渲染 kernel、剪枝工具、内存高效资产加载、多场景 batch 示例。
- [ ] Real2Sim 工具：场景/物体分割、inpainting、3DGS/mesh 重建、位姿对齐、碰撞同步和资产打包。
- [ ] 传感器模块：depth、contact 和 batch LiDAR 示例。
- [ ] 训练代码：locomotion、视觉导航、manipulation 的 PPO 和视觉策略训练脚本。
- [ ] Benchmark suite：RSS 2026 论文中的 visual fidelity、rendering throughput、physics stability、locomotion、navigation 和 manipulation 复现实验。
- [ ] Hugging Face 发布：压缩 3DGS 资产、示例场景、机器人资产、训练策略和评测轨迹。

## 🔗 相关项目

GS-Playground 建立在我们生态中的多个组件和前序系统之上。它们目前尚未完整整合进这个预览仓库；后续 release 会将 RSS 2026 论文中涉及的物理、渲染、传感和学习接口逐步统一到 GS-Playground 工作流中。

- **物理仿真器：** [MotrixSim](https://github.com/Motphys/motrixsim-docs) 是高吞吐、接触丰富机器人仿真栈背后的物理后端。
- **State-based RL：** [MotrixLab](https://github.com/Motphys/MotrixLab) 包含 state-based reinforcement learning 基础设施，后续会与 GS-Playground 训练流水线连接。
- **RLGK 渲染：** [GaussianRenderer](https://github.com/discoverse-dev/GaussianRenderer) 包含与 Rigid-Link Gaussian Kinematics 相关的 Gaussian rendering 组件。
- **Batch LiDAR：** [MuJoCo-LiDAR](https://github.com/discoverse-dev/MuJoCo-LiDAR) 是我们此前的 batch LiDAR 模块；GS-Playground 的传感器套件会沿着这条工作整合到 navigation 和 locomotion 任务中。
- **上一代平台：** [DISCOVERSE](https://github.com/discoverse-dev/discoverse/) 是我们此前的具身仿真平台。GS-Playground 可以看作 DISCOVERSE 的下一代高保真、高吞吐版本。

## 📚 Citation

```bibtex

```
