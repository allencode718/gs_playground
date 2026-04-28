# GSP Render Dev

语言：[English](README.md) | 简体中文

这是 live replay demo 和 batch rendering benchmark 的统一 MotrixSim 3D Gaussian 渲染工作目录。

以下命令均在当前目录执行。

## 安装

```bash
UV_CACHE_DIR=.uv-cache uv sync --reinstall-package motrixsim-core
```

## Live Demo

```bash
UV_CACHE_DIR=.uv-cache uv run python demo/live_demo/replay.py
```

## Benchmark Notebook

```bash
UV_CACHE_DIR=.uv-cache uv run jupyter nbconvert \
  --to notebook \
  --execute benchmark/mtx_batch_minimal.ipynb \
  --ExecutePreprocessor.cwd=benchmark \
  --output mtx_batch_minimal.executed.ipynb
```

## Jupyter Kernel

```bash
UV_CACHE_DIR=.uv-cache uv run python -m ipykernel install \
  --user \
  --name gsp-render-dev \
  --display-name "gsp-render-dev"
```

## 依赖

- Python `>=3.10,<3.11`
- `motrixsim_core==0.7.1.dev97295`
- `torch==2.4.1+cu121`
- `gaussian_renderer==0.2.0`
- `gsplat==1.5.3`
- 完整依赖见 `pyproject.toml`。

## 目录说明

- `demo/live_demo/`：live replay demo、demo 资产和 replay 数据
- `benchmark/`：batch rendering notebook、benchmark 资产、辅助脚本和输出
- `pyproject.toml`：两个 demo 共用的统一 uv 环境
- `.uv-cache/`：上面命令会使用的本地 uv 缓存目录
- `.venv/`：执行 `uv sync` 后生成的本地虚拟环境
