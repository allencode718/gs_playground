# GSP Render Dev

Languages: English | [简体中文](README_CN.md)

Unified MotrixSim 3D Gaussian rendering workspace for the live replay demo and batch rendering benchmark.

Run all commands from this directory.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/discoverse-dev/gs_playground.git
cd gs_playground
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

## Navigation Demo

```bash
UV_CACHE_DIR=.uv-cache uv run python demo/navigation/robot_locomotion.py --robot go2 --scene nav_scene_1/mjcf/scene.xml --gs_ply nav_scene_1/3dgs/point_cloud.ply
```

## Jupyter Kernel

```bash
UV_CACHE_DIR=.uv-cache uv run python -m ipykernel install \
  --user \
  --name gsp-render-dev \
  --display-name "gsp-render-dev"
```

## Dependencies

- Python `>=3.10,<3.11`
- `motrixsim_core==0.7.1.dev97295`
- `torch==2.4.1+cu121`
- `gaussian_renderer==0.2.0`
- `gsplat==1.5.3`
- `onnxruntime==1.22.1`
- See `pyproject.toml` for the full dependency list.

## Repository Map

- `demo/live_demo/`: interactive live replay demo, local assets, and replay data
- `demo/navigation/`: keyboard-controlled robot navigation demo, policies, scene assets, and robot assets
- `benchmark/`: batch rendering notebook, benchmark assets, helper scripts, and outputs
- `pyproject.toml`: unified uv environment for both demos
- `.uv-cache/`: optional local uv cache created by the commands above
- `.venv/`: local virtual environment created by `uv sync`
