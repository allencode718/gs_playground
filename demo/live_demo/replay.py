from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
from typing import Dict, Optional, Tuple

import motrixsim as mx
import numpy as np
import torch
from gaussian_renderer import BatchSplatConfig, MtxBatchSplatRenderer
from motrixsim import SceneData, forward_kinematic
from motrixsim.render import RenderApp

DEMO_DIR = Path(__file__).resolve().parent
ASSETS_DIR = DEMO_DIR / "assets"
ROBOT_DIR = ASSETS_DIR / "models" / "robots" / "manipulation" / "franka_emika_panda_robotiq"
TASK_DIR = ASSETS_DIR / "models" / "tasks" / "table30" / "_04_hang_toothbrush_cup"

MODEL_XML = ROBOT_DIR / "xmls" / "table30_04_hang_toothbrush_cup.xml"
BACKGROUND_PLY = ROBOT_DIR / "3dgs" / "background_085.ply"

SCENE_Z_OFFSET = 0.85
LINK0_POS = "0,0,0.85"
BASE_POS = "0,0,0.924244"


def set_asset_root(asset_dir: Path) -> None:
    global ASSETS_DIR, ROBOT_DIR, TASK_DIR, MODEL_XML, BACKGROUND_PLY
    ASSETS_DIR = asset_dir.resolve()
    ROBOT_DIR = ASSETS_DIR / "models" / "robots" / "manipulation" / "franka_emika_panda_robotiq"
    TASK_DIR = ASSETS_DIR / "models" / "tasks" / "table30" / "_04_hang_toothbrush_cup"
    MODEL_XML = ROBOT_DIR / "xmls" / "table30_04_hang_toothbrush_cup.xml"
    BACKGROUND_PLY = ROBOT_DIR / "3dgs" / "background_085.ply"


class ProfileStats:
    def __init__(self) -> None:
        self.samples: Dict[str, list[float]] = {}

    def add(self, name: str, seconds: float) -> None:
        self.samples.setdefault(name, []).append(float(seconds))

    def report(self, frames: int) -> None:
        total = sum(self.samples.get("frame_total", []))
        fps = float(frames) / total if total > 0.0 else 0.0
        print(f"[profile] frames={frames} total={total:.3f}s fps={fps:.2f}", flush=True)
        for name, values in self.samples.items():
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            print(
                f"[profile] {name}: mean={arr.mean() * 1000.0:.2f}ms "
                f"p95={np.percentile(arr, 95) * 1000.0:.2f}ms max={arr.max() * 1000.0:.2f}ms",
                flush=True,
            )


def parse_xyz_csv(csv_text: str) -> Tuple[float, float, float]:
    parts = [float(x.strip()) for x in str(csv_text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected x,y,z, got: {csv_text!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def add_z(xyz, z_offset: float) -> np.ndarray:
    out = np.asarray(xyz, dtype=np.float32).copy()
    out[2] += float(z_offset)
    return out


def quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float32,
    )


def matrix_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q.astype(np.float32)


def find_root_body_by_name(world, name: str):
    for body in world.hierarchy.bodies:
        if getattr(body.link, "name", None) == name:
            return body
    return None


def ensure_pedestal(world) -> None:
    if find_root_body_by_name(world, "base") is not None:
        return

    base_xml = """<mujoco model="demo_base">
  <worldbody>
    <body name="base" pos="-0.07 0 0.924">
      <body name="controller_box" pos="0 0 0">
        <inertial diaginertia="1.71363 1.27988 0.809981" mass="46.64" pos="-0.325 0 -0.38"/>
        <geom name="controller_box_col" type="box" pos="-0.325 0 -0.38" size="0.11 0.2 0.265" rgba="0.2 0.2 0.2 1" group="4"/>
      </body>
      <body name="pedestal_feet" pos="0 0 0">
        <inertial diaginertia="8.16095 9.59375 15.0785" mass="167.09" pos="-0.1225 0 -0.758"/>
        <geom name="pedestal_feet_col" type="box" pos="-0.1225 0 -0.758" size="0.385 0.35 0.155" rgba="0.2 0.2 0.2 1" group="4"/>
      </body>
      <body name="torso" pos="0 0 0">
        <inertial diaginertia="1e-08 1e-08 1e-08" mass="0.0001" pos="0 0 0"/>
        <geom name="torso_vis" type="box" pos="0 0 -0.05" size="0.05 0.05 0.05" rgba="0.2 0.2 0.2 1" group="4" conaffinity="0" contype="0"/>
      </body>
      <body name="pedestal" pos="0 0 0">
        <inertial diaginertia="6.0869 5.81635 4.20915" mass="60.864" pos="0 0 0" quat="0.659267 -0.259505 -0.260945 0.655692"/>
        <geom name="pedestal_col" type="cylinder" pos="-0.02 0 -0.29" size="0.10 0.31" rgba="0.2 0.2 0.2 1" group="4"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""
    world.attach(mx.msd.from_str(base_xml))


def ensure_table_legs(world) -> None:
    if find_root_body_by_name(world, "table_legs") is not None:
        return

    task_table_geom = None
    for geom in world.hierarchy.geoms:
        if getattr(geom, "name", None) == "table":
            task_table_geom = geom
            break
    if task_table_geom is None:
        return

    pos = np.asarray(task_table_geom.position, dtype=np.float32)
    quat_xyzw = np.asarray(task_table_geom.orientation, dtype=np.float32)
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    table_xml = f"""<mujoco model="demo_table_legs">
  <worldbody>
    <body name="table_legs" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" quat="{quat_wxyz[0]:.8f} {quat_wxyz[1]:.8f} {quat_wxyz[2]:.8f} {quat_wxyz[3]:.8f}">
      <geom name="table_leg_xp_yp" type="box" size="0.025 0.025 0.45" pos="0.295 0.49 -0.45" contype="1" conaffinity="1" rgba="0.45 0.45 0.45 1"/>
      <geom name="table_leg_xp_ym" type="box" size="0.025 0.025 0.45" pos="0.295 -0.49 -0.45" contype="1" conaffinity="1" rgba="0.45 0.45 0.45 1"/>
      <geom name="table_leg_xm_yp" type="box" size="0.025 0.025 0.45" pos="-0.295 0.49 -0.45" contype="1" conaffinity="1" rgba="0.45 0.45 0.45 1"/>
      <geom name="table_leg_xm_ym" type="box" size="0.025 0.025 0.45" pos="-0.295 -0.49 -0.45" contype="1" conaffinity="1" rgba="0.45 0.45 0.45 1"/>
    </body>
  </worldbody>
</mujoco>"""
    world.attach(mx.msd.from_str(table_xml))


def apply_scene_overrides(world) -> None:
    ensure_pedestal(world)
    ensure_table_legs(world)

    base_body = find_root_body_by_name(world, "base")
    if base_body is not None:
        base_body.link.local_translation = np.asarray(parse_xyz_csv(BASE_POS), dtype=np.float32)

    link0_body = find_root_body_by_name(world, "link0")
    if link0_body is None:
        raise RuntimeError("Could not find root body 'link0' in demo scene")
    link0_body.link.local_translation = np.asarray(parse_xyz_csv(LINK0_POS), dtype=np.float32)

    for camera in world.hierarchy.cameras:
        camera.position = add_z(camera.position, SCENE_Z_OFFSET)
    for light in world.hierarchy.lights:
        light.position = add_z(light.position, SCENE_Z_OFFSET)
    for geom in world.hierarchy.geoms:
        if getattr(geom, "name", None) != "floor":
            geom.position = add_z(geom.position, SCENE_Z_OFFSET)

    keep_names = {"base", "link0", "toothbrush_cup"}
    for body in world.hierarchy.bodies:
        if getattr(body.link, "name", None) in keep_names:
            continue
        body.link.local_translation = add_z(body.link.local_translation, SCENE_Z_OFFSET)

def build_frustum_mjcf(cam_pos, cam_x, cam_y, cam_fwd, fovy_deg, dist, aspect, tex_w, tex_h) -> str:
    half_h = dist * np.tan(np.deg2rad(fovy_deg) * 0.5)
    half_w = half_h * aspect
    center = cam_pos + cam_fwd * dist
    c0 = center + (-cam_x * half_w) + (cam_y * half_h)
    c1 = center + (cam_x * half_w) + (cam_y * half_h)
    c2 = center + (cam_x * half_w) + (-cam_y * half_h)
    c3 = center + (-cam_x * half_w) + (-cam_y * half_h)
    q = matrix_to_quat_wxyz(np.column_stack([-cam_x, cam_y, cam_fwd]))
    quat_wxyz = f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}"

    def v3(a) -> str:
        return f"{a[0]:.6f} {a[1]:.6f} {a[2]:.6f}"

    edge_radius = 0.002
    return f"""<mujoco>
  <asset>
    <texture name="gs_screen_tex" type="2d" builtin="dynamic" width="{tex_w}" height="{tex_h}" _perinstance="true"/>
    <material name="gs_screen_mat" texture="gs_screen_tex" emission="1 1 1 1" castshadow="false"/>
    <material name="frustum_edge_mat" rgba="1 1 1 1" emission="0.6 0.6 0.6 1" castshadow="false"/>
  </asset>
  <worldbody>
    <geom name="gs_screen" type="box" size="{half_w:.6f} {half_h:.6f} 0.005" pos="{v3(center)}" quat="{quat_wxyz}" material="gs_screen_mat" contype="0" conaffinity="0"/>
    <geom type="capsule" size="{edge_radius}" fromto="{v3(cam_pos)} {v3(c0)}" material="frustum_edge_mat" contype="0" conaffinity="0"/>
    <geom type="capsule" size="{edge_radius}" fromto="{v3(cam_pos)} {v3(c1)}" material="frustum_edge_mat" contype="0" conaffinity="0"/>
    <geom type="capsule" size="{edge_radius}" fromto="{v3(cam_pos)} {v3(c2)}" material="frustum_edge_mat" contype="0" conaffinity="0"/>
    <geom type="capsule" size="{edge_radius}" fromto="{v3(cam_pos)} {v3(c3)}" material="frustum_edge_mat" contype="0" conaffinity="0"/>
    <geom type="capsule" size="{edge_radius}" fromto="{v3(c0)} {v3(c1)}" material="frustum_edge_mat" contype="0" conaffinity="0"/>
    <geom type="capsule" size="{edge_radius}" fromto="{v3(c1)} {v3(c2)}" material="frustum_edge_mat" contype="0" conaffinity="0"/>
    <geom type="capsule" size="{edge_radius}" fromto="{v3(c2)} {v3(c3)}" material="frustum_edge_mat" contype="0" conaffinity="0"/>
    <geom type="capsule" size="{edge_radius}" fromto="{v3(c3)} {v3(c0)}" material="frustum_edge_mat" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>"""


def load_demo_model(gs_cam_id: int, gs_w: int, gs_h: int, frustum_dist: float):
    scene = mx.msd.from_file(MODEL_XML.as_posix())
    apply_scene_overrides(scene)
    camera = scene.hierarchy.cameras[int(gs_cam_id)]
    cam_pos = np.asarray(camera.position, dtype=np.float32)
    cam_rot = quat_xyzw_to_matrix(np.asarray(camera.orientation, dtype=np.float32))
    frustum = mx.msd.from_str(
        build_frustum_mjcf(
            cam_pos=cam_pos,
            cam_x=cam_rot[:, 0],
            cam_y=cam_rot[:, 1],
            cam_fwd=-cam_rot[:, 2],
            fovy_deg=float(camera.fovy),
            dist=float(frustum_dist),
            aspect=float(gs_w) / float(gs_h),
            tex_w=int(gs_w),
            tex_h=int(gs_h),
        )
    )
    scene.attach(frustum)
    return scene.build()


def build_gs_renderer(model, batch_size: int) -> MtxBatchSplatRenderer:
    gaussians: Dict[str, str] = {
        "link1": (ROBOT_DIR / "3dgs" / "franka" / "link1.ply").as_posix(),
        "link2": (ROBOT_DIR / "3dgs" / "franka" / "link2.ply").as_posix(),
        "link3": (ROBOT_DIR / "3dgs" / "franka" / "link3.ply").as_posix(),
        "link4": (ROBOT_DIR / "3dgs" / "franka" / "link4.ply").as_posix(),
        "link5": (ROBOT_DIR / "3dgs" / "franka" / "link5.ply").as_posix(),
        "link6": (ROBOT_DIR / "3dgs" / "franka" / "link6.ply").as_posix(),
        "link7": (ROBOT_DIR / "3dgs" / "franka" / "link7.ply").as_posix(),
        "robotiq_base": (ROBOT_DIR / "3dgs" / "robotiq" / "robotiq_base.ply").as_posix(),
        "left_driver": (ROBOT_DIR / "3dgs" / "robotiq" / "left_driver.ply").as_posix(),
        "left_coupler": (ROBOT_DIR / "3dgs" / "robotiq" / "left_coupler.ply").as_posix(),
        "left_spring_link": (ROBOT_DIR / "3dgs" / "robotiq" / "left_spring_link.ply").as_posix(),
        "left_follower": (ROBOT_DIR / "3dgs" / "robotiq" / "left_follower.ply").as_posix(),
        "right_driver": (ROBOT_DIR / "3dgs" / "robotiq" / "right_driver.ply").as_posix(),
        "right_coupler": (ROBOT_DIR / "3dgs" / "robotiq" / "right_coupler.ply").as_posix(),
        "right_spring_link": (ROBOT_DIR / "3dgs" / "robotiq" / "right_spring_link.ply").as_posix(),
        "right_follower": (ROBOT_DIR / "3dgs" / "robotiq" / "right_follower.ply").as_posix(),
        "toothbrush_cup": (TASK_DIR / "3dgs" / "toothbrush_cup.ply").as_posix(),
        "rack": (TASK_DIR / "3dgs" / "rack.ply").as_posix(),
    }
    return MtxBatchSplatRenderer(BatchSplatConfig(body_gaussians=gaussians, background_ply=None, minibatch=int(batch_size)), model)


def build_background_renderer(model, batch_size: int) -> MtxBatchSplatRenderer:
    return MtxBatchSplatRenderer(BatchSplatConfig(body_gaussians={}, background_ply=BACKGROUND_PLY.as_posix(), minibatch=int(batch_size)), model)


def get_camera_pose(model, data, cam_id: int) -> Tuple[np.ndarray, np.ndarray, float]:
    cam = model.cameras[int(cam_id)]
    pose = np.asarray(cam.get_pose(data), dtype=np.float32)
    if pose.ndim == 1:
        pose = pose[None, :]
    if pose.ndim >= 3:
        pose = pose.reshape(pose.shape[0], -1, pose.shape[-1])[:, 0, :]
    pos = pose[:, :3].astype(np.float32)
    quat = pose[:, 3:7].astype(np.float32)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12
    return pos, quat, float(getattr(cam, "fovy", 45.0))


def render_gs_frame(gs_renderer, bg_renderer, bg_imgs, model, data, cam_id: int, height: int, width: int):
    forward_kinematic(model, data)
    link_poses = model.get_link_poses(data)
    body_pos = link_poses[..., :3]
    body_quat = link_poses[..., 3:7]
    cam_pos, cam_quat, fovy = get_camera_pose(model, data, cam_id)
    cam_xmat = np.stack([quat_xyzw_to_matrix(q) for q in cam_quat], axis=0)
    device = gs_renderer.device
    cam_pos_t = torch.from_numpy(cam_pos[:, None, :]).to(device=device, dtype=torch.float32)
    cam_xmat_t = torch.from_numpy(cam_xmat[:, None, :, :]).to(device=device, dtype=torch.float32)
    fovy_np = np.full((cam_pos.shape[0], 1), fovy, dtype=np.float32)

    if bg_imgs is None:
        bg_gsb = bg_renderer.batch_update_gaussians(body_pos, body_quat)
        bg_imgs, _ = bg_renderer.batch_env_render(bg_gsb, cam_pos_t, cam_xmat_t, int(height), int(width), fovy_np)

    gsb = gs_renderer.batch_update_gaussians(body_pos, body_quat)
    rgb_t, _ = gs_renderer.batch_env_render(gsb, cam_pos_t, cam_xmat_t, int(height), int(width), fovy_np, bg_imgs=bg_imgs)
    rgb = rgb_t.detach().cpu().numpy() if isinstance(rgb_t, torch.Tensor) else np.asarray(rgb_t)
    if rgb.ndim == 5 and rgb.shape[2] == 3:
        rgb = np.transpose(rgb, (0, 1, 3, 4, 2))
    if rgb.ndim == 5 and rgb.shape[1] == 1:
        rgb = rgb.squeeze(1)
    if rgb.ndim == 4 and rgb.shape[1] == 3:
        rgb = np.transpose(rgb, (0, 2, 3, 1))
    if rgb.ndim == 5:
        rgb = rgb[:, 0, ...]
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), bg_imgs



def make_grid_offsets(batch_size: int, spacing: float = 2.0):
    cols = int(np.ceil(np.sqrt(batch_size)))
    rows = int(np.ceil(batch_size / cols))
    sx = 1.25 * float(spacing)
    sy = float(spacing)
    x0 = 0.5 * (cols - 1) * sx
    y0 = 0.5 * (rows - 1) * sy
    return [[(i % cols) * sx - x0, y0 - (i // cols) * sy, 0.0] for i in range(batch_size)]


def normalize_full_qpos(qpos: np.ndarray, cup_xy_offset: str) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32).copy()
    if qpos.shape[-1] < 20:
        raise RuntimeError(
            f"Replay qpos has {qpos.shape[-1]} values, but this demo needs full Motrix dof_pos with at least 20 values. "
            "Regenerate replay data with dof_pos saved in replay/ep_*.npz."
        )
    offset = np.asarray([float(v.strip()) for v in cup_xy_offset.split(",")], dtype=np.float32)
    if offset.shape != (2,):
        raise ValueError("--cup_xy_offset must be dx,dy")
    qpos[..., 13:15] += offset.reshape((1,) * (qpos.ndim - 1) + (2,))
    qpos[..., 15] += SCENE_Z_OFFSET
    quat = qpos[..., 16:20]
    quat_norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    qpos[..., 16:20] = quat / np.maximum(quat_norm, 1e-8)
    return qpos


def load_replay_npz(path: Path) -> np.ndarray:
    pack = np.load(path.as_posix())
    if "dof_pos" not in pack:
        raise RuntimeError(f"{path} does not contain dof_pos. Regenerate the replay data.")
    return pack["dof_pos"].astype(np.float32)


def load_replay_data_dir(data_dir: Path, batch_size: int, cup_xy_offset: str) -> tuple[np.ndarray, int]:
    replay_files = sorted((data_dir / "replay").glob("ep_*.npz"))
    if not replay_files:
        raise RuntimeError(f"{data_dir}/replay has no replay npz files")
    episodes = [load_replay_npz(replay_files[i % len(replay_files)]) for i in range(batch_size)]
    min_len = min(ep.shape[0] for ep in episodes)
    qpos = np.stack([ep[:min_len] for ep in episodes], axis=1).astype(np.float32)  # (T,B,nq)
    return normalize_full_qpos(qpos, cup_xy_offset), 30


def drain_capture_tasks(capture_tasks, capture_dir: Path) -> None:
    while capture_tasks:
        idx, task = capture_tasks[0]
        if task.state == "pending":
            break
        capture_tasks.popleft()
        try:
            img = task.take_image()
            if img is None:
                continue
            capture_dir.mkdir(parents=True, exist_ok=True)
            out_path = capture_dir / f"capture_{idx:06d}.png"
            img.save_to_disk(out_path.as_posix())
            print(f"Captured image: {out_path}", flush=True)
        except Exception as exc:
            print(f"Error saving capture {idx}: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimum 04 draw live demo")
    parser.add_argument("--replay_data_dir", type=str, default="replay_data_16")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gs_cam_id", type=int, default=0)
    parser.add_argument("--gs_w", type=int, default=320)
    parser.add_argument("--gs_h", type=int, default=240)
    parser.add_argument("--frustum_dist", type=float, default=0.5)
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument("--cup_xy_offset", type=str, default="0,0", help="Small cup XY correction, format: dx,dy")
    parser.add_argument("--profile_frames", type=int, default=0, help="Run N timed frames, print FPS, then exit")
    parser.add_argument("--assets_dir", type=str, default="assets", help="Asset root directory")
    parser.add_argument("--capture_dir", type=str, default="shot", help="Directory for system camera screenshots")
    parser.add_argument("--capture_every_frame", action="store_true", help="Save the render system camera every replay frame")
    parser.add_argument("--capture_frames", type=int, default=0, help="Stop after saving N automatic captures; 0 means no limit")
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir)
    if not assets_dir.is_absolute():
        assets_dir = DEMO_DIR / assets_dir
    set_asset_root(assets_dir)

    replay_data_dir = Path(args.replay_data_dir)
    if not replay_data_dir.is_absolute():
        replay_data_dir = DEMO_DIR / replay_data_dir

    if not replay_data_dir.exists():
        raise RuntimeError(f"Replay data directory does not exist: {replay_data_dir}")
    qpos, fps = load_replay_data_dir(replay_data_dir, int(args.batch_size), args.cup_xy_offset)

    total_steps, batch_size, _ = qpos.shape
    model = load_demo_model(args.gs_cam_id, args.gs_w, args.gs_h, args.frustum_dist)
    data = SceneData(model, batch=(batch_size,))
    data.set_dof_pos(qpos[0], model)
    forward_kinematic(model, data)

    gs_renderer = build_gs_renderer(model, batch_size=batch_size)
    bg_renderer = build_background_renderer(model, batch_size=batch_size)
    bg_imgs: Optional[object] = None

    print(
        f"Preparing first GS frame for batch={batch_size}... first launch can take about 1 minute while gsplat loads CUDA kernels.",
        flush=True,
    )
    enable_profile = int(args.profile_frames) > 0
    t0 = time.perf_counter()
    first_gs_u8, bg_imgs = render_gs_frame(
        gs_renderer, bg_renderer, bg_imgs, model, data, args.gs_cam_id, args.gs_h, args.gs_w
    )
    first_frame_seconds = time.perf_counter() - t0
    if enable_profile:
        print(f"First GS frame prepared in {first_frame_seconds:.3f}s.", flush=True)

    print("Live demo is running. Close the MotrixSim window to exit.", flush=True)
    frame_id = 1 % total_steps
    frame_dt = 1.0 / max(1, fps)
    next_frame_at = time.perf_counter() + frame_dt
    capture_dir = Path(args.capture_dir)
    if not capture_dir.is_absolute():
        capture_dir = DEMO_DIR / capture_dir
    capture_tasks = deque()
    capture_index = 0
    saved_auto_captures = 0
    prof = ProfileStats()
    if enable_profile:
        prof.add("first_gs_frame", first_frame_seconds)
    profiled_frames = 0

    with RenderApp() as render:
        render.launch(model, batch=batch_size, render_offset=make_grid_offsets(batch_size, spacing=float(args.spacing)))
        render.sync(data)
        rcam = render.get_camera(0)
        gs_screen_img = render.get_texture_image("gs_screen_tex")
        gs_screen_img.pixels = np.ascontiguousarray(first_gs_u8[0] if batch_size == 1 else first_gs_u8)
        render.sync(data)
        if rcam is None:
            print("Warning: render camera 0 is unavailable; screenshots are disabled.", flush=True)
        while not render.is_closed:
            now = time.perf_counter()
            if rcam is not None and render.input.is_key_just_pressed("space"):
                capture_tasks.append((capture_index, rcam.capture()))
                capture_index += 1

            if enable_profile or now >= next_frame_at:
                frame_t0 = time.perf_counter()
                t = time.perf_counter()
                data.set_dof_pos(qpos[frame_id], model)
                if enable_profile:
                    prof.add("set_dof_pos", time.perf_counter() - t)

                t = time.perf_counter()
                gs_u8, bg_imgs = render_gs_frame(
                    gs_renderer, bg_renderer, bg_imgs, model, data, args.gs_cam_id, args.gs_h, args.gs_w
                )
                if enable_profile:
                    prof.add("render_gs_frame", time.perf_counter() - t)

                t = time.perf_counter()
                gs_screen_img.pixels = np.ascontiguousarray(gs_u8[0] if batch_size == 1 else gs_u8)
                if enable_profile:
                    prof.add("texture_pixels", time.perf_counter() - t)
                frame_id = (frame_id + 1) % total_steps
                next_frame_at = now + frame_dt

                t = time.perf_counter()
                render.sync(data)
                if rcam is not None and args.capture_every_frame:
                    capture_tasks.append((capture_index, rcam.capture()))
                    capture_index += 1
                    saved_auto_captures += 1
                if enable_profile:
                    prof.add("render_sync_after_update", time.perf_counter() - t)
                    prof.add("frame_total", time.perf_counter() - frame_t0)
                drain_capture_tasks(capture_tasks, capture_dir)
                profiled_frames += 1
                if args.capture_every_frame and args.capture_frames > 0 and saved_auto_captures >= int(args.capture_frames):
                    while capture_tasks:
                        render.sync(data)
                        drain_capture_tasks(capture_tasks, capture_dir)
                    break
                if enable_profile and profiled_frames >= int(args.profile_frames):
                    prof.report(profiled_frames)
                    break
            else:
                render.sync(data)
                drain_capture_tasks(capture_tasks, capture_dir)


if __name__ == "__main__":
    main()
