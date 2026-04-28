import argparse
import json
import os
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent

os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["ALSOFT_DRIVERS"] = "null"
os.environ["ALSA_CONFIG_PATH"] = (DEMO_DIR / "configs" / "asound-null.conf").as_posix()

import numpy as np
from scipy.spatial.transform import Rotation
from motrixsim import SceneData, msd, run
from motrixsim.render import RenderApp, Layout
from utils.controller import KeyboardCommandAdapter
from utils.policy import G1LocomotionPolicy, Go1LocomotionPolicy, Go2LocomotionPolicy
from utils.robot import G1Robot, Go1Robot, Go2Robot

camera_positions = {"g1": [-1.5, 0, 1.0], "go1": [-2, 0, 0.5], "go2": [-2, 0, 0.5]}
G1_HEAD_CAMERA_POS = [0.09, 0.0, 0.39]

DEFAULT_SCENE_GAUSSIANS = {
    "scene": Path("nav_scene_1/3dgs/point_cloud.ply"),
}

ROBOT_GS_LINK_ALIASES = {
    "go2": {
        "FL_foot": "FL_calf",
        "FR_foot": "FR_calf",
        "RL_foot": "RL_calf",
        "RR_foot": "RR_calf",
    },
    "g1": {
        "head": "torso_link",
        "left_ankle_pitc_linkh": "left_ankle_pitch_link",
        "right_ankle_pitc_linkh": "right_ankle_pitch_link",
    },
}


def resolve_demo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else DEMO_DIR / path


def load_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    path = resolve_demo_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Navigation config must be a JSON object: {path}")
    return config


def collect_scene_gaussians(scene_gaussian_cfg: dict[str, str | Path]) -> dict[str, str]:
    gaussians = {}
    for gs_name, rel_path in scene_gaussian_cfg.items():
        ply = resolve_demo_path(rel_path)
        if ply.exists():
            gaussians[gs_name] = ply.as_posix()
    return gaussians


def scene_gaussian_paths(scene_gaussian_cfg: dict[str, str | Path]) -> set[Path]:
    return {resolve_demo_path(rel_path).resolve() for rel_path in scene_gaussian_cfg.values()}


def collect_robot_gaussians(robot_name: str, model, robot_gs_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    if not robot_gs_dir.exists():
        return {}, {}

    link_names = set(model.link_names)
    aliases = ROBOT_GS_LINK_ALIASES.get(robot_name, {})
    gaussians = {}
    gaussian_links = {}
    skipped = []
    for ply in sorted(robot_gs_dir.glob("*.ply")):
        gs_name = ply.stem
        link_name = gs_name if gs_name in link_names else aliases.get(gs_name)
        if link_name in link_names:
            gaussians[gs_name] = ply.as_posix()
            gaussian_links[gs_name] = link_name
        else:
            skipped.append(gs_name)

    if skipped:
        print(f"Skipping GS assets without matching links: {', '.join(skipped)}", flush=True)
    return gaussians, gaussian_links


def apply_gaussian_links(gs_renderer, model, gaussian_links: dict[str, str]) -> None:
    if not gaussian_links:
        return

    link_ids = {name: idx for idx, name in enumerate(model.link_names)}
    objects_info = []
    body_ids = []
    for gs_name, link_name in gaussian_links.items():
        if gs_name not in gs_renderer.gaussian_start_indices:
            continue
        if link_name not in link_ids:
            continue
        objects_info.append((gs_name, gs_renderer.gaussian_start_indices[gs_name], gs_renderer.gaussian_end_indices[gs_name]))
        body_ids.append(link_ids[link_name])

    if not objects_info:
        return

    gs_renderer.gs_idx_start = np.array([start for _, start, _ in objects_info])
    gs_renderer.gs_idx_end = np.array([end for _, _, end in objects_info])
    gs_renderer.gs_body_ids = np.array(body_ids)
    gs_renderer.set_objects_mapping(objects_info)


def apply_initial_qpos(data: SceneData, model, initial_qpos) -> None:
    if not initial_qpos:
        return
    qpos = np.asarray(initial_qpos, dtype=np.float32)
    if qpos.shape[0] != data.dof_pos.shape[0]:
        raise ValueError(f"initial_qpos has {qpos.shape[0]} values, expected {data.dof_pos.shape[0]}")
    data.set_dof_pos(qpos, model)
    data.set_dof_vel(np.zeros_like(data.dof_vel))


def find_camera_id(model, camera_name: str) -> int | None:
    for idx, camera in enumerate(model.cameras):
        if camera.name == camera_name or camera.name.endswith(camera_name):
            return idx
    return None


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None, help="Path to navigation JSON config")
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(description="Keyboard control for robots", parents=[config_parser])
    parser.add_argument("--robot", type=str, choices=["g1", "go1", "go2"], default="go2")
    parser.add_argument("--scene", type=str, default="plane", help="Path to scene XML file")
    parser.add_argument("--gs_ply", type=str, default="", help="Path to background gaussian splatting ply file")
    parser.add_argument("--assets_dir", type=str, default="assets", help="Navigation GS asset root directory")
    parser.add_argument("--robot_gs_dir", type=str, default="", help="Robot gaussian splatting asset directory")
    parser.add_argument("--no_robot_gs", action="store_true", help="Disable robot gaussian splatting assets")
    parser.add_argument("--no-sync", action="store_true", help="Disable real-time clock sync")
    parser.add_argument("--save_data", action="store_true", help="Enable data collection")
    parser.add_argument("--save_dir", type=str, default="./data/navigation", help="Data save directory")
    parser.add_argument("--prompt", type=str, default="Navigate in the scene", help="Task prompt")
    parser.add_argument("--debug_input", action="store_true", help="Print keyboard command changes")
    parser.add_argument("--auto_reset", action=argparse.BooleanOptionalAction, default=True, help="Reset robot when the policy detects a fall")
    parser.add_argument("--initial_qpos", type=float, nargs="*", default=None, help="Initial full dof_pos values")
    parser.set_defaults(
        robot=config.get("robot", "go2"),
        scene=config.get("scene", "plane"),
        gs_ply=config.get("gs_ply", ""),
        assets_dir=config.get("assets_dir", "assets"),
        robot_gs_dir=config.get("robot_gs_dir", ""),
        save_dir=config.get("save_dir", "./data/navigation"),
        prompt=config.get("prompt", "Navigate in the scene"),
        auto_reset=config.get("auto_reset", True),
        initial_qpos=config.get("initial_qpos"),
    )
    args = parser.parse_args()
    scene_gaussian_cfg = {
        name: Path(path)
        for name, path in config.get("scene_gaussians", DEFAULT_SCENE_GAUSSIANS).items()
    }

    scene_file = DEMO_DIR / "models" / "robots" / "navigation" / "flat_scene.xml" if args.scene == "plane" else Path(args.scene)
    if not scene_file.is_absolute():
        scene_file = DEMO_DIR / scene_file

    if args.robot == "g1":
        RobotClass = G1Robot
        PolicyClass = G1LocomotionPolicy
    elif args.robot == "go1":
        RobotClass = Go1Robot
        PolicyClass = Go1LocomotionPolicy
    else:
        RobotClass = Go2Robot
        PolicyClass = Go2LocomotionPolicy

    scene = msd.from_file(scene_file.as_posix())
    robot = msd.from_file(RobotClass.mjcf_path.as_posix())
    pos = camera_positions[args.robot]
    camera_mjcf = f"""<mujoco model="camera">
  <worldbody>
    <camera name="follower" pos="{" ".join(str(x) for x in pos)}"
      xyaxes="0 -1 0 0 0 1" trackposspeed="2" trackrotspeed="2" />
  </worldbody>
</mujoco>"""

    camera = msd.from_str(camera_mjcf)
    robot.attach(camera, RobotClass.base_link_name)
    if args.robot == "g1":
        head_camera_mjcf = f"""<mujoco model="g1_head_camera">
  <worldbody>
    <camera name="head_camera" pos="{" ".join(str(x) for x in G1_HEAD_CAMERA_POS)}"
      xyaxes="0 -1 0 0 0 1" />
  </worldbody>
</mujoco>"""
        head_camera = msd.from_str(head_camera_mjcf)
        robot.attach(head_camera, "torso_link")
    scene.attach(robot)
    model = scene.build()

    camera = model.cameras["follower"]
    camera.rotation_track = "look_at_link"
    camera.position_track = "fixed_local"
    camera.track_target_link = model.get_link(RobotClass.base_link_name)

    body = model.get_body(RobotClass.base_link_name)
    robot = RobotClass(body)
    policy = PolicyClass(robot=robot)
    keyboard_adapter = KeyboardCommandAdapter()

    assets_dir = Path(args.assets_dir)
    if not assets_dir.is_absolute():
        assets_dir = DEMO_DIR / assets_dir
    robot_gs_dir = resolve_demo_path(args.robot_gs_dir) if args.robot_gs_dir else assets_dir / args.robot

    gs_ply = None
    if args.gs_ply:
        gs_ply = Path(args.gs_ply)
        if not gs_ply.is_absolute():
            gs_ply = DEMO_DIR / gs_ply

    print(f"Controlling {args.robot.upper()} robot")
    print("=" * 50)
    print("Keyboard Controls:")
    print("  W / Up Arrow    : Forward")
    print("  S / Down Arrow  : Backward")
    print("  Left Arrow      : Strafe Left")
    print("  Right Arrow     : Strafe Right")
    print("  A / D           : Rotate Left / Right")
    print("  ESC             : Exit")
    if args.save_data:
        print("  R               : Save episode and start new")
    print("=" * 50)

    with RenderApp() as render:
        print("Launching render app...", flush=True)
        render.launch(model)
        render.opt.set_group_vis(2, True)
        render.opt.set_group_vis(3, True)
        data = SceneData(model)
        apply_initial_qpos(data, model, args.initial_qpos)
        print("Render app launched.", flush=True)

        gs_renderer = None
        gaussians = {}
        gaussian_links = {}
        if gs_ply is not None:
            if gs_ply.resolve() in scene_gaussian_paths(scene_gaussian_cfg):
                print(f"Skipping background GS because it is already bound to the scene body: {gs_ply}", flush=True)
            else:
                gaussians["background"] = gs_ply.as_posix()
        scene_gaussians = collect_scene_gaussians(scene_gaussian_cfg)
        gaussians.update(scene_gaussians)
        if not args.no_robot_gs:
            robot_gaussians, robot_gaussian_links = collect_robot_gaussians(args.robot, model, robot_gs_dir)
            gaussians.update(robot_gaussians)
            gaussian_links.update(robot_gaussian_links)
            if robot_gaussians:
                print(f"Loaded {len(robot_gaussians)} robot GS assets for {args.robot}: {', '.join(sorted(robot_gaussians))}", flush=True)

        if gaussians:
            print(f"Loading 3DGS assets: {', '.join(sorted(gaussians))}", flush=True)
            from gaussian_renderer import GSRendererMotrixSim

            gs_renderer = GSRendererMotrixSim(gaussians, model)
            apply_gaussian_links(gs_renderer, model, gaussian_links)
            print("3DGS renderer ready.", flush=True)

        head_camera_id = find_camera_id(model, "head_camera")
        if head_camera_id is None and len(model.cameras) > 0:
            head_camera_id = 0

        head_camera_img = np.full((360, 480, 3), [255, 0, 0], dtype=np.uint8)
        head_img = render.create_image(head_camera_img)
        head_widget = render.widgets.create_image_widget(head_img, layout=Layout(left=10, top=10, width=480, height=360))

        bottom_cam_img = np.full((360, 480, 3), [0, 255, 0], dtype=np.uint8)
        bottom_img = render.create_image(bottom_cam_img)
        bottom_widget = render.widgets.create_image_widget(bottom_img, layout=Layout(left=10, top=370, width=480, height=360))

        system_camera = render.system_camera

        step = [0]
        render_tick = [0]
        last_debug_command = [None]
        gs_debug_printed = [False]
        contrl_dt = 0.02
        n_ctrl = max(1, round(contrl_dt / model.options.timestep))

        data_collector = None
        if args.save_data:
            from nav_collect_common import NavDataCollector, VideoCfg
            video_cfg = VideoCfg(fps=30, width=480, height=360)
            data_collector = NavDataCollector(args.save_dir, ["head_camera", "system_camera"], video_cfg, dt=contrl_dt)
            data_collector.start_episode()
            print(f"Data collection enabled. Saving to: {args.save_dir}")

        def phys_step():
            from motrixsim import step as mstep
            mstep(model, data)
            step[0] += 1
            if step[0] % n_ctrl == 0:
                need_reset = policy.step(data, keyboard_adapter.command)
                if need_reset and args.auto_reset:
                    data.reset(model)
                    apply_initial_qpos(data, model, args.initial_qpos)
                    step[0] = 0

                if data_collector:
                    base_link = model.get_link(RobotClass.base_link_name)
                    pose = base_link.get_pose(data)
                    pos = pose[:3]
                    quat = pose[3:7]
                    rot = Rotation.from_quat(quat)
                    euler = rot.as_euler('xyz')
                    base_pose = np.array([pos[0], pos[1], pos[2], euler[0], euler[1], euler[2]])
                    ctrl = np.array([keyboard_adapter.command[0], keyboard_adapter.command[1], keyboard_adapter.command[2]])
                    data_collector.add_step(step[0] // n_ctrl, base_pose, ctrl)

        def render_step():
            render_tick[0] += 1
            keyboard_adapter.update_from_input(render.input)
            if args.debug_input and render_tick[0] % 120 == 1:
                print("debug_input: render loop alive; click the render viewport, then press W/A/S/D or arrow keys", flush=True)
            if args.debug_input:
                pressed = [
                    key
                    for key in (
                        "w", "a", "s", "d",
                        "up", "down", "left", "right",
                        "esc", "escape", "space",
                        "W", "A", "S", "D",
                        "arrowup", "arrowdown", "arrowleft", "arrowright",
                    )
                    if render.input.is_key_pressed(key)
                ]
                command_tuple = tuple(float(x) for x in keyboard_adapter.command)
                if pressed or command_tuple != last_debug_command[0]:
                    print(f"pressed={pressed} command={keyboard_adapter.command.tolist()}", flush=True)
                    last_debug_command[0] = command_tuple
            if args.debug_input and np.any(keyboard_adapter.command):
                print(f"command={keyboard_adapter.command.tolist()}", flush=True)
            if render.input.is_key_just_pressed("esc") or render.input.is_key_just_pressed("escape"):
                if data_collector:
                    data_collector.cleanup()
                return False

            if data_collector and render.input.is_key_just_pressed("r"):
                data_collector.save_episode(args.prompt)
                print(f"Episode saved. Starting new episode...")
                data_collector.start_episode()

            head_rgb = None
            system_rgb = None

            if gs_renderer and head_camera_id is not None:
                gs_renderer.update_gaussians(data)
                results = gs_renderer.render(model, data, [head_camera_id, -1], 480, 360, system_camera=system_camera)
                if args.debug_input and not gs_debug_printed[0]:
                    print(f"3DGS rendered camera ids: {sorted(results.keys())}", flush=True)
                    gs_debug_printed[0] = True

                if head_camera_id in results:
                    rgb_tensor, _ = results[head_camera_id]
                    rgb_np = rgb_tensor.cpu().numpy()
                    if rgb_np.dtype != np.uint8:
                        rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
                    head_img.pixels = rgb_np
                    head_rgb = rgb_np

                if -1 in results:
                    rgb_tensor, _ = results[-1]
                    rgb_np = rgb_tensor.cpu().numpy()
                    if rgb_np.dtype != np.uint8:
                        rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
                    bottom_img.pixels = rgb_np
                    system_rgb = rgb_np

            if data_collector and head_rgb is not None and system_rgb is not None:
                data_collector.add_frame({"head_camera": head_rgb, "system_camera": system_rgb})

            render.sync(data)
            return True

        print("Starting render loop.", flush=True)
        render_fps = 1000000.0 if args.no_sync else 60.0
        run.render_loop(model.options.timestep, render_fps, phys_step, render_step)


if __name__ == "__main__":
    main()
