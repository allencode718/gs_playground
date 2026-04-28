import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from motrixsim import SceneData, msd, run
from motrixsim.render import RenderApp, Layout
from utils.controller import KeyboardCommandAdapter
from utils.policy import G1LocomotionPolicy, Go1LocomotionPolicy, Go2LocomotionPolicy
from utils.robot import G1Robot, Go1Robot, Go2Robot

DEMO_DIR = Path(__file__).resolve().parent

camera_positions = {"g1": [-1.5, 0, 1.0], "go1": [-2, 0, 0.5], "go2": [-2, 0, 0.5]}


def main():
    parser = argparse.ArgumentParser(description="Keyboard control for robots")
    parser.add_argument("--robot", type=str, choices=["g1", "go1", "go2"], default="go2")
    parser.add_argument("--scene", type=str, default="plane", help="Path to scene XML file")
    parser.add_argument("--gs_ply", type=str, default="", help="Path to background gaussian splatting ply file")
    parser.add_argument("--no-sync", action="store_true", help="Disable real-time clock sync")
    parser.add_argument("--save_data", action="store_true", help="Enable data collection")
    parser.add_argument("--save_dir", type=str, default="./data/navigation", help="Data save directory")
    parser.add_argument("--prompt", type=str, default="Navigate in the scene", help="Task prompt")
    parser.add_argument("--debug_input", action="store_true", help="Print keyboard command changes")
    args = parser.parse_args()

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
        data = SceneData(model)
        print("Render app launched.", flush=True)

        gs_renderer = None
        if gs_ply is not None:
            print(f"Loading 3DGS background: {gs_ply}", flush=True)
            from gaussian_renderer import GSRendererMotrixSim

            gaussians = {"background": gs_ply.as_posix()}
            gs_renderer = GSRendererMotrixSim(gaussians, model)
            print("3DGS renderer ready.", flush=True)

        head_camera_id = None
        if "head_camera" in model.cameras:
            head_camera_id = model.cameras["head_camera"].id
        else:
            for i, c in enumerate(model.cameras):
                if c.name.endswith("head_camera"):
                    head_camera_id = i
                    break
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
                if need_reset:
                    data.reset()
                    step[0] = 0

                if data_collector:
                    base_link = model.get_link(RobotClass.base_link_name)
                    pos = base_link.get_position(data)
                    quat = base_link.get_quaternion(data)
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
