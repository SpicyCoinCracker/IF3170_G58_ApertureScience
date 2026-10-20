import os
import sys
from ctypes import byref
from pathlib import Path
import pyray as pr


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.state import State
from models.truck import Truck
from utils.objective import calculate_objective
from utils.constraints import to_truck_list


def _box_center_and_size(x: int, y: int, z: int, w: int, l: int, h: int):
    center = (x + w / 2.0, z + h / 2.0, y + l / 2.0)
    size = (float(w), float(h), float(l))
    return center, size


def _draw_box_corner(x: int, y: int, z: int, w: int, l: int, h: int, fill, wire):
    center, (rw, rh, rl) = _box_center_and_size(x, y, z, w, l, h)
    pr.draw_cube(center, rw, rh, rl, fill)
    pr.draw_cube_wires(center, rw, rh, rl, wire)


def _truck_offsets(trucks: list[Truck]) -> tuple[list[float], float]:
    offsets: list[float] = []
    cur_x = 0.0
    for t in trucks:
        offsets.append(cur_x)
        cur_x += t.width + 4.0
    return offsets, cur_x


def _make_camera(truck: Truck | list[Truck]) -> pr.Camera3D:
    trucks = to_truck_list(truck)
    offsets, total_w = _truck_offsets(trucks)
    max_l = max(t.length for t in trucks)
    max_h = max(t.height for t in trucks)

    target = [total_w / 2.0, max_h / 2.0, max_l / 2.0]
    cam_pos = [total_w * 1.3 + 4.0, max_h * 2.4 + 4.0, max_l * 1.8 + 4.0]
    return pr.Camera3D(
        cam_pos,
        target,
        [0, 1, 0],
        45.0,
        pr.CameraProjection.CAMERA_PERSPECTIVE
    )


def draw_truck(truck: Truck | list[Truck]) -> None:
    trucks = to_truck_list(truck)
    offsets, total_w = _truck_offsets(trucks)
    max_l = max(t.length for t in trucks)

    for i, t in enumerate(trucks):
        ox = offsets[i]
        w, l, h = t.width, t.length, t.height
        center = (ox + w / 2.0, h / 2.0, l / 2.0)
        pr.draw_cube_wires(center, float(w), float(h), float(l), pr.DARKGRAY)
    pr.draw_grid(int(max(total_w, max_l) * 2), 1.0)


def draw_state(state: State, truck: Truck | list[Truck]) -> None:
    trucks = to_truck_list(truck)
    offsets, total_w = _truck_offsets(trucks)

    for placement in state.inside_placements:
        if placement.position is None:
            continue
        x, y, z = placement.position
        pw, pl, ph = placement.dimensions
        t_idx = placement.truck_idx % len(trucks)
        ox = offsets[t_idx]
        fill = pr.Color(255, 180, 120, 255) if placement.package.is_fragile else pr.SKYBLUE
        _draw_box_corner(int(ox + x), y, z, pw, pl, ph, fill, pr.MAROON)

    staging_x = total_w + 2.0
    for index, placement in enumerate(state.outside_placements):
        pw, pl, ph = placement.dimensions
        y_slot = index * (pl + 1)
        fill = pr.Color(200, 200, 200, 255) if placement.package.is_fragile else pr.LIGHTGRAY
        _draw_box_corner(int(staging_x), y_slot, 0, pw, pl, ph, fill, pr.GRAY)


def _draw_hud(state: State, truck: Truck | list[Truck], title: str, window_height: int) -> None:
    trucks = to_truck_list(truck)
    objective = calculate_objective(state, truck)
    loaded = len(state.inside_placements)
    total = len(state.placements)
    total_cap = sum(t.max_capacity for t in trucks)

    pr.draw_rectangle(8, 8, 560, 96, pr.fade(pr.RAYWHITE, 0.85))
    pr.draw_rectangle_lines(8, 8, 560, 96, pr.DARKGRAY)
    pr.draw_text(title, 16, 16, 22, pr.BLACK)
    pr.draw_text(f"Objective: {objective:.2f}", 16, 44, 18, pr.DARKGRAY)

    if len(trucks) == 1:
        weight_str = f"Weight: {state.total_weight:.1f}/{trucks[0].max_capacity:.1f}"
    else:
        weight_str = f"Fleet Weight: {state.total_weight:.1f}/{total_cap:.1f} ({len(trucks)} Trucks)"

    pr.draw_text(
        f"Loaded: {loaded}/{total} | {weight_str}",
        16,
        68,
        16,
        pr.GRAY,
    )
    pr.draw_text("ESC: close | Mouse: orbit (interactive mode)", 16, window_height - 28, 14, pr.GRAY)


def _render_frame(state, truck, title, camera, interactive):
    height = pr.get_screen_height()

    pr.begin_drawing()
    pr.clear_background(pr.RAYWHITE)

    pr.begin_mode_3d(camera)
    draw_truck(truck)
    draw_state(state, truck)
    pr.end_mode_3d()

    _draw_hud(state, truck, title, height)
    pr.end_drawing()

    if interactive:
        pr.update_camera(camera, pr.CameraMode.CAMERA_ORBITAL)

def save_state_image(
    state: State,
    truck: Truck | list[Truck],
    title: str,
    save_path: Path | str,
    width: int = 1280,
    height: int = 720,
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pr.init_window(width, height, title)
    pr.set_target_fps(60)
    camera = _make_camera(truck)

    for _ in range(3):
        _render_frame(state, truck, title, camera, interactive=False)

    rel_path = os.path.relpath(save_path, Path.cwd()).replace("\\", "/")
    pr.take_screenshot(rel_path)

    pr.close_window()
    return save_path


def show_state(
    state: State,
    truck: Truck | list[Truck],
    title: str,
    width: int = 1280,
    height: int = 720,
) -> None:
    pr.init_window(width, height, title)
    pr.set_target_fps(60)
    camera = _make_camera(truck)

    while not pr.window_should_close():
        if pr.is_key_pressed(pr.KeyboardKey.KEY_ESCAPE):
            break
        _render_frame(state, truck, title, camera, interactive=True)

    pr.close_window()

