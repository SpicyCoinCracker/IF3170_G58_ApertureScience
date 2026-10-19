import os
import sys
from ctypes import byref
from pathlib import Path
import pyray as pr



try:
    from models.state import State
    from models.truck import Truck
    from utils.objective import calculate_objective
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from src.models.state import State
    from src.models.truck import Truck
    from src.utils.objective import calculate_objective


def _box_center_and_size(x: int, y: int, z: int, w: int, l: int, h: int):
    """Map corner (x,y,z) with model axes to raylib center (Y-up)."""
    center = (x + w / 2, z + h / 2, y + l / 2)
    size = (float(w), float(h), float(l))
    return center, size


def _draw_box_corner(x: int, y: int, z: int, w: int, l: int, h: int, fill, wire):
    center, (rw, rh, rl) = _box_center_and_size(x, y, z, w, l, h)
    pr.draw_cube(center, rw, rh, rl, fill)
    pr.draw_cube_wires(center, rw, rh, rl, wire)


def _make_camera(truck: Truck) -> pr.Camera3D:
    w, l, h = truck.width, truck.length, truck.height
    return pr.Camera3D(
        [w * 1.6, h * 2.2, l * 1.8],   
        [w / 2, h / 2, l / 2],         
        [0, 1, 0],                     
        45.0,                          
        pr.CAMERA_PERSPECTIVE          
    )


def draw_truck(truck: Truck) -> None:
    w, l, h = truck.width, truck.length, truck.height
    pr.draw_cube_wires((w / 2, h / 2, l / 2), float(w), float(h), float(l), pr.DARKGRAY)
    pr.draw_grid(int(max(w, l) * 2), 1.0)


def draw_state(state: State, truck: Truck) -> None:
    for placement in state.inside_placements:
        if placement.position is None:
            continue
        x, y, z = placement.position
        pw, pl, ph = placement.dimensions
        fill = pr.Color(255, 180, 120, 255) if placement.package.is_fragile else pr.SKYBLUE
        _draw_box_corner(x, y, z, pw, pl, ph, fill, pr.MAROON)

    staging_x = truck.width + 2
    for index, placement in enumerate(state.outside_placements):
        pw, pl, ph = placement.dimensions
        y_slot = index * (pl + 1)
        fill = pr.Color(200, 200, 200, 255) if placement.package.is_fragile else pr.LIGHTGRAY
        _draw_box_corner(staging_x, y_slot, 0, pw, pl, ph, fill, pr.GRAY)


def _draw_hud(state: State, truck: Truck, title: str, window_height: int) -> None:
    objective = calculate_objective(state, truck)
    loaded = len(state.inside_placements)
    total = len(state.placements)

    pr.draw_rectangle(8, 8, 520, 92, pr.fade(pr.RAYWHITE, 0.85))
    pr.draw_rectangle_lines(8, 8, 520, 92, pr.DARKGRAY)
    pr.draw_text(title, 16, 16, 22, pr.BLACK)
    pr.draw_text(f"Objective: {objective:.2f}", 16, 44, 18, pr.DARKGRAY)
    pr.draw_text(
        f"Loaded: {loaded}/{total} | Weight: {state.total_weight:.1f}/{truck.max_capacity:.1f}",
        16,
        68,
        16,
        pr.GRAY,
    )
    pr.draw_text("ESC: close | Mouse: orbit (interactive mode)", 16, window_height - 28, 14, pr.GRAY)


def _render_frame(state, truck, title, camera, interactive):
    width = pr.get_screen_width()
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
        pr.update_camera(camera, pr.CAMERA_ORBITAL)

def save_state_image(
    state: State,
    truck: Truck,
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

    old_cwd = Path.cwd()
    try:
        os.chdir(save_path.parent)
        pr.take_screenshot(save_path.name)
    finally:
        os.chdir(old_cwd)

    pr.close_window()
    return save_path


def show_state(
    state: State,
    truck: Truck,
    title: str,
    width: int = 1280,
    height: int = 720,
) -> None:
    pr.init_window(width, height, title)
    pr.set_target_fps(60)
    camera = _make_camera(truck)

    while not pr.window_should_close():
        if pr.is_key_pressed(pr.KEY_ESCAPE):
            break
        _render_frame(state, truck, title, camera, interactive=True)

    pr.close_window()
