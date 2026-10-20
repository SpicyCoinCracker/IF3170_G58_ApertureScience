import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.package import Package
from models.truck import Truck
from models.state import State, PackagePlacement, ORIENTATIONS
from utils.constraints import to_truck_list, is_within_bounds, check_overlap, is_supported

def _footprint_overlap(a: PackagePlacement, b: PackagePlacement) -> bool:
    assert a.position is not None and b.position is not None
    ax, ay, _ = a.position
    aw, al, _ = a.dimensions
    bx, by, _ = b.position
    bw, bl, _ = b.dimensions
    return max(ax, bx) < min(ax + aw, bx + bw) and max(ay, by) < min(ay + al, by + bl)

def _can_place(cand: PackagePlacement, placed: list[PackagePlacement], t: Truck) -> bool:
    if not is_within_bounds(cand, t):
        return False
    same_truck = [p for p in placed if p.truck_idx == cand.truck_idx]
    if any(check_overlap(cand, p) for p in same_truck):
        return False
    if not is_supported(cand, same_truck):
        return False
    assert cand.position is not None
    cz, ch = cand.position[2], cand.dimensions[2]
    for p in same_truck:
        if not _footprint_overlap(cand, p):
            continue
        assert p.position is not None
        pz, ph = p.position[2], p.dimensions[2]
        if p.package.is_fragile and cz >= pz + ph:
            return False
        if cand.package.is_fragile and pz >= cz + ch:
            return False
    return True

def create_random_state(
    packages: list[Package],
    truck: Truck | list[Truck],
    place_prob: float = 0.6,
    max_tries: int = 30,
) -> State:
    trucks = to_truck_list(truck)
    num_trucks = len(trucks)
    placements: list[PackagePlacement] = []
    placed: list[PackagePlacement] = []
    loaded_weight = [0.0] * num_trucks

    for pkg in packages:
        chosen = None
        if random.random() < place_prob:
            for _ in range(max_tries):
                t_idx = random.randrange(num_trucks)
                t = trucks[t_idx]
                if loaded_weight[t_idx] + pkg.weight > t.max_capacity:
                    continue
                orientation = random.choice(ORIENTATIONS)
                w, l, h = (pkg.dimensions[i] for i in orientation)
                if w > t.width or l > t.length or h > t.height:
                    continue

                bases = [p for p in placed if p.truck_idx == t_idx and not p.package.is_fragile]
                base_options: list[PackagePlacement | None] = [None, *bases]
                base = random.choice(base_options)
                if base is None:
                    x = random.randint(0, t.width - w)
                    y = random.randint(0, t.length - l)
                    z = 0
                else:
                    assert base.position is not None
                    bx, by, bz = base.position
                    bw, bl, bh = base.dimensions
                    x_lo, x_hi = max(0, bx - w + 1), min(bx + bw - 1, t.width - w)
                    y_lo, y_hi = max(0, by - l + 1), min(by + bl - 1, t.length - l)
                    if x_lo > x_hi or y_lo > y_hi:
                        continue
                    x = random.randint(x_lo, x_hi)
                    y = random.randint(y_lo, y_hi)
                    z = bz + bh

                cand = PackagePlacement(pkg, (x, y, z), orientation, truck_idx=t_idx)
                if _can_place(cand, placed, t):
                    chosen = cand
                    break

        if chosen is None:
            chosen = PackagePlacement(pkg, None, random.choice(ORIENTATIONS), truck_idx=random.randrange(num_trucks))
        else:
            placed.append(chosen)
            loaded_weight[chosen.truck_idx] += pkg.weight
        placements.append(chosen)

    return State(placements)

def swap_move(state: State, truck: Truck | list[Truck]) -> State:
    trucks = to_truck_list(truck)
    new_state = state.clone()
    n = len(new_state.placements)
    if n < 2:
        return new_state

    i, j = random.sample(range(n), 2)
    _apply_swap(new_state, trucks, i, j)
    return new_state

def _apply_swap(new_state: State, trucks: list[Truck], i: int, j: int) -> None:
    num_trucks = len(trucks)
    p1 = new_state.placements[i]
    p2 = new_state.placements[j]

    pos1, pos2 = p1.position, p2.position
    t1_idx, t2_idx = p1.truck_idx, p2.truck_idx

    def adjust_pos(pos, dims, t: Truck):
        if pos is None:
            return None
        x = min(pos[0], max(0, t.width - dims[0]))
        y = min(pos[1], max(0, t.length - dims[1]))
        z = min(pos[2], max(0, t.height - dims[2]))
        return (x, y, z)

    new_t1_idx = t2_idx % num_trucks
    new_t2_idx = t1_idx % num_trucks

    p1.truck_idx = new_t1_idx
    p1.position = adjust_pos(pos2, p1.dimensions, trucks[new_t1_idx])

    p2.truck_idx = new_t2_idx
    p2.position = adjust_pos(pos1, p2.dimensions, trucks[new_t2_idx])

def relocate_move(state: State, truck: Truck | list[Truck]) -> State:
    trucks = to_truck_list(truck)
    num_trucks = len(trucks)
    new_state = state.clone()
    if not new_state.placements:
        return new_state

    idx = random.randrange(len(new_state.placements))
    target = new_state.placements[idx]
    w, l, h = target.dimensions

    if target.position is not None:
        if random.random() < 0.2:
            target.position = None
            return new_state

    target_truck_idx = random.randrange(num_trucks)
    t = trucks[target_truck_idx]
    target.truck_idx = target_truck_idx

    if w <= t.width and l <= t.length and h <= t.height:
        mode = random.random()
        if mode < 0.4:
            x = random.randint(0, t.width - w)
            y = random.randint(0, t.length - l)
            target.position = (x, y, 0)
        elif mode < 0.7:
            inside = [
                p for p in new_state.inside_placements_for_truck(target_truck_idx)
                if p is not target and not p.package.is_fragile
            ]
            if inside:
                base = random.choice(inside)
                assert base.position is not None
                bx, by, bz = base.position
                bw, bl, bh = base.dimensions
                new_z = bz + bh
                if new_z + h <= t.height:
                    x = min(bx, max(0, t.width - w))
                    y = min(by, max(0, t.length - l))
                    target.position = (x, y, new_z)
                else:
                    x = random.randint(0, t.width - w)
                    y = random.randint(0, t.length - l)
                    target.position = (x, y, 0)
            else:
                x = random.randint(0, t.width - w)
                y = random.randint(0, t.length - l)
                target.position = (x, y, 0)
        else:
            x = random.randint(0, t.width - w)
            y = random.randint(0, t.length - l)
            z = random.randint(0, t.height - h)
            target.position = (x, y, z)
    else:
        target.position = None

    return new_state

def rotate_move(state: State, truck: Truck | list[Truck]) -> State:
    trucks = to_truck_list(truck)
    new_state = state.clone()
    if not new_state.placements:
        return new_state

    inside_indices = [i for i, p in enumerate(new_state.placements) if p.position is not None]
    if inside_indices and random.random() < 0.75:
        idx = random.choice(inside_indices)
    else:
        idx = random.randrange(len(new_state.placements))

    _apply_rotation(new_state, trucks, idx, random.choice(tuple(AXIS_SWAPS)))
    return new_state

AXIS_SWAPS = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}

def _apply_rotation(new_state: State, trucks: list[Truck], idx: int, axis: str) -> None:
    target = new_state.placements[idx]
    i, j = AXIS_SWAPS[axis]
    new_ori = list(target.orientation)
    new_ori[i], new_ori[j] = new_ori[j], new_ori[i]
    target.orientation = (new_ori[0], new_ori[1], new_ori[2])

    if target.position is not None:
        t = trucks[target.truck_idx % len(trucks)]
        w, l, h = target.dimensions
        x = min(target.position[0], max(0, t.width - w))
        y = min(target.position[1], max(0, t.length - l))
        z = min(target.position[2], max(0, t.height - h))
        target.position = (x, y, z)

def get_random_neighbor(state: State, truck: Truck | list[Truck]) -> State:
    move_choice = random.choice(["swap", "relocate", "rotate"])
    if move_choice == "swap":
        return swap_move(state, truck)
    elif move_choice == "relocate":
        return relocate_move(state, truck)
    else:
        return rotate_move(state, truck)

def generate_neighbors(state: State, truck: Truck | list[Truck], count: int = 40) -> list[State]:
    return [get_random_neighbor(state, truck) for _ in range(count)]


def generate_full_neighbors(state: State, truck: Truck | list[Truck], relocate_samples: int = 50) -> list[State]:
    trucks = to_truck_list(truck)
    placements = state.placements
    n = len(placements)
    neighbors: list[State] = []

    for i in range(n):
        for j in range(i + 1, n):
            if placements[i].position is None and placements[j].position is None:
                continue
            s = state.clone()
            _apply_swap(s, trucks, i, j)
            neighbors.append(s)

    for idx in range(n):
        for axis in AXIS_SWAPS:
            s = state.clone()
            _apply_rotation(s, trucks, idx, axis)
            neighbors.append(s)

    neighbors.extend(relocate_move(state, truck) for _ in range(relocate_samples))
    return neighbors
