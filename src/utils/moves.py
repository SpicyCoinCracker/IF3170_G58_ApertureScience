import sys
from pathlib import Path
import random

try:
    from ..models.package import Package
    from ..models.truck import Truck
    from ..models.state import State, PackagePlacement, ORIENTATIONS
    from .constraints import to_truck_list
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from src.models.package import Package
    from src.models.truck import Truck
    from src.models.state import State, PackagePlacement, ORIENTATIONS
    from src.utils.constraints import to_truck_list

def create_random_state(packages: list[Package], truck: Truck | list[Truck], place_prob: float = 0.6) -> State:
    trucks = to_truck_list(truck)
    num_trucks = len(trucks)
    placements = []

    for pkg in packages:
        orientation = random.choice(ORIENTATIONS)
        eff_dims = tuple(pkg.dimensions[i] for i in orientation)
        target_truck_idx = random.randrange(num_trucks)
        t = trucks[target_truck_idx]

        can_fit = (
            eff_dims[0] <= t.width and
            eff_dims[1] <= t.length and
            eff_dims[2] <= t.height
        )

        if can_fit and random.random() < place_prob:
            max_x = t.width - eff_dims[0]
            max_y = t.length - eff_dims[1]
            max_z = t.height - eff_dims[2]

            pos_x = random.randint(0, max(0, max_x))
            pos_y = random.randint(0, max(0, max_y))
            pos_z = 0 if random.random() < 0.7 else random.randint(0, max(0, max_z))
            placements.append(PackagePlacement(pkg, (pos_x, pos_y, pos_z), orientation, truck_idx=target_truck_idx))
        else:
            placements.append(PackagePlacement(pkg, None, orientation, truck_idx=target_truck_idx))

    return State(placements)

def swap_move(state: State, truck: Truck | list[Truck]) -> State:
    trucks = to_truck_list(truck)
    num_trucks = len(trucks)
    new_state = state.clone()
    n = len(new_state.placements)
    if n < 2:
        return new_state

    i, j = random.sample(range(n), 2)
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

    return new_state

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
    num_trucks = len(trucks)
    new_state = state.clone()
    if not new_state.placements:
        return new_state

    inside_indices = [i for i, p in enumerate(new_state.placements) if p.position is not None]
    if inside_indices and random.random() < 0.75:
        idx = random.choice(inside_indices)
    else:
        idx = random.randrange(len(new_state.placements))

    target = new_state.placements[idx]
    current_ori = target.orientation

    other_oris = [o for o in ORIENTATIONS if o != current_ori]
    target.orientation = random.choice(other_oris)

    if target.position is not None:
        t = trucks[target.truck_idx % num_trucks]
        w, l, h = target.dimensions
        x = min(target.position[0], max(0, t.width - w))
        y = min(target.position[1], max(0, t.length - l))
        z = min(target.position[2], max(0, t.height - h))
        target.position = (x, y, z)

    return new_state

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

