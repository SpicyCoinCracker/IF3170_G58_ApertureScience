import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.truck import Truck
from models.state import State, PackagePlacement

def to_truck_list(truck: Truck | list[Truck]) -> list[Truck]:
    return truck if isinstance(truck, list) else [truck]

def is_within_bounds(placement: PackagePlacement, truck: Truck) -> bool:
    if placement.position is None:
        return True
    x, y, z = placement.position
    w, l, h = placement.dimensions
    return (
        x >= 0 and y >= 0 and z >= 0 and
        x + w <= truck.width and
        y + l <= truck.length and
        z + h <= truck.height
    )

def check_overlap(p1: PackagePlacement, p2: PackagePlacement) -> bool:
    if p1.position is None or p2.position is None:
        return False
    if p1.truck_idx != p2.truck_idx:
        return False
    x1, y1, z1 = p1.position
    w1, l1, h1 = p1.dimensions
    x2, y2, z2 = p2.position
    w2, l2, h2 = p2.dimensions

    overlap_x = max(x1, x2) < min(x1 + w1, x2 + w2)
    overlap_y = max(y1, y2) < min(y1 + l1, y2 + l2)
    overlap_z = max(z1, z2) < min(z1 + h1, z2 + h2)

    return overlap_x and overlap_y and overlap_z

def is_supported(placement: PackagePlacement, inside_placements: list[PackagePlacement]) -> bool:
    if placement.position is None:
        return True
    x, y, z = placement.position
    w, l, _ = placement.dimensions

    if z == 0:
        return True

    for other in inside_placements:
        if other is placement or other.position is None or other.truck_idx != placement.truck_idx:
            continue
        if other.package.is_fragile:
            continue

        ox, oy, oz = other.position
        ow, ol, oh = other.dimensions

        if oz + oh == z:
            if max(x, ox) < min(x + w, ox + ow) and max(y, oy) < min(y + l, oy + ol):
                return True

    return False

def check_fragile_violations(inside_placements: list[PackagePlacement]) -> int:
    violations = 0
    for p in inside_placements:
        if not p.package.is_fragile or p.position is None:
            continue
        px, py, pz = p.position
        pw, pl, ph = p.dimensions

        for other in inside_placements:
            if other is p or other.position is None or other.truck_idx != p.truck_idx:
                continue
            ox, oy, oz = other.position
            ow, ol, _ = other.dimensions

            if oz >= pz + ph:
                if max(px, ox) < min(px + pw, ox + ow) and max(py, oy) < min(py + pl, oy + ol):
                    violations += 1
                    break
    return violations

def get_violations(state: State, truck: Truck | list[Truck]) -> dict:
    trucks = to_truck_list(truck)
    num_trucks = len(trucks)

    inside = state.inside_placements
    out_of_bounds = 0
    for p in inside:
        if p.truck_idx < 0 or p.truck_idx >= num_trucks:
            out_of_bounds += 1
        elif not is_within_bounds(p, trucks[p.truck_idx]):
            out_of_bounds += 1

    overlaps = 0
    n = len(inside)
    for i in range(n):
        for j in range(i + 1, n):
            if check_overlap(inside[i], inside[j]):
                overlaps += 1

    floating = sum(1 for p in inside if not is_supported(p, inside))
    fragile_violations = check_fragile_violations(inside)
    overweight = sum(
        max(0.0, state.truck_weight(t_idx) - t.max_capacity)
        for t_idx, t in enumerate(trucks)
    )

    is_valid = (
        out_of_bounds == 0 and
        overlaps == 0 and
        floating == 0 and
        fragile_violations == 0 and
        overweight == 0.0
    )

    return {
        "out_of_bounds": out_of_bounds,
        "overlaps": overlaps,
        "floating": floating,
        "fragile_violations": fragile_violations,
        "overweight": overweight,
        "is_valid": is_valid
    }

def is_valid_state(state: State, truck: Truck | list[Truck]) -> bool:
    return get_violations(state, truck)["is_valid"]

def violation_units(state: State, truck: Truck | list[Truck]) -> float:
    violations = get_violations(state, truck)
    if violations["is_valid"]:
        return 0.0

    units = (
        violations["out_of_bounds"]
        + violations["overlaps"]
        + violations["floating"]
        + violations["fragile_violations"]
    )
    if violations["overweight"] > 0:
        units += 1.0 + violations["overweight"]
    return float(units)

def calculate_penalty(state: State, truck: Truck | list[Truck], scale: float | None = None) -> float:
    units = violation_units(state, truck)
    if units == 0:
        return 0.0
    if scale is None:
        scale = sum(p.package.value for p in state.placements) + 1.0
    return scale * units
