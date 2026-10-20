import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.truck import Truck
from models.state import State, PackagePlacement
from utils.constraints import calculate_penalty, is_valid_state, to_truck_list


def _with_penalty(score_fn, state: State, truck: Truck | list[Truck], penalized: bool) -> float:
    score = score_fn(state.inside_placements, truck)
    if not penalized:
        return score
    scale = score_fn(state.placements, truck) + 1.0
    return score - calculate_penalty(state, truck, scale=scale)


def _score_value(placements: list[PackagePlacement], truck) -> float:
    return sum(p.package.value for p in placements)

def _score_eta(placements: list[PackagePlacement], truck) -> float:
    return sum(p.package.value * (1.0 + (1.0 / (p.package.eta + 1))) for p in placements)

def _score_density(placements: list[PackagePlacement], truck) -> float:
    return sum((p.package.value / max(0.1, p.package.weight)) * 5.0 for p in placements)

def _score_volume(placements: list[PackagePlacement], truck) -> float:
    trucks = to_truck_list(truck)
    total_truck_vol = sum(t.width * t.length * t.height for t in trucks)
    used_vol = sum(p.dimensions[0] * p.dimensions[1] * p.dimensions[2] for p in placements)
    vol_ratio = used_vol / max(1.0, total_truck_vol)
    return _score_value(placements, truck) * (0.8 + 0.2 * vol_ratio)


def calculate_objective(state: State, truck: Truck | list[Truck], penalized: bool = True) -> float:
    return _with_penalty(_score_value, state, truck, penalized)

def objective_eta_priority(state: State, truck: Truck | list[Truck], penalized: bool = True) -> float:
    return _with_penalty(_score_eta, state, truck, penalized)

def objective_value_density(state: State, truck: Truck | list[Truck], penalized: bool = True) -> float:
    return _with_penalty(_score_density, state, truck, penalized)

def objective_volume_efficiency(state: State, truck: Truck | list[Truck], penalized: bool = True) -> float:
    return _with_penalty(_score_volume, state, truck, penalized)

OBJECTIVE_FUNCTIONS = {
    "default": calculate_objective,
    "eta": objective_eta_priority,
    "density": objective_value_density,
    "volume": objective_volume_efficiency,
}
