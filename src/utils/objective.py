import sys
from pathlib import Path

try:
    from ..models.truck import Truck
    from ..models.state import State
    from .constraints import calculate_penalty, is_valid_state, to_truck_list
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from src.models.truck import Truck
    from src.models.state import State
    from src.utils.constraints import calculate_penalty, is_valid_state, to_truck_list

def calculate_objective(state: State, truck: Truck | list[Truck], penalized: bool = True) -> float:
    total_val = sum(p.package.value for p in state.inside_placements)
    if penalized:
        return total_val - calculate_penalty(state, truck)
    return total_val

def objective_eta_priority(state: State, truck: Truck | list[Truck], penalized: bool = True) -> float:
    score = sum(
        p.package.value * (1.0 + (1.0 / (p.package.eta + 1)))
        for p in state.inside_placements
    )
    if penalized:
        return score - calculate_penalty(state, truck)
    return score

def objective_value_density(state: State, truck: Truck | list[Truck], penalized: bool = True) -> float:
    score = sum(
        (p.package.value / max(0.1, p.package.weight)) * 5.0
        for p in state.inside_placements
    )
    if penalized:
        return score - calculate_penalty(state, truck)
    return score

def objective_volume_efficiency(state: State, truck: Truck | list[Truck], penalized: bool = True) -> float:
    trucks = to_truck_list(truck)
    total_truck_vol = sum(t.width * t.length * t.height for t in trucks)
    used_vol = sum(p.dimensions[0] * p.dimensions[1] * p.dimensions[2] for p in state.inside_placements)
    vol_ratio = used_vol / max(1.0, total_truck_vol)

    total_val = sum(p.package.value for p in state.inside_placements)
    score = total_val * (0.8 + 0.2 * vol_ratio)

    if penalized:
        return score - calculate_penalty(state, truck)
    return score

OBJECTIVE_FUNCTIONS = {
    "default": calculate_objective,
    "eta": objective_eta_priority,
    "density": objective_value_density,
    "volume": objective_volume_efficiency,
}

