import sys
from pathlib import Path
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from prettytable import PrettyTable

try:
    from ..models.truck import Truck
    from ..models.state import State
    from ..models.package import Package
    from ..utils.objective import calculate_objective
    from ..utils.moves import get_random_neighbor, generate_neighbors, create_random_state
    from ..utils.constraints import is_valid_state, to_truck_list
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from src.models.truck import Truck
    from src.models.state import State
    from src.models.package import Package
    from src.utils.objective import calculate_objective
    from src.utils.moves import get_random_neighbor, generate_neighbors, create_random_state
    from src.utils.constraints import is_valid_state, to_truck_list

@dataclass
class HillClimbingResult:
    algorithm_name: str
    initial_state: State
    final_state: State
    initial_value: float
    final_value: float
    iterations: int
    duration: float
    history: list[float]
    metadata: dict = field(default_factory=dict)

def simple_hill_climbing(
    initial_state: State,
    truck: Truck,
    max_iterations: int = 500,
    patience: int = 50,
    objective_func = calculate_objective
) -> HillClimbingResult:
    start_time = time.time()
    current = initial_state.clone()
    current_val = objective_func(current, truck)
    history = [current_val]
    
    no_improve_count = 0
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        neighbor = get_random_neighbor(current, truck)
        neighbor_val = objective_func(neighbor, truck)
        
        if neighbor_val > current_val:
            current = neighbor
            current_val = neighbor_val
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        history.append(current_val)
        
        if no_improve_count >= patience:
            break
            
    duration = time.time() - start_time
    return HillClimbingResult(
        algorithm_name="Simple Hill Climbing",
        initial_state=initial_state,
        final_state=current,
        initial_value=history[0],
        final_value=current_val,
        iterations=iteration,
        duration=duration,
        history=history,
        metadata={"stop_reason": "patience_limit" if no_improve_count >= patience else "max_iterations"}
    )

def steepest_ascent_hill_climbing(
    initial_state: State,
    truck: Truck,
    max_iterations: int = 300,
    num_candidates: int = 50,
    objective_func = calculate_objective
) -> HillClimbingResult:
    start_time = time.time()
    current = initial_state.clone()
    current_val = objective_func(current, truck)
    history = [current_val]
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        candidates = generate_neighbors(current, truck, count=num_candidates)
        best_candidate = None
        best_cand_val = -float("inf")
        
        for cand in candidates:
            cand_val = objective_func(cand, truck)
            if cand_val > best_cand_val:
                best_cand_val = cand_val
                best_candidate = cand
                
        if best_candidate is not None and best_cand_val > current_val:
            current = best_candidate
            current_val = best_cand_val
            history.append(current_val)
        else:
            history.append(current_val)
            break
            
    duration = time.time() - start_time
    return HillClimbingResult(
        algorithm_name="Steepest-Ascent Hill Climbing",
        initial_state=initial_state,
        final_state=current,
        initial_value=history[0],
        final_value=current_val,
        iterations=iteration,
        duration=duration,
        history=history,
        metadata={"stop_reason": "local_optima" if iteration < max_iterations else "max_iterations"}
    )

def stochastic_hill_climbing(
    initial_state: State,
    truck: Truck,
    max_iterations: int = 500,
    num_candidates: int = 30,
    objective_func = calculate_objective
) -> HillClimbingResult:
    start_time = time.time()
    current = initial_state.clone()
    current_val = objective_func(current, truck)
    history = [current_val]
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        candidates = generate_neighbors(current, truck, count=num_candidates)
        uphill_moves = []
        
        for cand in candidates:
            cand_val = objective_func(cand, truck)
            if cand_val > current_val:
                uphill_moves.append((cand, cand_val))
                
        if uphill_moves:
            chosen_state, chosen_val = random.choice(uphill_moves)
            current = chosen_state
            current_val = chosen_val
            history.append(current_val)
        else:
            history.append(current_val)
            break
            
    duration = time.time() - start_time
    return HillClimbingResult(
        algorithm_name="Stochastic Hill Climbing",
        initial_state=initial_state,
        final_state=current,
        initial_value=history[0],
        final_value=current_val,
        iterations=iteration,
        duration=duration,
        history=history,
        metadata={"stop_reason": "local_optima" if iteration < max_iterations else "max_iterations"}
    )

def hill_climbing_sideways_move(
    initial_state: State,
    truck: Truck,
    max_iterations: int = 500,
    max_sideways: int = 50,
    num_candidates: int = 40,
    objective_func = calculate_objective
) -> HillClimbingResult:
    start_time = time.time()
    current = initial_state.clone()
    current_val = objective_func(current, truck)
    history = [current_val]
    
    sideways_count = 0
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        candidates = generate_neighbors(current, truck, count=num_candidates)
        
        best_candidate = None
        best_cand_val = -float("inf")
        
        for cand in candidates:
            cand_val = objective_func(cand, truck)
            if cand_val > best_cand_val:
                best_cand_val = cand_val
                best_candidate = cand
                
        if best_candidate is None:
            history.append(current_val)
            break
            
        if best_cand_val > current_val:
            current = best_candidate
            current_val = best_cand_val
            sideways_count = 0
            history.append(current_val)
        elif abs(best_cand_val - current_val) < 1e-6:
            sideways_count += 1
            current = best_candidate
            current_val = best_cand_val
            history.append(current_val)
            
            if sideways_count >= max_sideways:
                break
        else:
            history.append(current_val)
            break
            
    duration = time.time() - start_time
    return HillClimbingResult(
        algorithm_name="Hill Climbing with Sideways Move",
        initial_state=initial_state,
        final_state=current,
        initial_value=history[0],
        final_value=current_val,
        iterations=iteration,
        duration=duration,
        history=history,
        metadata={
            "sideways_moves_done": sideways_count,
            "max_sideways": max_sideways,
            "stop_reason": "max_sideways_reached" if sideways_count >= max_sideways else "local_optima"
        }
    )

def random_restart_hill_climbing(
    packages: list[Package],
    truck: Truck,
    max_restarts: int = 5,
    max_iterations_per_restart: int = 150,
    num_candidates: int = 40,
    objective_func = calculate_objective
) -> HillClimbingResult:
    start_time = time.time()
    best_overall_state = None
    best_overall_val = -float("inf")
    initial_state_first = None
    
    combined_history = []
    iterations_per_restart = []
    
    for r in range(max_restarts):
        start_state = create_random_state(packages, truck)
        if r == 0:
            initial_state_first = start_state
            
        res = steepest_ascent_hill_climbing(
            initial_state=start_state,
            truck=truck,
            max_iterations=max_iterations_per_restart,
            num_candidates=num_candidates,
            objective_func=objective_func
        )
        
        iterations_per_restart.append(res.iterations)
        combined_history.extend(res.history)
        
        if res.final_value > best_overall_val:
            best_overall_val = res.final_value
            best_overall_state = res.final_state
            
    duration = time.time() - start_time
    return HillClimbingResult(
        algorithm_name="Random Restart Hill Climbing",
        initial_state=initial_state_first,
        final_state=best_overall_state,
        initial_value=combined_history[0] if combined_history else 0.0,
        final_value=best_overall_val,
        iterations=sum(iterations_per_restart),
        duration=duration,
        history=combined_history,
        metadata={
            "restarts": max_restarts,
            "iterations_per_restart": iterations_per_restart
        }
    )

def plot_objective_history(result: HillClimbingResult, save_path: str | None = None):
    plt.figure(figsize=(8, 5))
    plt.plot(result.history, marker="o", markersize=2, label="Objective Value")
    plt.title(f"Plot Objective Function - {result.algorithm_name}")
    plt.xlabel("Iterasi")
    plt.ylabel("Nilai Objective")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def print_result_table(result: HillClimbingResult, truck: Truck | list[Truck]):
    trucks = to_truck_list(truck)
    table = PrettyTable()
    table.field_names = ["Metrik", "Nilai"]
    table.align["Metrik"] = "l"
    table.align["Nilai"] = "r"
    
    table.add_row(["Algoritma", result.algorithm_name])
    table.add_row(["Nilai Awal", f"{result.initial_value:.2f}"])
    table.add_row(["Nilai Akhir", f"{result.final_value:.2f}"])
    table.add_row(["Total Iterasi", result.iterations])
    table.add_row(["Durasi Pencarian (detik)", f"{result.duration:.4f}"])
    table.add_row(["Paket Dimuat", f"{len(result.final_state.inside_placements)} / {len(result.final_state.placements)}"])
    if len(trucks) == 1:
        table.add_row(["Total Bobot Dimuat", f"{result.final_state.total_weight:.2f} / {trucks[0].max_capacity:.2f}"])
    else:
        for idx, t in enumerate(trucks):
            table.add_row([f"Bobot {t.id} (Truk {idx+1})", f"{result.final_state.truck_weight(idx):.2f} / {t.max_capacity:.2f}"])
        total_cap = sum(t.max_capacity for t in trucks)
        table.add_row(["Total Bobot Armada", f"{result.final_state.total_weight:.2f} / {total_cap:.2f}"])
    table.add_row(["Status State Valid?", "Ya" if is_valid_state(result.final_state, truck) else "Tidak"])
    
    for k, v in result.metadata.items():
        table.add_row([k, str(v)])
        
    print(table)

if __name__ == "__main__":
    truck = Truck(width=10, length=20, height=5, max_capacity=500)
    packages = [
        Package(id=f"P{i}", width=2, length=3, height=1, value=float((i+1)*15), weight=4.0, is_fragile=(i==1), eta=i)
        for i in range(10)
    ]
    state = create_random_state(packages, truck)
    print("Running Steepest-Ascent Hill Climbing...")
    res = steepest_ascent_hill_climbing(state, truck)
    print_result_table(res, truck)