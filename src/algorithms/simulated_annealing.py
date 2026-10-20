import sys
from pathlib import Path
import time
import math
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
    from ..utils.moves import get_random_neighbor, create_random_state
    from ..utils.constraints import is_valid_state, to_truck_list
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from src.models.truck import Truck
    from src.models.state import State
    from src.models.package import Package
    from src.utils.objective import calculate_objective
    from src.utils.moves import get_random_neighbor, create_random_state
    from src.utils.constraints import is_valid_state, to_truck_list

@dataclass
class SimulatedAnnealingResult:
    initial_state: State
    final_state: State
    best_state: State
    initial_value: float
    final_value: float
    best_value: float
    iterations: int
    duration: float
    history: list[float]
    exp_history: list[float]
    acceptance_prob_history: list[float]
    temperature_history: list[float]
    stuck_count: int
    stuck_frequency: float
    metadata: dict = field(default_factory=dict)

def simulated_annealing(
    initial_state: State,
    truck: Truck,
    initial_temp: float = 1000.0,
    cooling_rate: float = 0.995,
    min_temp: float = 1e-3,
    max_iterations: int = 1500,
    objective_func = calculate_objective
) -> SimulatedAnnealingResult:
    start_time = time.time()
    
    current = initial_state.clone()
    current_val = objective_func(current, truck)
    
    best_state = current.clone()
    best_val = current_val
    
    temp = initial_temp
    iteration = 0
    
    history = [current_val]
    exp_history = []
    acceptance_prob_history = []
    temperature_history = [temp]
    stuck_count = 0
    
    while iteration < max_iterations and temp > min_temp:
        iteration += 1
        neighbor = get_random_neighbor(current, truck)
        neighbor_val = objective_func(neighbor, truck)
        
        delta_e = neighbor_val - current_val
        exponent = max(-700.0, min(700.0, delta_e / max(1e-9, temp)))
        exp_val = math.exp(exponent)
        exp_history.append(exp_val)
        
        if delta_e > 0:
            prob = 1.0
            acceptance_prob_history.append(prob)
            current = neighbor
            current_val = neighbor_val
            
            if current_val > best_val:
                best_val = current_val
                best_state = current.clone()
        else:
            prob = min(1.0, exp_val)
            acceptance_prob_history.append(prob)
            
            if random.random() < prob:
                current = neighbor
                current_val = neighbor_val
            else:
                stuck_count += 1
                
        history.append(current_val)
        temp *= cooling_rate
        temperature_history.append(temp)
        
    duration = time.time() - start_time
    stuck_frequency = stuck_count / max(1, iteration)
    
    return SimulatedAnnealingResult(
        initial_state=initial_state,
        final_state=current,
        best_state=best_state,
        initial_value=history[0],
        final_value=current_val,
        best_value=best_val,
        iterations=iteration,
        duration=duration,
        history=history,
        exp_history=exp_history,
        acceptance_prob_history=acceptance_prob_history,
        temperature_history=temperature_history,
        stuck_count=stuck_count,
        stuck_frequency=stuck_frequency,
        metadata={
            "initial_temp": initial_temp,
            "final_temp": temp,
            "cooling_rate": cooling_rate
        }
    )

def plot_simulated_annealing(
    result: SimulatedAnnealingResult,
    save_obj_path: str | None = None,
    save_exp_path: str | None = None
):
    plt.figure(figsize=(8, 5))
    plt.plot(result.history, color="tab:blue", label="Objective Value")
    plt.axhline(y=result.best_value, color="tab:red", linestyle="--", label=f"Best Value ({result.best_value:.2f})")
    plt.title("Simulated Annealing - Objective Function vs Iterasi")
    plt.xlabel("Iterasi")
    plt.ylabel("Nilai Objective")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_obj_path:
        plt.savefig(save_obj_path)
    else:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(result.acceptance_prob_history, color="tab:orange", alpha=0.7, label=r"Peluang Penerimaan $\min(1, e^{\Delta E / T})$")
    plt.title(r"Simulated Annealing - Plot $e^{\Delta E / T}$ vs Iterasi")
    plt.xlabel("Iterasi")
    plt.ylabel(r"$e^{\Delta E / T}$")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_exp_path:
        plt.savefig(save_exp_path)
    else:
        plt.show()
    plt.close()

def print_result_table(result: SimulatedAnnealingResult, truck: Truck | list[Truck]):
    trucks = to_truck_list(truck)
    table = PrettyTable()
    table.field_names = ["Metrik", "Nilai"]
    table.align["Metrik"] = "l"
    table.align["Nilai"] = "r"
    
    table.add_row(["Algoritma", "Simulated Annealing"])
    table.add_row(["Nilai Awal", f"{result.initial_value:.2f}"])
    table.add_row(["Nilai Akhir", f"{result.final_value:.2f}"])
    table.add_row(["Nilai Terbaik Ditemukan", f"{result.best_value:.2f}"])
    table.add_row(["Total Iterasi", result.iterations])
    table.add_row(["Durasi Pencarian (detik)", f"{result.duration:.4f}"])
    table.add_row(["Frekuensi Stuck di Local Optima", f"{result.stuck_frequency:.4f} ({result.stuck_count}/{result.iterations})"])
    table.add_row(["Suhu Awal / Akhir", f"{result.metadata.get('initial_temp', 0):.2f} -> {result.metadata.get('final_temp', 0):.4f}"])
    table.add_row(["Paket Dimuat (State Akhir)", f"{len(result.final_state.inside_placements)} / {len(result.final_state.placements)}"])
    if len(trucks) == 1:
        table.add_row(["Total Bobot (State Akhir)", f"{result.final_state.total_weight:.2f} / {trucks[0].max_capacity:.2f}"])
    else:
        for idx, t in enumerate(trucks):
            table.add_row([f"Bobot {t.id} (Truk {idx+1})", f"{result.final_state.truck_weight(idx):.2f} / {t.max_capacity:.2f}"])
        total_cap = sum(t.max_capacity for t in trucks)
        table.add_row(["Total Bobot Armada", f"{result.final_state.total_weight:.2f} / {total_cap:.2f}"])
    table.add_row(["Status State Akhir Valid?", "Ya" if is_valid_state(result.final_state, truck) else "Tidak"])
    print(table)

if __name__ == "__main__":
    truck = Truck(width=10, length=20, height=5, max_capacity=500)
    packages = [
        Package(id=f"P{i}", width=2, length=3, height=1, value=float((i+1)*15), weight=4.0, is_fragile=(i==1), eta=i)
        for i in range(10)
    ]
    state = create_random_state(packages, truck)
    print("Running Simulated Annealing...")
    res = simulated_annealing(state, truck, max_iterations=200)
    print_result_table(res, truck)