import sys
from pathlib import Path
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.truck import Truck
from models.state import State, PackagePlacement
from models.package import Package
from utils.objective import calculate_objective
from utils.moves import create_random_state, get_random_neighbor
from utils.constraints import is_valid_state, to_truck_list

@dataclass
class GeneticAlgorithmResult:
    algorithm_name: str
    initial_state: State
    final_state: State
    initial_value: float
    final_value: float
    population_size: int
    iterations: int
    duration: float
    max_history: list[float]
    avg_history: list[float]
    metadata: dict = field(default_factory=dict)

def tournament_selection(population: list[State], fitnesses: list[float], k: int = 3) -> State:
    selected = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(selected, key=lambda idx: fitnesses[idx])
    return population[best_idx]

def single_point_crossover(p1: State, p2: State) -> tuple[State, State]:
    n = len(p1.placements)
    if n < 2:
        return p1.clone(), p2.clone()
        
    point = random.randint(1, n - 1)
    c1_placements = [
        PackagePlacement(p.package, p.position, p.orientation, p.truck_idx)
        for p in (p1.placements[:point] + p2.placements[point:])
    ]
    c2_placements = [
        PackagePlacement(p.package, p.position, p.orientation, p.truck_idx)
        for p in (p2.placements[:point] + p1.placements[point:])
    ]
    return State(c1_placements), State(c2_placements)

def mutate(state: State, truck: Truck | list[Truck], mutation_rate: float) -> State:
    if random.random() < mutation_rate:
        return get_random_neighbor(state, truck)
    return state

def genetic_algorithm(
    packages: list[Package],
    truck: Truck | list[Truck],
    population_size: int = 40,
    iterations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    elite_size: int = 2,
    tournament_k: int = 3,
    objective_func = calculate_objective
) -> GeneticAlgorithmResult:
    start_time = time.time()
    
    population = [create_random_state(packages, truck) for _ in range(population_size)]
    fitnesses = [objective_func(ind, truck) for ind in population]
    
    best_init_idx = max(range(population_size), key=lambda i: fitnesses[i])
    initial_state = population[best_init_idx].clone()
    initial_value = fitnesses[best_init_idx]
    
    max_history = [initial_value]
    avg_history = [float(np.mean(fitnesses))]
    
    for _ in range(iterations):
        sorted_indices = sorted(range(population_size), key=lambda i: fitnesses[i], reverse=True)
        new_population = [population[idx].clone() for idx in sorted_indices[:elite_size]]
        
        while len(new_population) < population_size:
            p1 = tournament_selection(population, fitnesses, k=tournament_k)
            p2 = tournament_selection(population, fitnesses, k=tournament_k)
            
            if random.random() < crossover_rate:
                c1, c2 = single_point_crossover(p1, p2)
            else:
                c1, c2 = p1.clone(), p2.clone()
                
            c1 = mutate(c1, truck, mutation_rate)
            c2 = mutate(c2, truck, mutation_rate)
            
            new_population.append(c1)
            if len(new_population) < population_size:
                new_population.append(c2)
                
        population = new_population
        fitnesses = [objective_func(ind, truck) for ind in population]
        
        max_history.append(float(np.max(fitnesses)))
        avg_history.append(float(np.mean(fitnesses)))
        
    duration = time.time() - start_time
    
    best_final_idx = max(range(population_size), key=lambda i: fitnesses[i])
    final_state = population[best_final_idx].clone()
    final_value = fitnesses[best_final_idx]
    
    return GeneticAlgorithmResult(
        algorithm_name="Genetic Algorithm",
        initial_state=initial_state,
        final_state=final_state,
        initial_value=initial_value,
        final_value=final_value,
        population_size=population_size,
        iterations=iterations,
        duration=duration,
        max_history=max_history,
        avg_history=avg_history,
        metadata={
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "elite_size": elite_size,
            "tournament_k": tournament_k
        }
    )

def plot_genetic_algorithm(result: GeneticAlgorithmResult, save_path: str | None = None):
    plt.figure(figsize=(8, 5))
    plt.plot(result.max_history, label="Maksimum Populasi", color="tab:blue")
    plt.plot(result.avg_history, label="Rata-rata Populasi", color="tab:orange", linestyle="--")
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

def print_result_table(result: GeneticAlgorithmResult, truck: Truck | list[Truck]):
    trucks = to_truck_list(truck)
    table = PrettyTable()
    table.field_names = ["Metrik", "Nilai"]
    table.align["Metrik"] = "l"
    table.align["Nilai"] = "r"
    
    table.add_row(["Algoritma", result.algorithm_name])
    table.add_row(["Jumlah Populasi", result.population_size])
    table.add_row(["Total Iterasi", result.iterations])
    table.add_row(["Nilai Awal (Best)", f"{result.initial_value:.2f}"])
    table.add_row(["Nilai Akhir (Best)", f"{result.final_value:.2f}"])
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
    print("Running Genetic Algorithm...")
    res = genetic_algorithm(packages, truck, population_size=30, iterations=50)
    print_result_table(res, truck)