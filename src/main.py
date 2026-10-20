import argparse
from typing import Any, Callable
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from algorithms.hill_climbing import (
    HillClimbingResult,
    hill_climbing_sideways_move,
    plot_objective_history,
    print_result_table as print_hc_table,
    random_restart_hill_climbing,
    simple_hill_climbing,
    steepest_ascent_hill_climbing,
    stochastic_hill_climbing,
)
from algorithms.simulated_annealing import (
    SimulatedAnnealingResult,
    plot_simulated_annealing,
    print_result_table as print_sa_table,
    simulated_annealing,
)
from algorithms.genetic_algorithm import (
    GeneticAlgorithmResult,
    genetic_algorithm,
    plot_genetic_algorithm,
    print_result_table as print_ga_table,
)
from models.truck import Truck
from models.package import Package
from utils.loader import load_instance
from utils.moves import create_random_state
from utils.objective import OBJECTIVE_FUNCTIONS, calculate_objective
from visualization.visualization import save_state_image, show_state

ALGORITHMS: dict[str, tuple[str, Callable[..., Any]]] = {
    "simple": ("Simple Hill Climbing", simple_hill_climbing),
    "steepest": ("Steepest-Ascent Hill Climbing", steepest_ascent_hill_climbing),
    "stochastic": ("Stochastic Hill Climbing", stochastic_hill_climbing),
    "sideways": ("Hill Climbing with Sideways Move", hill_climbing_sideways_move),
    "random_restart": ("Random Restart Hill Climbing", random_restart_hill_climbing),
    "sa": ("Simulated Annealing", simulated_annealing),
    "ga": ("Genetic Algorithm", genetic_algorithm),
}


def _write_metrics_hc(result: HillClimbingResult, truck: Truck | list[Truck], path: Path) -> None:
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_hc_table(result, truck)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _write_metrics_sa(result: SimulatedAnnealingResult, truck: Truck | list[Truck], path: Path) -> None:
    import io
    from contextlib import redirect_stdout


    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_sa_table(result, truck)
    path.write_text(buffer.getvalue(), encoding="utf-8")

def _write_metrics_ga(result: GeneticAlgorithmResult, truck: Truck | list[Truck], path: Path) -> None:
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_ga_table(result, truck)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def run_single(
    algorithm_key: str,
    truck: Truck | list[Truck],
    packages: list[Package],
    seed: int,
    output_dir: Path,
    interactive: bool,
    population_size: int = 40,
    iterations: int | None = None,
    objective_func = calculate_objective,
) -> HillClimbingResult | SimulatedAnnealingResult | GeneticAlgorithmResult:
    random.seed(seed)
    name, algo_fn = ALGORITHMS[algorithm_key]

    kwargs = {"objective_func": objective_func}
    if algorithm_key == "random_restart":
        if iterations is not None:
            kwargs["max_iterations_per_restart"] = iterations
        result = algo_fn(packages, truck, **kwargs)
        initial = result.initial_state
    elif algorithm_key == "ga":
        kwargs["population_size"] = population_size
        if iterations is not None:
            kwargs["iterations"] = iterations
        result = algo_fn(packages, truck, **kwargs)
        initial = result.initial_state
    else:
        initial = create_random_state(packages, truck)
        if iterations is not None:
            kwargs["max_iterations"] = iterations
        result = algo_fn(initial, truck, **kwargs)

    run_dir = output_dir / algorithm_key / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_state_image(
        initial, truck, f"{name} - Initial", run_dir / "initial.png"
    )
    save_state_image(
        result.final_state, truck, f"{name} - Final", run_dir / "final.png"
    )

    if algorithm_key == "sa":
        sa_result: SimulatedAnnealingResult = result

        plot_simulated_annealing(
            sa_result,
            save_obj_path=str(run_dir / "objective.png"),
            save_exp_path=str(run_dir / "acceptance.png"),
        )
        _write_metrics_sa(sa_result, truck, run_dir / "metrics.txt")

        if interactive:
            show_state(initial, truck, f"{name} - Initial")
            show_state(sa_result.final_state, truck, f"{name} - Final")

        return sa_result

    if algorithm_key == "ga":
        ga_result: GeneticAlgorithmResult = result
        plot_genetic_algorithm(ga_result, save_path=str(run_dir / "objective.png"))
        _write_metrics_ga(ga_result, truck, run_dir / "metrics.txt")
        if interactive:
            show_state(initial, truck, f"{name} - Initial")
            show_state(ga_result.final_state, truck, f"{name} - Final")
        return ga_result

    hc_result: HillClimbingResult = result
    plot_objective_history(hc_result, save_path=str(run_dir / "objective.png"))
    _write_metrics_hc(hc_result, truck, run_dir / "metrics.txt")

    if interactive:
        show_state(initial, truck, f"{name} - Initial")
        show_state(hc_result.final_state, truck, f"{name} - Final")

    return hc_result


def run_experiment(
    algorithm_key: str,
    truck: Truck | list[Truck],
    packages: list[Package],
    runs: int,
    output_dir: Path,
    base_seed: int,
    interactive: bool,
    population_size: int = 40,
    iterations: int | None = None,
    objective_func = calculate_objective,
) -> None:
    for run_index in range(runs):
        seed = base_seed + run_index
        open_viewer = interactive and runs == 1 and run_index == 0
        print(f"Running {algorithm_key} (run {run_index + 1}/{runs}, seed={seed})...")
        run_single(
            algorithm_key=algorithm_key,
            truck=truck,
            packages=packages,
            seed=seed,
            output_dir=output_dir,
            interactive=open_viewer,
            population_size=population_size,
            iterations=iterations,
            objective_func=objective_func,
        )

    print(f"Done. Outputs written under: {output_dir / algorithm_key}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IF3170 Tugas Besar 1"
    )
    parser.add_argument(
        "--instance",
        type=Path,
        default=ROOT.parent / "data" / "instances" / "sample.json",
        help="Path to JSON instance (truck + packages)",
    )
    parser.add_argument(
        "--algorithm",
        choices=list(ALGORITHMS.keys()),
        default="steepest",
        help="Search algorithm to run",
    )
    parser.add_argument(
        "--objective",
        choices=list(OBJECTIVE_FUNCTIONS.keys()),
        default="default",
        help="Objective function (default: penalized value; bonus: eta, density, volume)",
    )
    parser.add_argument(
        "--truck-count",
        type=int,
        default=None,
        help="Multi-truk bonus: duplicate truck capacity into N armada trucks",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of experiment runs (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed; run i uses seed + i",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=40,
        help="Population size for Genetic Algorithm (default: 40)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations / generations (defaults per algorithm if not set)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT.parent / "output",
        help="Directory for plots, screenshots, and metrics",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open raylib viewer after a single run (use with --runs 1)",
    )
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="Skip search; show random initial state in raylib",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.instance.exists():
        print(f"Instance not found: {args.instance}", file=sys.stderr)
        return 1

    truck, packages = load_instance(args.instance)

    if args.truck_count and args.truck_count > 1:
        if isinstance(truck, Truck):
            truck = [
                Truck(
                    id=f"Truk_{i+1}",
                    width=truck.width,
                    length=truck.length,
                    height=truck.height,
                    max_capacity=truck.max_capacity,
                )
                for i in range(args.truck_count)
            ]

    if args.view_only:
        initial = create_random_state(packages, truck)
        show_state(initial, truck, "Random initial state")
        return 0

    if args.interactive and args.runs != 1:
        print("Note: --interactive only opens the viewer when --runs 1.", file=sys.stderr)

    objective_fn = OBJECTIVE_FUNCTIONS[args.objective]

    run_experiment(
        algorithm_key=args.algorithm,
        truck=truck,
        packages=packages,
        runs=args.runs,
        output_dir=args.output_dir,
        base_seed=args.seed,
        interactive=args.interactive,
        population_size=args.population_size,
        iterations=args.iterations,
        objective_func=objective_fn,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

