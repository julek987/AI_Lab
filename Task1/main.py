import time
import tracemalloc
import math
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from aco import aco_tsp
from bfs_dfs import bfs, dfs
from city import generate_cities
from dijkstra import dijkstra_approx
from graph import Graph
from nn import nearest_neighbor
from astar import a_star_tsp


def run_bfs(graph: Graph, start_city: int):
    return bfs(graph, start_city)


def run_dfs(graph: Graph, start_city: int):
    return dfs(graph, start_city)


def run_nearest_neighbor(graph: Graph, start_city: int):
    return nearest_neighbor(graph, start_city)


def run_dijkstra_approx(graph: Graph, start_city: int):
    return dijkstra_approx(graph, start_city)


def run_a_star_admissible(graph: Graph, start_city: int):
    return a_star_tsp(graph, start_city, heuristic_mode="admissible")


def run_a_star_inadmissible(graph: Graph, start_city: int):
    return a_star_tsp(graph, start_city, heuristic_mode="inadmissible")


def run_aco(graph: Graph):
    return aco_tsp(graph, num_ants=10, num_iterations=50)


def run_algorithm_worker(queue, algorithm_func, graph, start):
    try:
        if start is None:
            result = algorithm_func(graph)
        else:
            result = algorithm_func(graph, start)
        queue.put(result)
    except Exception:
        queue.put((None, math.inf))


def test_algorithm(algorithm_func, graph) -> (Optional[list], float, float, float):
    # Start memory tracking.
    tracemalloc.start()
    start_time = time.perf_counter()

    # Create a Queue to get the result from the worker.
    queue = multiprocessing.Queue()

    # For run_aco, we expect only the graph; for others, pass start=0.
    start_param = None if algorithm_func.__name__ == "run_aco" else 0

    # Create and start the process.
    p = multiprocessing.Process(target=run_algorithm_worker, args=(queue, algorithm_func, graph, start_param))
    p.start()

    # Wait for up to 15 seconds.
    p.join(timeout=30)

    # If the process is still alive, terminate it.
    if p.is_alive():
        p.terminate()
        p.join()
        result = (None, math.inf)
    else:
        try:
            result = queue.get_nowait()
        except Exception:
            result = (None, math.inf)

    end_time = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    time_elapsed = end_time - start_time
    peak_memory = peak_mem / 1024  # Convert bytes to KB.
    return result[0], result[1], time_elapsed, peak_memory


def plot_results(results, algorithms):

    for scenario, data in results.items():
        connection, symmetry = scenario
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Results for scenario: connection='{connection}', symmetry={symmetry}")

        num_algs = len(algorithms)

        # Plot cost vs. number of cities.
        ax = axs[0]
        for idx, (alg_name, _) in enumerate(algorithms):
            # Calculate a small offset for each algorithm.
            offset = (idx - num_algs / 2) * 0.1
            adjusted_n = [n + offset for n in data["n"]]
            # Replace math.inf with np.nan so that failures are not plotted.
            cost_values = [np.nan if math.isinf(c) else c for c in data[alg_name]["cost"]]
            ax.plot(adjusted_n, cost_values, marker='o', label=alg_name)
        ax.set_title("Cost")
        ax.set_xlabel("Number of cities")
        ax.set_ylabel("Cost")
        ax.legend()

        # Plot time vs. number of cities.
        ax = axs[1]
        for idx, (alg_name, _) in enumerate(algorithms):
            offset = (idx - num_algs / 2) * 0.1
            adjusted_n = [n + offset for n in data["n"]]
            time_values = [np.nan if math.isinf(t) else t for t in data[alg_name]["time"]]
            ax.plot(adjusted_n, time_values, marker='o', label=alg_name)
        ax.set_title("Time")
        ax.set_xlabel("Number of cities")
        ax.set_ylabel("Time (s)")
        ax.legend()

        # Plot memory vs. number of cities.
        ax = axs[2]
        for idx, (alg_name, _) in enumerate(algorithms):
            offset = (idx - num_algs / 2) * 0.1
            adjusted_n = [n + offset for n in data["n"]]
            mem_values = [np.nan if math.isinf(m) else m for m in data[alg_name]["mem"]]
            ax.plot(adjusted_n, mem_values, marker='o', label=alg_name)
        ax.set_title("Memory")
        ax.set_xlabel("Number of cities")
        ax.set_ylabel("Memory (KB)")
        ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def main():
    test_sizes = [5, 7, 10, 12]
    scenarios = [
        ("complete", True),
        ("complete", False),
        ("80_percent", True),
        ("80_percent", False)
    ]

    # List of (algorithm name, function) tuples.
    algorithms = [
        ("BFS", run_bfs),
        ("DFS", run_dfs),
        ("NN", run_nearest_neighbor),
        ("Dijkstra", run_dijkstra_approx),
        ("A*_adm", run_a_star_admissible),
        ("A*_inad", run_a_star_inadmissible),
        ("ACO", run_aco)
    ]

    # Initialise a results dictionary.
    # For each scenario we record the number of cities and for each algorithm, lists for cost, time and memory.
    results = {}
    for scenario in scenarios:
        results[scenario] = {"n": []}
        for alg_name, _ in algorithms:
            results[scenario][alg_name] = {"cost": [], "time": [], "mem": []}

    for n in test_sizes:
        print(f"\n===== Testing n = {n} cities =====")
        # 1) Generate cities.
        cities = generate_cities(n)

        for scenario in scenarios:
            connection, symmetry = scenario
            print(f"\nScenario: {connection}, Symmetry={symmetry}")
            # 2) Build the graph.
            graph = Graph(
                cities=cities,
                scenario_connection=connection,
                scenario_symmetry=symmetry
            )

            # Record the number of cities for the current scenario.
            results[scenario]["n"].append(n)

            for alg_name, alg_func in algorithms:
                try:
                    route, cost, t_elapsed, mem_used = test_algorithm(alg_func, graph)
                except Exception:
                    route, cost, t_elapsed, mem_used = (None, math.inf, math.inf, math.inf)
                print(f"  {alg_name} => cost={cost}, time={t_elapsed:.3f}s, mem={mem_used:.1f}KB")
                results[scenario][alg_name]["cost"].append(cost)
                results[scenario][alg_name]["time"].append(t_elapsed)
                results[scenario][alg_name]["mem"].append(mem_used)

    # After testing all cases, plot the results.
    plot_results(results, algorithms)


if __name__ == "__main__":
    main()
