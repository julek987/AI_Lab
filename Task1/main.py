import time
import tracemalloc
import math
import multiprocessing
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

    # Wait for up to 60 seconds.
    p.join(timeout=15)

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


def main():
    test_sizes = [5, 10, 15, 20]
    scenarios = [
        ("complete", True),
        ("complete", False),
        ("80_percent", True),
        ("80_percent", False)
    ]

    for n in test_sizes:
        print(f"\n===== Testing n = {n} cities =====")
        # 1) Generate cities.
        cities = generate_cities(n)

        for scenario_connection, scenario_symmetry in scenarios:
            print(f"\nScenario: {scenario_connection}, Symmetry={scenario_symmetry}")
            # 2) Build the graph.
            graph = Graph(
                cities=cities,
                scenario_connection=scenario_connection,
                scenario_symmetry=scenario_symmetry
            )

            # BFS TSP.
            try:
                route_bfs, cost_bfs, time_bfs, mem_bfs = test_algorithm(run_bfs, graph)
                print(f"  BFS => cost={cost_bfs}, time={time_bfs:.3f}s, mem={mem_bfs:.1f}KB")
            except Exception:
                print("  BFS => too large, or an error occurred")

            # DFS TSP.
            try:
                route_dfs, cost_dfs, time_dfs, mem_dfs = test_algorithm(run_dfs, graph)
                print(f"  DFS => cost={cost_dfs}, time={time_dfs:.3f}s, mem={mem_dfs:.1f}KB")
            except Exception:
                print("  DFS => too large, or an error occurred")

            # Nearest Neighbor.
            route_nn, cost_nn, time_nn, mem_nn = test_algorithm(run_nearest_neighbor, graph)
            print(f"  NN  => cost={cost_nn}, time={time_nn:.3f}s, mem={mem_nn:.1f}KB")

            # Dijkstra-based.
            route_djk, cost_djk, time_djk, mem_djk = test_algorithm(run_dijkstra_approx, graph)
            print(f"  Dijkstra => cost={cost_djk}, time={time_djk:.3f}s, mem={mem_djk:.1f}KB")

            # A* (admissible heuristic).
            route_astar_adm, cost_astar_adm, time_astar_adm, mem_astar_adm = test_algorithm(run_a_star_admissible,
                                                                                            graph)
            print(f"  A*(adm) => cost={cost_astar_adm}, time={time_astar_adm:.3f}s, mem={mem_astar_adm:.1f}KB")

            # A* (inadmissible heuristic).
            route_astar_inad, cost_astar_inad, time_astar_inad, mem_astar_inad = test_algorithm(run_a_star_inadmissible,
                                                                                                graph)
            print(f"  A*(inad)=> cost={cost_astar_inad}, time={time_astar_inad:.3f}s, mem={mem_astar_inad:.1f}KB")

            # ACO.
            route_aco, cost_aco, time_aco, mem_aco = test_algorithm(run_aco, graph)
            print(f"  ACO => cost={cost_aco}, time={time_aco:.3f}s, mem={mem_aco:.1f}KB")


if __name__ == "__main__":
    main()
