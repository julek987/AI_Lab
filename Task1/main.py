import time
import tracemalloc
import math
from typing import Optional

from aco import aco_tsp
from bfs_dfs import bfs, dfs
from city import generate_cities
from dijkstra import dijkstra_approx
from graph import Graph
from nn import nearest_neighbor
from astar import a_star_tsp


def test_algorithm(algorithm_func, graph, algorithm_name="Unknown") -> (Optional[list], float, float, float):
    """
    Runs a single TSP algorithm, measuring time, memory usage, and result cost.

    :param algorithm_func: A callable that takes (graph, start_index, etc.) and returns (route, cost).
    :param graph: Graph object
    :param algorithm_name: A string to identify the algorithm in logs
    :return: (route, cost, time_elapsed, peak_memory)
    """
    # Start memory tracking
    tracemalloc.start()
    start_time = time.perf_counter()

    # Run the algorithm
    # Many TSP functions expect a start_city_index, e.g.: algorithm_func(graph, 0)
    # If your function signature differs, adjust accordingly:
    try:
        route, cost = algorithm_func(graph, 0)
    except:
        # If something goes wrong or times out, you can catch it
        route, cost = None, math.inf

    end_time = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    time_elapsed = end_time - start_time
    peak_memory = peak_mem / 1024  # convert bytes -> KB
    return route, cost, time_elapsed, peak_memory

def main():
    # We will test n in [5, 10, 15]
    test_sizes = [5, 10]

    # We will test these scenario combos:
    # (scenario_connection, scenario_symmetry)
    scenarios = [
        ("complete", True),
        ("complete", False),
        ("80_percent", True),
        ("80_percent", False)
    ]

    # Prepare a results list or file
    # For demonstration, we'll just print results
    for n in test_sizes:
        print(f"\n===== Testing n = {n} cities =====")

        # 1) Generate cities
        cities = generate_cities(n)

        for scenario_connection, scenario_symmetry in scenarios:
            print(f"\nScenario: {scenario_connection}, Symmetry={scenario_symmetry}")

            # 2) Build the graph
            graph = Graph(
                cities=cities,
                scenario_connection=scenario_connection,
                scenario_symmetry=scenario_symmetry
            )

            # We'll now run each TSP algorithm and record results

            # BFS TSP (WARNING: might blow up for n=15 or 20!)
            # If your BFS function is named 'bfs_tsp(graph, start_index)', do:
            try:
                route_bfs, cost_bfs, time_bfs, mem_bfs = test_algorithm(bfs, graph, "BFS")
                print(f"  BFS => cost={cost_bfs}, time={time_bfs:.3f}s, mem={mem_bfs:.1f}KB")
            except:
                print("  BFS => too large, or an error occurred")

            # DFS TSP
            try:
                route_dfs, cost_dfs, time_dfs, mem_dfs = test_algorithm(dfs, graph, "DFS")
                print(f"  DFS => cost={cost_dfs}, time={time_dfs:.3f}s, mem={mem_dfs:.1f}KB")
            except:
                print("  DFS => too large, or an error occurred")

            # Nearest Neighbor
            route_nn, cost_nn, time_nn, mem_nn = test_algorithm(nearest_neighbor, graph, "NN")
            print(f"  NN  => cost={cost_nn}, time={time_nn:.3f}s, mem={mem_nn:.1f}KB")

            # Dijkstra-based
            route_djk, cost_djk, time_djk, mem_djk = test_algorithm(dijkstra_approx, graph, "Dijkstra-based")
            print(f"  Dijkstra => cost={cost_djk}, time={time_djk:.3f}s, mem={mem_djk:.1f}KB")

            # A* (admissible heuristic)
            def a_star_admissible(g, start_index):
                return a_star_tsp(g, start_index, heuristic_mode="admissible")
            route_astar_adm, cost_astar_adm, time_astar_adm, mem_astar_adm = test_algorithm(a_star_admissible, graph, "A*_adm")
            print(f"  A*(adm) => cost={cost_astar_adm}, time={time_astar_adm:.3f}s, mem={mem_astar_adm:.1f}KB")

            # A* (inadmissible heuristic)
            def a_star_inadmissible(g, start_index):
                return a_star_tsp(g, start_index, heuristic_mode="inadmissible")
            route_astar_inad, cost_astar_inad, time_astar_inad, mem_astar_inad = test_algorithm(a_star_inadmissible, graph, "A*_inad")
            print(f"  A*(inad)=> cost={cost_astar_inad}, time={time_astar_inad:.3f}s, mem={mem_astar_inad:.1f}KB")

            # ACO
            def aco_runner(g, start_index):
                # your aco function might not need start_index; adapt if needed
                return aco_tsp(g, num_ants=10, num_iterations=50)

            route_aco, cost_aco, time_aco, mem_aco = test_algorithm(aco_runner, graph, "ACO")
            print(f"  ACO => cost={cost_aco}, time={time_aco:.3f}s, mem={mem_aco:.1f}KB")


if __name__ == "__main__":
    main()
