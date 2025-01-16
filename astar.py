import heapq
from math import inf, sqrt
from typing import List, Set, Tuple

# IMPORT your existing classes and cost function
# Adjust these import paths as needed, depending on your project structure.
from city import City, calculate_cost
from graph import Graph

def heuristic_estimate(
    graph: Graph,
    current_city: int,
    visited: Set[int],
    start_city: int,
    mode: str = "admissible"
) -> float:
    """
    Estimate the cost to finish visiting all remaining unvisited cities and
    return to the start.

    :param graph: The Graph object (with .adj_matrix, .cities, etc.)
    :param current_city: Index of the current city
    :param visited: Set of city indices already visited
    :param start_city: Index of the start city (we must return here at the end)
    :param mode: "admissible" or "inadmissible" to switch heuristic type
    :return: A numeric estimate of the remaining cost.
    """
    n = graph.n
    adj = graph.adj_matrix

    unvisited = [city_idx for city_idx in range(n) if city_idx not in visited]

    # If no unvisited left, just consider the direct cost back to the start (if any)
    if not unvisited:
        cost_back = adj[current_city][start_city]
        return cost_back if cost_back is not None else 0.0

    # -------------------------
    # EXAMPLE: Simple heuristic
    # -------------------------
    # We'll estimate:
    # 1) minimal direct edge from current_city to unvisited,
    # 2) minimal edge among unvisited (repeated some factor),
    # 3) minimal edge from unvisited back to start,
    # combined into one sum. (Using an MST or min matching would be stronger.)

    # 1) Minimal cost out of current_city to one unvisited city
    min_out = inf
    for u in unvisited:
        if adj[current_city][u] is not None:
            min_out = min(min_out, adj[current_city][u])
    if min_out is inf:
        min_out = 0  # No direct connection => treat as 0 for the heuristic

    # 2) Minimal cost from unvisited -> start
    min_back = inf
    for u in unvisited:
        if adj[u][start_city] is not None:
            min_back = min(min_back, adj[u][start_city])
    if min_back is inf:
        min_back = 0

    # 3) Internal cost among unvisited (very rough):
    #    find the smallest edge among unvisited and multiply by (count-1)
    #    or do something slightly bigger if we want inadmissible.
    min_unvisited_edge = inf
    if len(unvisited) > 1:
        for i in range(len(unvisited)):
            for j in range(i + 1, len(unvisited)):
                c1 = unvisited[i]
                c2 = unvisited[j]
                # Consider both directions if asymmetrical
                possible_costs = []
                if adj[c1][c2] is not None:
                    possible_costs.append(adj[c1][c2])
                if adj[c2][c1] is not None:
                    possible_costs.append(adj[c2][c1])
                if possible_costs:
                    local_min = min(possible_costs)
                    if local_min < min_unvisited_edge:
                        min_unvisited_edge = local_min
        if min_unvisited_edge is inf:
            min_unvisited_edge = 0

        internal_estimate = min_unvisited_edge * (len(unvisited) - 1)
    else:
        internal_estimate = 0

    h_admissible = min_out + internal_estimate + min_back

    if mode == "admissible":
        return h_admissible
    else:
        # For demonstration, multiply by a factor, making it potentially overestimate
        return 1.3 * h_admissible  # E.g., 30% over

def a_star_tsp(
    graph: Graph,
    start_city: int,
    heuristic_mode: str = "admissible"
) -> Tuple[List[int], float]:
    """
    A* search to solve/approximate TSP starting from 'start_city'.

    :param graph: Graph object (with .adj_matrix, .n)
    :param start_city: Index of the city to start from
    :param heuristic_mode: "admissible" or "inadmissible"
    :return: (best_route_indices, best_cost)
             best_route_indices is a list of city indices
             best_cost is the numeric cost of that route
    """
    n = graph.n
    adj = graph.adj_matrix

    # State in priority queue: (f_value, g_cost, current_city, visited_set, route_list)
    # where f = g_cost + h(current_state).
    open_list = []
    visited_initial = frozenset([start_city])
    route_initial = [start_city]

    # Initial heuristic
    h0 = heuristic_estimate(graph, start_city, visited_initial, start_city, mode=heuristic_mode)
    initial_state = (h0, 0.0, start_city, visited_initial, route_initial)
    heapq.heappush(open_list, initial_state)

    best_route = None
    best_cost = inf

    # Dictionary to track (city, visited_set) => best known g_cost
    visited_cost_map = {}

    while open_list:
        f_val, g_cost, current_city, visited_set, route_so_far = heapq.heappop(open_list)

        # If we have visited all cities, check cost to return to start
        if len(visited_set) == n:
            cost_back = adj[current_city][start_city]
            if cost_back is not None:
                total_cost = g_cost + cost_back
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_route = route_so_far + [start_city]
            continue

        # Check if there's a cheaper cost for this sub-problem
        state_key = (current_city, visited_set)
        if state_key in visited_cost_map and visited_cost_map[state_key] <= g_cost:
            # We've reached this (city, visited config) with cheaper or equal cost before
            continue
        visited_cost_map[state_key] = g_cost

        # Expand neighbors: try going to each unvisited city
        for next_city in range(n):
            if next_city not in visited_set:
                cost_edge = adj[current_city][next_city]
                if cost_edge is not None:
                    new_g = g_cost + cost_edge
                    new_visited = visited_set.union({next_city})
                    new_route = route_so_far + [next_city]
                    h_val = heuristic_estimate(graph, next_city, new_visited, start_city, mode=heuristic_mode)
                    f_val_new = new_g + h_val
                    new_state = (f_val_new, new_g, next_city, frozenset(new_visited), new_route)
                    heapq.heappush(open_list, new_state)

    return best_route, best_cost