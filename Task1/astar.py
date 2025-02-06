import heapq
from math import inf
from typing import List, Set, Tuple

from graph import Graph

def heuristic_estimate(
    graph: Graph,
    current_city: int,
    visited: Set[int],
    start_city: int,
    mode: str = "admissible"
) -> float:
    # Get the total number of cities and the cost matrix.
    n = graph.n
    matrix = graph.adj_matrix

    # Create a list of city IDs that haven't been visited yet.
    unvisited = [city_idx for city_idx in range(n) if city_idx not in visited]

    # If all cities are visited, return the cost to go back to the start.
    if not unvisited:
        cost_back = matrix[current_city][start_city]
        return cost_back if cost_back is not None else 0.0

    # Find the smallest cost from the current city to any unvisited city.
    min_out = inf
    for u in unvisited:
        if matrix[current_city][u] is not None:
            min_out = min(min_out, matrix[current_city][u])
    if min_out is inf:
        min_out = 0

    # Find the smallest cost from any unvisited city back to the start.
    min_back = inf
    for u in unvisited:
        if matrix[u][start_city] is not None:
            min_back = min(min_back, matrix[u][start_city])
    if min_back is inf:
        min_back = 0

    # Estimate the cost to travel among unvisited cities.
    min_unvisited_edge = inf
    if len(unvisited) > 1:
        for i in range(len(unvisited)):
            for j in range(i + 1, len(unvisited)):
                c1 = unvisited[i]
                c2 = unvisited[j]
                # Consider both directions if the cost is asymmetrical.
                possible_costs = []
                if matrix[c1][c2] is not None:
                    possible_costs.append(matrix[c1][c2])
                if matrix[c2][c1] is not None:
                    possible_costs.append(matrix[c2][c1])
                if possible_costs:
                    local_min = min(possible_costs)
                    if local_min < min_unvisited_edge:
                        min_unvisited_edge = local_min
        if min_unvisited_edge is inf:
            min_unvisited_edge = 0

        internal_estimate = min_unvisited_edge * (len(unvisited) - 1)
    else:
        internal_estimate = 0

    # Total heuristic: cost out + cost among unvisited + cost back.
    h_admissible = min_out + internal_estimate + min_back

    # If mode is inadmissible, inflate the heuristic by 30%.
    if mode == "admissible":
        return h_admissible
    else:
        return 1.3 * h_admissible

def a_star_tsp(
    graph: Graph,
    start_city: int,
    heuristic_mode: str = "admissible"
) -> Tuple[List[int], float]:

    # Get number of cities and the cost matrix.
    n = graph.n
    adj = graph.adj_matrix

    # Initialize the open list for A* with the starting state.
    open_list = []
    visited_initial = frozenset([start_city])
    route_initial = [start_city]
    h0 = heuristic_estimate(graph, start_city, visited_initial, start_city, mode=heuristic_mode)
    initial_state = (h0, 0.0, start_city, visited_initial, route_initial)
    heapq.heappush(open_list, initial_state)

    best_route = None  # Best complete route found.
    best_cost = inf  # Best cost found.
    visited_cost_map = {}  # For pruning: map of (current_city, visited_set) to best cost so far.

    while open_list:
        # Pop the state with the lowest f = g + h.
        f_val, g_cost, current_city, visited_set, route_so_far = heapq.heappop(open_list)

        # If all cities have been visited, try to complete the tour.
        if len(visited_set) == n:
            cost_back = adj[current_city][start_city]
            if cost_back is not None:
                total_cost = g_cost + cost_back
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_route = route_so_far + [start_city]
            continue

        # Skip if this state has been seen with a lower cost.
        state_key = (current_city, visited_set)
        if state_key in visited_cost_map and visited_cost_map[state_key] <= g_cost:
            continue
        visited_cost_map[state_key] = g_cost

        # Expand the current state: try all unvisited cities.
        for next_city in range(n):
            if next_city not in visited_set:
                cost_edge = adj[current_city][next_city]
                if cost_edge is not None:
                    new_g = g_cost + cost_edge  # New cost so far.
                    new_visited = visited_set.union({next_city})
                    new_route = route_so_far + [next_city]
                    # Compute heuristic for the new state.
                    h_val = heuristic_estimate(graph, next_city, new_visited, start_city, mode=heuristic_mode)
                    f_val_new = new_g + h_val
                    new_state = (f_val_new, new_g, next_city, frozenset(new_visited), new_route)
                    heapq.heappush(open_list, new_state)

    return best_route, best_cost
