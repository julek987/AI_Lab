from collections import deque

from graph import Graph


def bfs(graph: Graph, start_city_index: int):
    n = graph.n
    adjacency_matrix = graph.adj_matrix

    # Each state in the BFS queue: (current_index, visited_set, route_list, current_cost)
    queue = deque()
    initial_state = (start_city_index, {start_city_index}, [start_city_index], 0.0)
    queue.append(initial_state)

    best_route = None
    best_cost = float('inf')

    while queue:
        current_city, visited_set, route_list, current_cost = queue.popleft()

        # If we've visited all cities, try returning to start
        if len(visited_set) == n:
            cost_back = adjacency_matrix[current_city][start_city_index]
            if cost_back is not None:  # There's a road to get back
                total_cost = current_cost + cost_back
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_route = route_list + [start_city_index]
        else:
            # Otherwise, explore next unvisited cities
            for next_city in range(n):
                if next_city not in visited_set:
                    cost_edge = adjacency_matrix[current_city][next_city]
                    if cost_edge is not None:
                        new_cost = current_cost + cost_edge
                        new_visited = visited_set.copy()
                        new_visited.add(next_city)
                        new_route = route_list + [next_city]
                        queue.append((next_city, new_visited, new_route, new_cost))

    # Convert best_route from city indices to actual City objects
    if best_route is None:
        return None, float('inf')
    else:
        best_route_cities = [graph.cities[i] for i in best_route]
        return best_route_cities, best_cost


def dfs(graph_obj: Graph, start_city_index: int):
    adjacency_matrix = graph_obj.adj_matrix
    n = graph_obj.n


    best_route = [None]
    best_cost = [float('inf')]

    def dfs_recursive(curr_index, visited, route, cost):
        # If we've visited all cities, try returning to start
        if len(visited) == n:
            cost_back = adjacency_matrix[curr_index][start_city_index]
            if cost_back is not None:
                total_cost = cost + cost_back
                if total_cost < best_cost[0]:
                    best_cost[0] = total_cost
                    best_route[0] = route + [start_city_index]
            return

        # Otherwise, explore next unvisited cities
        for next_city in range(n):
            if next_city not in visited:
                cost_edge = adjacency_matrix[curr_index][next_city]
                if cost_edge is not None:
                    visited.add(next_city)
                    dfs_recursive(
                        next_city,
                        visited,
                        route + [next_city],
                        cost + cost_edge
                    )
                    visited.remove(next_city)

    visited_init = set([start_city_index])
    dfs_recursive(start_city_index, visited_init, [start_city_index], 0.0)

    # Convert best_route from city IDss to City objects
    if best_route[0] is None:
        return None, float('inf')
    else:
        route_cities = [graph_obj.cities[i] for i in best_route[0]]
        return route_cities, best_cost[0]