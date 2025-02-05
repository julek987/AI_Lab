import math
import heapq
from graph import Graph

def dijkstra(adj_matrix, start_index):
    """
    Computes the shortest-path distances from 'start_index'
    to every other index using Dijkstra's algorithm.

    :param adj_matrix: A 2D list where adj_matrix[i][j] is the cost of traveling
                       from city i to city j, or None if there's no direct road.
    :param start_index: Index of the city from which to compute shortest paths
    :return: A list 'dist' where dist[i] is the minimum cost to reach city i
             from 'start_index'. Unreachable cities have dist[i] = float('inf').
    """
    n = len(adj_matrix)
    dist = [float('inf')] * n  # Use numeric infinity, not the string 'inf'
    dist[start_index] = 0.0

    visited = set()
    priority_queue = [(0.0, start_index)]  # (distance_so_far, city_index)

    while priority_queue:
        current_dist, current_city = heapq.heappop(priority_queue)

        if current_city in visited:
            continue
        visited.add(current_city)

        # Explore neighbors of current_city
        for neighbor_city in range(n):
            cost_edge = adj_matrix[current_city][neighbor_city]
            if cost_edge is not None and neighbor_city not in visited:
                new_dist = current_dist + cost_edge
                if new_dist < dist[neighbor_city]:
                    dist[neighbor_city] = new_dist
                    heapq.heappush(priority_queue, (new_dist, neighbor_city))

    return dist

def dijkstra_approx(graph: Graph, start_city_index: int):
    """
    Greedy TSP approximation using repeated Dijkstra shortest paths.
    At each step:
      1. Run Dijkstra from current city to all unvisited cities.
      2. Pick the city with the minimum distance among unvisited.
      3. Move to that city, mark it visited.
      4. Repeat until all are visited, then do Dijkstra one last time
         to return to the start city.

    :param graph: Graph object with .adj_matrix (2D list), .cities, and .n
    :param start_city_index: Which city index to start from
    :return: (route_as_list_of_City, total_cost)
    """
    n = graph.n
    matrix = graph.adj_matrix

    unvisited = set(range(n))
    unvisited.remove(start_city_index)

    route = [start_city_index]
    current_city = start_city_index
    total_cost = 0.0

    # Iteratively pick the next city with minimal shortest-path distance
    while unvisited:
        dist = dijkstra(matrix, current_city)

        best_next_city = None
        best_next_dist = float('inf')
        for city_idx in unvisited:
            if dist[city_idx] < best_next_dist:
                best_next_dist = dist[city_idx]
                best_next_city = city_idx

        # If no unvisited city is reachable, return no route
        if best_next_city is None or math.isinf(best_next_dist):
            return None, float('inf')

        route.append(best_next_city)
        unvisited.remove(best_next_city)
        total_cost += best_next_dist
        current_city = best_next_city

    # Finally, return to the start city if possible
    dist_back = dijkstra(matrix, current_city)
    cost_back = dist_back[start_city_index]
    if math.isinf(cost_back):
        return None, float('inf')
    else:
        total_cost += cost_back
        route.append(start_city_index)

    # Convert city indices to the actual City objects
    route_cities = [graph.cities[i] for i in route]
    return route_cities, total_cost
