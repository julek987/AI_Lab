import math
import heapq
from graph import Graph

def dijkstra(adj_matrix, start_index):
    # Get number of cities.
    n = len(adj_matrix)
    # Set initial distances to infinity; start city's distance is 0.
    dist = [float('inf')] * n
    dist[start_index] = 0.0

    # Use a set to track visited cities.
    visited = set()
    # Priority queue to always process the city with the smallest distance.
    priority_queue = [(0.0, start_index)]
    while priority_queue:
        # Get the city with the smallest current distance.
        current_dist, current_city = heapq.heappop(priority_queue)

        # Skip if we've already processed this city.
        if current_city in visited:
            continue
        visited.add(current_city)

        # Check each neighbor of the current city.
        for neighbor_city in range(n):
            cost_edge = adj_matrix[current_city][neighbor_city]
            # Process only if a road exists and the neighbor is not yet visited.
            if cost_edge is not None and neighbor_city not in visited:
                new_dist = current_dist + cost_edge
                # If a shorter path is found, update the distance and add to the queue.
                if new_dist < dist[neighbor_city]:
                    dist[neighbor_city] = new_dist
                    heapq.heappush(priority_queue, (new_dist, neighbor_city))

    return dist

def dijkstra_approx(graph: Graph, start_city_index: int):
    # Get number of cities and the adjacency matrix.
    n = graph.n
    matrix = graph.adj_matrix

    # Create a set of unvisited city IDs and remove the start city.
    unvisited = set(range(n))
    unvisited.remove(start_city_index)

    # Initialize the route with the start city.
    route = [start_city_index]
    current_city = start_city_index
    total_cost = 0.0

    # Loop until all cities are visited.
    while unvisited:
        # Calculate the shortest distances from the current city.
        dist = dijkstra(matrix, current_city)

        best_next_city = None
        best_next_dist = float('inf')
        # Find the closest unvisited city.
        for city_idx in unvisited:
            if dist[city_idx] < best_next_dist:
                best_next_dist = dist[city_idx]
                best_next_city = city_idx

        # If no reachable city is found, return failure.
        if best_next_city is None or math.isinf(best_next_dist):
            return None, float('inf')

        # Add the chosen city to the route and update total cost.
        route.append(best_next_city)
        unvisited.remove(best_next_city)
        total_cost += best_next_dist
        current_city = best_next_city

    # After visiting all cities, calculate cost to return to the start.
    dist_back = dijkstra(matrix, current_city)
    cost_back = dist_back[start_city_index]
    if math.isinf(cost_back):
        return None, float('inf')
    else:
        total_cost += cost_back
        route.append(start_city_index)

    # Convert the route city IDs to City objects.
    route_cities = [graph.cities[i] for i in route]
    return route_cities, total_cost
