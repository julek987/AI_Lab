from graph import Graph

def nearest_neighbor(graph: Graph, start_city_index: int):
    # Get the total number of cities and the cost matrix.
    n = graph.n
    matrix = graph.adj_matrix

    # Create a set of all city IDs and remove the start city.
    unvisited = set(range(n))
    unvisited.remove(start_city_index)

    # Initialize the route with the starting city.
    route = [start_city_index]
    current_city = start_city_index
    total_cost = 0.0

    # Loop until all cities are visited.
    while unvisited:
        best_next_city = None  # Best next city to visit.
        best_next_cost = float('inf')  # Cost to reach the best next city.

        # Find the nearest unvisited city.
        for city_id in unvisited:
            cost_edge = matrix[current_city][city_id]
            if cost_edge is not None and cost_edge < best_next_cost:
                best_next_cost = cost_edge
                best_next_city = city_id

        # If no city is reachable, return failure.
        if best_next_city is None:
            return None, float('inf')

        # Add the chosen city to the route and update cost.
        route.append(best_next_city)
        unvisited.remove(best_next_city)
        total_cost += best_next_cost
        current_city = best_next_city

    # After visiting all cities, try to return to the start.
    cost_back = matrix[current_city][start_city_index]
    if cost_back is None:
        return None, float('inf')
    else:
        total_cost += cost_back
        route.append(start_city_index)

    # Convert the route city IDs to City objects and return.
    route_cities = [graph.cities[i] for i in route]
    return route_cities, total_cost
