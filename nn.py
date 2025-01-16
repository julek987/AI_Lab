from graph import Graph

def nearest_neighbor(graph: Graph, start_city_index: int):

    n = graph.n
    adj = graph.adj_matrix

    unvisited = set(range(n))
    unvisited.remove(start_city_index)

    route = [start_city_index]
    current_city = start_city_index
    total_cost = 0.0

    while unvisited:
        best_next_city = None
        best_next_cost = float('inf')

        for city_id in unvisited:
            cost_edge = adj[current_city][city_id]
            if cost_edge is not None and cost_edge < best_next_cost:
                best_next_cost = cost_edge
                best_next_city = city_id

        if best_next_city is None:
            return None, float('inf')

        route.append(best_next_city)
        unvisited.remove(best_next_city)
        total_cost += best_next_cost
        current_city = best_next_city


    cost_back = adj[current_city][start_city_index]
    if cost_back is None:
        return None, float('inf')
    else:
        total_cost += cost_back
        route.append(start_city_index)

    route_cities = [graph.cities[i] for i in route]
    return route_cities, total_cost
