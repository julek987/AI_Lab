import random
import math
from typing import List, Tuple
from math import inf

from city import City
from graph import Graph


def aco_tsp(
    graph: Graph,
    num_ants: int = 10,
    num_iterations: int = 50,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation_rate: float = 0.5,
    q: float = 100.0
) -> Tuple[List[int], float]:
    """
    Approximate TSP solution using Ant Colony Optimization (ACO).

    :param graph: A Graph object (with .cities, .adj_matrix)
    :param num_ants: Number of ants (agents) in each iteration
    :param num_iterations: How many times we let all ants build routes
    :param alpha: Importance of pheromone
    :param beta: Importance of distance (visibility)
    :param evaporation_rate: Fraction of pheromone that evaporates each iteration
    :param q: Constant for depositing pheromone (pheromone added = Q / route_cost)
    :return: (best_route_indices, best_cost)
             best_route_indices is a list of city indices describing the route
             best_cost is the numeric total cost of that route
    """
    n = graph.n
    adj = graph.adj_matrix

    # 1) Initialize pheromone levels
    #    If 0 is too low, ants can't differentiate edges well early on. We pick a small positive value.
    initial_pheromone = 1.0
    pheromone = [[initial_pheromone for _ in range(n)] for _ in range(n)]

    # A utility function to build one route for a single ant
    def construct_solution(start_city: int) -> Tuple[List[int], float]:
        """
        Builds a route for one ant using the current pheromone matrix.
        Returns (route_indices, route_cost).
        """
        visited = set()
        visited.add(start_city)
        route = [start_city]
        current_city = start_city
        total_cost = 0.0

        while len(visited) < n:
            # Compute probabilities for all unvisited neighbors
            probabilities = []
            city_indices = []
            denominator = 0.0

            for next_city in range(n):
                if next_city not in visited and adj[current_city][next_city] is not None:
                    tau = pheromone[current_city][next_city] ** alpha
                    distance = adj[current_city][next_city]
                    eta = (1.0 / distance) ** beta if distance > 0 else 0

                    prob = tau * eta
                    denominator += prob
                    probabilities.append(prob)
                    city_indices.append(next_city)

            if not probabilities:
                # If we can't move anywhere (disconnected?), break
                # or return an incomplete route with infinite cost
                return route, inf

            # Normalize probabilities
            probabilities = [p / denominator for p in probabilities]

            # Choose the next city via roulette wheel
            next_city = random.choices(city_indices, weights=probabilities, k=1)[0]

            # Update route info
            route.append(next_city)
            visited.add(next_city)
            total_cost += adj[current_city][next_city]
            current_city = next_city

        # Finally, add the cost to return to start (TSP loop closure)
        if adj[current_city][start_city] is None:
            # No way back => incomplete
            return route, inf
        else:
            total_cost += adj[current_city][start_city]
            route.append(start_city)

        return route, total_cost

    best_route = None
    best_cost = inf

    # 2) Main ACO loop
    for iteration in range(num_iterations):
        # a) Each ant constructs a route
        ant_routes = []
        ant_costs = []

        for _ in range(num_ants):
            # You can choose random start or fixed start (say 0).
            # Here, let's pick a random start city for variety:
            start_city = random.randint(0, n - 1)
            route, cost_ = construct_solution(start_city)
            ant_routes.append(route)
            ant_costs.append(cost_)

            # Update best
            if cost_ < best_cost:
                best_cost = cost_
                best_route = route

        # b) Evaporate pheromones
        for i in range(n):
            for j in range(n):
                pheromone[i][j] *= (1.0 - evaporation_rate)

        # c) Deposit new pheromones
        for route, cost_ in zip(ant_routes, ant_costs):
            if cost_ == inf:
                # Skip deposit if route is incomplete
                continue
            deposit_amount = q / cost_
            # The route includes the final city returning to start,
            # so edges are consecutive pairs in route
            for k in range(len(route) - 1):
                city_i = route[k]
                city_j = route[k + 1]
                if adj[city_i][city_j] is not None:
                    pheromone[city_i][city_j] += deposit_amount
                    # If you want a symmetrical deposit in an asymmetrical graph, do:
                    # pheromone[city_j][city_i] += deposit_amount

    # Convert best_route from city indices to a simpler format
    # best_route might be None if no feasible path was found
    return best_route, best_cost