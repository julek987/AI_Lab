import random
from typing import List, Tuple
from math import inf

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
    # Get the number of cities and the cost matrix from the graph.
    n = graph.n
    adj = graph.adj_matrix

    # Initialize pheromone levels on each edge.
    initial_pheromone = 1.0
    pheromone = [[initial_pheromone for _ in range(n)] for _ in range(n)]

    def construct_solution(start_city: int) -> Tuple[List[int], float]:
        # Build a route starting from a given city.
        visited = set()
        visited.add(start_city)
        route = [start_city]
        current_city = start_city
        total_cost = 0.0

        # Continue until all cities are visited.
        while len(visited) < n:
            probabilities = [] # Probabilities for moving to each candidate city.
            city_ids = [] # Candidate city IDs.
            denominator = 0.0 # Sum of probability factors for normalization.

            # Compute probability factors for each unvisited city.
            for next_city in range(n):
                if next_city not in visited and adj[current_city][next_city] is not None:
                    # Pheromone factor raised to alpha.
                    tau = pheromone[current_city][next_city] ** alpha
                    distance = adj[current_city][next_city]
                    # Inverse distance factor (eta) raised to beta.
                    eta = (1.0 / distance) ** beta if distance > 0 else 0

                    prob = tau * eta  # Combined probability factor.
                    denominator += prob
                    probabilities.append(prob)
                    city_ids.append(next_city)

            # If no candidate city is found, return failure.
            if not probabilities:
                return route, inf

            # Normalize probabilities.
            probabilities = [p / denominator for p in probabilities]

            # Randomly select the next city based on computed probabilities.
            next_city = random.choices(city_ids, weights=probabilities, k=1)[0]

            # Update the route, visited set, and total cost.
            route.append(next_city)
            visited.add(next_city)
            total_cost += adj[current_city][next_city]
            current_city = next_city

        # After visiting all cities, return to the start.
        if adj[current_city][start_city] is None:
            return route, inf
        else:
            total_cost += adj[current_city][start_city]
            route.append(start_city)

        return route, total_cost

    best_route = None  # Store the best route found.
    best_cost = inf    # Store the lowest cost found.

    # Run the ACO algorithm for a fixed number of iterations.
    for iteration in range(num_iterations):
        ant_routes = []  # List to store each ant's route.
        ant_costs = []   # List to store each ant's route cost.

        # Each ant constructs a solution.
        for _ in range(num_ants):
            start_city = random.randint(0, n - 1)  # Random start city for this ant.
            route, cost_ = construct_solution(start_city)
            ant_routes.append(route)
            ant_costs.append(cost_)

            # Update best route if this ant found a better one.
            if cost_ < best_cost:
                best_cost = cost_
                best_route = route

        # Evaporate pheromone on all edges.
        for i in range(n):
            for j in range(n):
                pheromone[i][j] *= (1.0 - evaporation_rate)

        # Deposit new pheromone based on each ant's route quality.
        for route, cost_ in zip(ant_routes, ant_costs):
            if cost_ == inf:
                continue  # Skip if the route is incomplete.
            deposit_amount = q / cost_
            for k in range(len(route) - 1):
                city_i = route[k]
                city_j = route[k + 1]
                if adj[city_i][city_j] is not None:
                    pheromone[city_i][city_j] += deposit_amount

    return best_route, best_cost
