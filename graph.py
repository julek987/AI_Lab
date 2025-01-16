import random
from typing import List, Optional

from city import City, calculate_cost


class Graph:

    def __init__(self,
                 cities: List[City],
                 scenario_connection: str = "complete", #scenario_connection: "complete" or "80_percent"
                 scenario_symmetry: bool = True): #scenario_symmetry: True = symmetrical cost, False = asymmetrical cost

        self.cities = cities
        self.n = len(cities)
        self.scenario_connection = scenario_connection
        self.scenario_symmetry = scenario_symmetry

        # Build the adjacency matrix on initialization
        self.adj_matrix = self.build_graph()

    def build_graph(self) -> List[List[Optional[float]]]:
        """
        Creates an n x n adjacency matrix where entry [i][j] is the cost of traveling
        from city i to city j, or None if there is no direct road.
        """
        # Initialize an n x n matrix filled with None
        adj_matrix = [[None for _ in range(self.n)] for _ in range(self.n)]

        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    # No self-loop
                    adj_matrix[i][j] = None
                else:
                    # Decide if there's a road-based on scenario
                    if self.scenario_connection == "complete":
                        road_exists = True
                    elif self.scenario_connection == "80_percent":
                        # 80% chance of having a direct road
                        road_exists = (random.random() < 0.8)
                    else:
                        road_exists = False

                    if road_exists:
                        cost_ij = calculate_cost(
                            self.cities[i],
                            self.cities[j],
                            self.scenario_symmetry
                        )
                        adj_matrix[i][j] = cost_ij
                    else:
                        adj_matrix[i][j] = None

        return adj_matrix

    def __repr__(self) -> str:
        return f"Graph with {self.n} cities. Connection: {self.scenario_connection}, Symmetry: {self.scenario_symmetry}"