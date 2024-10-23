class Graph:
    def __init__(self):
        # Graph is represented as an adjacency list: {City: [(Neighbor, Distance), ...]}
        self.graph = {}

    def add_city(self, city):
        # Add a city (node) to the graph
        if city not in self.graph:
            self.graph[city] = []

    def add_road(self, city1, city2, asymmetrical=False):
        # Add a directed edge (road) from city1 to city2 with a weight (distance)
        distance = city1.distance_to(city2, asymmetrical)
        self.graph[city1].append((city2, distance))

        # For undirected (bidirectional) roads, also add the reverse road
        if not asymmetrical:
            self.graph[city2].append((city1, distance))

    def display_graph(self):
        for city, neighbors in self.graph.items():
            print(f"{city}:")
            for neighbor, distance in neighbors:
                print(f"  -> {neighbor} (Cost: {distance})")