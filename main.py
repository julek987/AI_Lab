from City import City, create_random_city
from Graph import Graph


#All direct connections and symmetrical cost:

city1 = create_random_city("CityA")
city2 = create_random_city("CityB")
city3 = create_random_city("CityC")
city4 = create_random_city("CityD")
city5 = create_random_city("CityE")

city_graph = Graph()
city_graph.add_city(city1)
city_graph.add_city(city2)
city_graph.add_city(city3)
city_graph.add_city(city4)
city_graph.add_city(city5)

city_graph.add_road(city1, city2, asymmetrical=False)
city_graph.add_road(city1, city3, asymmetrical=False)
city_graph.add_road(city1, city4, asymmetrical=False)
city_graph.add_road(city1, city5, asymmetrical=False)

city_graph.add_road(city2, city3, asymmetrical=False)
city_graph.add_road(city2, city4, asymmetrical=False)
city_graph.add_road(city2, city5, asymmetrical=False)


city_graph.add_road(city3, city4, asymmetrical=False)
city_graph.add_road(city3, city5, asymmetrical=False)


city_graph.add_road(city4, city5, asymmetrical=False)

# Display the graph
city_graph.display_graph()

class State:
    def __init__(self, current_city, visited, total_cost):
        self.current_city = current_city      # The city the salesman is currently in
        self.visited = visited                # The set of cities visited so far
        self.total_cost = total_cost          # The total travel cost so far

    def __repr__(self):
        return f"State(city={self.current_city.name}, visited={len(self.visited)}, cost={self.total_cost})"


from collections import deque


def bfs(graph, start_city):
    # Queue for BFS, containing the initial state (starting from start_city)
    initial_state = State(start_city, {start_city}, 0)
    queue = deque([initial_state])

    n = len(graph.graph)  # Total number of cities
    best_cost = float('inf')
    best_path = None

    while queue:
        current_state = queue.popleft()

        # If all cities have been visited, complete the cycle by returning to the start city
        if len(current_state.visited) == n:
            final_cost = current_state.total_cost + current_state.current_city.distance_to(start_city)
            if final_cost < best_cost:
                best_cost = final_cost
                best_path = current_state.visited
            continue

        # Expand the current node by visiting each unvisited city
        for neighbor, travel_cost in graph.graph[current_state.current_city]:
            if neighbor not in current_state.visited:
                new_visited = current_state.visited | {neighbor}
                new_total_cost = current_state.total_cost + travel_cost
                new_state = State(neighbor, new_visited, new_total_cost)
                queue.append(new_state)

    return best_path, best_cost


def dfs(graph, start_city):
    # Stack for DFS, containing the initial state (starting from start_city)
    initial_state = State(start_city, {start_city}, 0)
    stack = [initial_state]

    n = len(graph.graph)  # Total number of cities
    best_cost = float('inf')
    best_path = None

    while stack:
        current_state = stack.pop()

        # If all cities have been visited, complete the cycle by returning to the start city
        if len(current_state.visited) == n:
            final_cost = current_state.total_cost + current_state.current_city.distance_to(start_city)
            if final_cost < best_cost:
                best_cost = final_cost
                best_path = current_state.visited
            continue

        # Expand the current node by visiting each unvisited city
        for neighbor, travel_cost in graph.graph[current_state.current_city]:
            if neighbor not in current_state.visited:
                new_visited = current_state.visited | {neighbor}
                new_total_cost = current_state.total_cost + travel_cost
                new_state = State(neighbor, new_visited, new_total_cost)
                stack.append(new_state)

    return best_path, best_cost

best_path, best_cost = bfs(city_graph, city1)
print(f"BFS Best Path: {best_path}, Cost: {best_cost}")

best_path, best_cost = dfs(city_graph, city1)
print(f"DFS Best Path: {best_path}, Cost: {best_cost}")

