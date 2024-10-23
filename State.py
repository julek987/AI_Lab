class State:
    def __init__(self, current_city, visited, total_cost):
        self.current_city = current_city      # The city the salesman is currently in
        self.visited = visited                # The set of cities visited so far
        self.total_cost = total_cost          # The total travel cost so far

    def __repr__(self):
        return f"State(city={self.current_city.name}, visited={len(self.visited)}, cost={self.total_cost})"
