import random
import math
from typing import List

class City:
    def __init__(self, name: str, x: int, y: int, z: int):
        self.name = name
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"{self.name}(x={self.x}, y={self.y}, z={self.z})"


def generate_cities(n: int) -> List[City]:

    cities = []
    for i in range(n):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        z = random.randint(0, 50)
        name = f"City_{i+1}"
        cities.append(City(name, x, y, z))
    return cities


def calculate_cost(cityA: City, cityB: City, symmetrical: bool) -> float:

    base_distance = math.sqrt((cityB.x - cityA.x) ** 2 + (cityB.y - cityA.y) ** 2 + (cityB.z - cityA.z) ** 2)

    if symmetrical:
        return base_distance
    else:
        if cityB.z > cityA.z:
            # Uphill => +10%
            return base_distance * 1.1
        elif cityB.z < cityA.z:
            # Downhill => -10%
            return base_distance * 0.9
        else:
            # Same height
            return base_distance