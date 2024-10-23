import math
import random

class City:
    def __init__(self, name, x, y, z):
        self.name = name
        self.x = x
        self.y = y
        self.z = z

    def distance_to(self, other_city, asymmetrical=False):

        # Euclidean distance on the (x, y) plane
        xy_distance = math.sqrt((other_city.x - self.x) ** 2 + (other_city.y - self.y) ** 2)

        if not asymmetrical:
            return xy_distance

        # Asymmetrical case: Adjust based on height difference (z-coordinate)
        if other_city.z > self.z:
            return xy_distance * 1.1
        else:
            return xy_distance * 0.9

    def __repr__(self):
        return f"{self.name} ({self.x}, {self.y}, {self.z})"



def create_random_city(name):
    x = random.randint(-100, 100)
    y = random.randint(-100, 100)
    z = random.randint(0, 50)
    return City(name, x, y, z)