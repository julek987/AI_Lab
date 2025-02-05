"""
pso_exercise2.py

Particle Swarm Optimization (PSO) implementation to minimize the function:
    f(x,y) = (1.5 - x - x*y)^2 + (2.25 - x + x*y**2)^2 + (2.625 - x + x*y**3)^2

The search domain for both x and y is [-4.5, 4.5].
"""

import numpy as np

def objective_function(position):
    """
    Compute the value of the objective function for a given 2D position.

    Parameters:
        position (np.ndarray): A 2-element array where
                               position[0] = x and position[1] = y.

    Returns:
        float: The function value.
    """
    x_val = position[0]
    y_val = position[1]
    term1 = (1.5 - x_val - x_val * y_val)**2
    term2 = (2.25 - x_val + x_val * y_val**2)**2
    term3 = (2.625 - x_val + x_val * y_val**3)**2
    return term1 + term2 + term3

class Particle:
    def __init__(self, bounds, dimensions):
        """
        Initialize a particle with random position and velocity.

        Parameters:
            bounds (tuple): A tuple (lower_bound, upper_bound) for each dimension.
            dimensions (int): The number of dimensions (here, 2).
        """
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        velocity_range = bounds[1] - bounds[0]
        self.velocity = np.random.uniform(-velocity_range, velocity_range, dimensions)
        self.best_position = np.copy(self.position)
        self.best_value = objective_function(self.position)

    def update_velocity(self, global_best, inertia_weight, cognitive_constant, social_constant):
        """
        Update the particle's velocity based on its own experience and that of the swarm.

        Parameters:
            global_best (np.ndarray): The best position found by the swarm.
            inertia_weight (float): Inertia factor.
            cognitive_constant (float): Cognitive acceleration coefficient.
            social_constant (float): Social acceleration coefficient.
        """
        r1 = np.random.rand(*self.position.shape)
        r2 = np.random.rand(*self.position.shape)

        cognitive_velocity = cognitive_constant * r1 * (self.best_position - self.position)
        social_velocity = social_constant * r2 * (global_best - self.position)

        self.velocity = inertia_weight * self.velocity + cognitive_velocity + social_velocity

    def update_position(self, bounds):
        """
        Update the particle's position based on its velocity.
        The new position is clamped to the given bounds.

        Parameters:
            bounds (tuple): (lower_bound, upper_bound) for each dimension.
        """
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])

        # Evaluate the new position and update the personal best if improved.
        current_value = objective_function(self.position)
        if current_value < self.best_value:
            self.best_value = current_value
            self.best_position = np.copy(self.position)

def pso(num_particles, dimensions, bounds, iterations,
        inertia_weight=0.7, cognitive_constant=2.0, social_constant=2.0):
    """
    Run the Particle Swarm Optimization algorithm.

    Parameters:
        num_particles (int): Number of particles in the swarm.
        dimensions (int): Number of dimensions (should be 2 for this problem).
        bounds (tuple): (lower_bound, upper_bound) for the search space.
        iterations (int): Number of iterations to run.
        inertia_weight (float): Inertia weight for velocity update.
        cognitive_constant (float): Cognitive constant.
        social_constant (float): Social constant.

    Returns:
        tuple: (global_best_position, global_best_value)
    """
    # Initialize swarm.
    swarm = [Particle(bounds, dimensions) for _ in range(num_particles)]

    # Initialize global best.
    global_best_value = float('inf')
    global_best_position = None
    for particle in swarm:
        if particle.best_value < global_best_value:
            global_best_value = particle.best_value
            global_best_position = np.copy(particle.best_position)

    # Main PSO loop.
    for iteration in range(iterations):
        for particle in swarm:
            particle.update_velocity(global_best_position, inertia_weight, cognitive_constant, social_constant)
            particle.update_position(bounds)

            current_value = objective_function(particle.position)
            if current_value < global_best_value:
                global_best_value = current_value
                global_best_position = np.copy(particle.position)

        print(f"Iteration {iteration+1}/{iterations}, best value = {global_best_value:.6f}")

    return global_best_position, global_best_value

if __name__ == "__main__":
    # PSO parameters for the problem.
    num_particles = 30       # Number of particles in the swarm.
    dimensions = 2           # We are optimizing over x and y.
    bounds = (-4.5, 4.5)       # Search space for both x and y.
    iterations = 100         # Number of iterations (generations) to run.

    best_position, best_value = pso(num_particles, dimensions, bounds, iterations,
                                    inertia_weight=0.7, cognitive_constant=2.0, social_constant=2.0)

    print("\nOptimization completed.")
    print("Best position found: x = {:.6f}, y = {:.6f}".format(best_position[0], best_position[1]))
    print("Best objective function value: {:.6f}".format(best_value))
