import numpy as np

def objective_function(position):
    # Get x and y values from the position vector.
    x_val = position[0]
    y_val = position[1]
    # Compute each term of the function.
    term1 = (1.5 - x_val - x_val * y_val)**2
    term2 = (2.25 - x_val + x_val * y_val**2)**2
    term3 = (2.625 - x_val + x_val * y_val**3)**2
    # Return the sum of the three terms.
    return term1 + term2 + term3

class Particle:
    def __init__(self, bounds, dimensions):
        # Randomly initialize the particle's position within the bounds.
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        # Set the velocity range based on the bounds.
        velocity_range = bounds[1] - bounds[0]
        # Randomly initialize the particle's velocity.
        self.velocity = np.random.uniform(-velocity_range, velocity_range, dimensions)
        # Set the particle's personal best to its initial position.
        self.best_position = np.copy(self.position)
        # Evaluate the objective function at the initial position.
        self.best_value = objective_function(self.position)

    def update_velocity(self, global_best, inertia_weight, cognitive_constant, social_constant):
        # Generate random factors for the cognitive and social components.
        r1 = np.random.rand(*self.position.shape)
        r2 = np.random.rand(*self.position.shape)
        # Compute the cognitive component (attraction to personal best).
        cognitive_velocity = cognitive_constant * r1 * (self.best_position - self.position)
        # Compute the social component (attraction to global best).
        social_velocity = social_constant * r2 * (global_best - self.position)
        # Update the velocity based on inertia, cognitive, and social factors.
        self.velocity = inertia_weight * self.velocity + cognitive_velocity + social_velocity

    def update_position(self, bounds):
        # Update the particle's position based on its velocity.
        self.position = self.position + self.velocity
        # Keep the position within the specified bounds.
        self.position = np.clip(self.position, bounds[0], bounds[1])
        # Evaluate the new position.
        current_value = objective_function(self.position)
        # Update personal best if the new position is better.
        if current_value < self.best_value:
            self.best_value = current_value
            self.best_position = np.copy(self.position)

def pso(num_particles, dimensions, bounds, iterations,
        inertia_weight=0.7, cognitive_constant=2.0, social_constant=2.0):
    # Initialize the swarm of particles.
    swarm = [Particle(bounds, dimensions) for _ in range(num_particles)]

    # Set initial global best to a high value.
    global_best_value = float('inf')
    global_best_position = None
    # Find the best initial position in the swarm.
    for particle in swarm:
        if particle.best_value < global_best_value:
            global_best_value = particle.best_value
            global_best_position = np.copy(particle.best_position)

    # Main PSO loop.
    for iteration in range(iterations):
        for particle in swarm:
            # Update velocity based on personal and global best.
            particle.update_velocity(global_best_position, inertia_weight, cognitive_constant, social_constant)
            # Update position and personal best.
            particle.update_position(bounds)
            # Check if this particle has a new global best.
            current_value = objective_function(particle.position)
            if current_value < global_best_value:
                global_best_value = current_value
                global_best_position = np.copy(particle.position)

        # Print the current iteration and best value found.
        print(f"Iteration {iteration+1}/{iterations}, best value = {global_best_value:.6f}")

    # Return the best position and its objective function value.
    return global_best_position, global_best_value

if __name__ == "__main__":
    # PSO parameters for this problem.
    num_particles = 30       # Number of particles in the swarm.
    dimensions = 2           # We are optimizing two variables: x and y.
    bounds = (-4.5, 4.5)     # The search space for both x and y.
    iterations = 100         # Total number of iterations to run.

    best_position, best_value = pso(num_particles, dimensions, bounds, iterations,
                                    inertia_weight=0.7, cognitive_constant=2.0, social_constant=2.0)

    print("\nOptimization completed.")
    print("Best position found: x = {:.6f}, y = {:.6f}".format(best_position[0], best_position[1]))
    print("Best objective function value: {:.6f}".format(best_value))
