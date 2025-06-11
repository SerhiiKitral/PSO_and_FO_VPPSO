import random


# MATH_36:
# MATH_37:
class PSO:
    """Classical Particle Swarm Optimization (PSO) algorithm."""

    def __init__(
        self,
        dim,
        swarm_size=30,
        w=0.7,
        c1=1.5,
        c2=1.5,
        max_iter=100,
        bounds=None,
        fitness_func=None,
    ):
        """
        dim: dimensionality of the search space (length of position vector).
        swarm_size: number of particles (swarm size).
        w: inertia weight.
        c1: cognitive acceleration coefficient.
        c2: social acceleration coefficient.
        max_iter: maximum number of iterations.
        bounds: list of (min, max) for each dimension (or None to not clamp).
        fitness_func: function to evaluate fitness given a position vector.
        """
        self.dim = dim
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.bounds = bounds  # bounds should be list of tuples [(min_d, max_d), ...]
        self.fitness_func = fitness_func
        # Initialize particle positions and velocities
        self.positions = []
        self.velocities = []
        for _ in range(swarm_size):
            pos = []
            vel = []
            for d in range(dim):
                if bounds and bounds[d] is not None:
                    min_d, max_d = bounds[d]
                    # Random position in [min_d, max_d]
                    pd = random.uniform(min_d, max_d)
                    pos.append(pd)
                    # Random initial velocity in range [-|max-min|, |max-min|]
                    v_range = (
                        (max_d - min_d)
                        if (max_d is not None and min_d is not None)
                        else 1.0
                    )
                    vd = random.uniform(-v_range, v_range)
                    vel.append(vd)
                else:
                    pd = random.random()
                    pos.append(pd)
                    vd = random.uniform(-1.0, 1.0)
                    vel.append(vd)
            self.positions.append(pos)
            self.velocities.append(vel)
        # Initialize personal bests and global best
        self.pbest_positions = [list(p) for p in self.positions]
        self.pbest_values = (
            [self.fitness_func(p) for p in self.positions]
            if fitness_func
            else [float("inf")] * swarm_size
        )
        self.gbest_index = min(range(swarm_size), key=lambda i: self.pbest_values[i])
        self.gbest_position = list(self.pbest_positions[self.gbest_index])
        self.gbest_value = self.pbest_values[self.gbest_index]
        self.iteration = 0

    def step(self):
        """Perform one iteration (update velocities, positions, evaluate fitness)."""
        for i in range(self.swarm_size):
            # Update velocity for particle i (classical PSO formula)
            for d in range(self.dim):
                r1 = random.random()
                r2 = random.random()
                # PSO velocity update (Equation 38): v_{i,d}(t+1) = w*v_{i,d}(t) + c1*r1*(pbest_{i,d} - x_{i,d}(t)) + c2*r2*(gbest_d - x_{i,d}(t))
                self.velocities[i][d] = (
                    self.w * self.velocities[i][d]
                    + self.c1 * r1 * (self.pbest_positions[i][d] - self.positions[i][d])
                    + self.c2 * r2 * (self.gbest_position[d] - self.positions[i][d])
                )
            # MATH_38: implemented above (velocity update).
            for d in range(self.dim):
                self.positions[i][d] += self.velocities[i][
                    d
                ]  # x_{i,d}(t+1) = x_{i,d}(t) + v_{i,d}(t+1)
                # Clamp position within bounds
                if self.bounds and self.bounds[d] is not None:
                    min_d, max_d = self.bounds[d]
                    if self.positions[i][d] < min_d:
                        self.positions[i][d] = min_d
                        self.velocities[i][d] = 0.0
                    elif self.positions[i][d] > max_d:
                        self.positions[i][d] = max_d
                        self.velocities[i][d] = 0.0
            # MATH_39: implemented above (position update).
            # Evaluate fitness of new position
            fitness_val = (
                self.fitness_func(self.positions[i])
                if self.fitness_func
                else float("inf")
            )
            # Update personal best if this is an improvement
            if fitness_val < self.pbest_values[i]:
                self.pbest_values[i] = fitness_val
                self.pbest_positions[i] = list(self.positions[i])
            # Update global best if needed
            if fitness_val < self.gbest_value:
                self.gbest_value = fitness_val
                self.gbest_position = list(self.positions[i])
        self.iteration += 1
        # Print progress to console
        print(f"Iteration {self.iteration}: BestFitness = {self.gbest_value}")

    def run(self):
        print("Starting PSO optimization...")
        self.prev_gbest_value = self.gbest_value
        self.no_improve_counter = 0

        for it in range(self.max_iter):
            self.iteration = it
            self.step()
            if abs(self.prev_gbest_value - self.gbest_value) < 1e-3:
                self.no_improve_counter += 1
            else:
                self.no_improve_counter = 0
            self.prev_gbest_value = self.gbest_value
            if self.no_improve_counter >= 20:
                print("Early stopping triggered (no improvement)")
                break

        print(f"PSO completed. Best fitness = {self.gbest_value}")
