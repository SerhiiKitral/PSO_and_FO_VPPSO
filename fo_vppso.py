import random
import math


class FOVPPSO:
    """
    Fractional-Order Velocity-Pausing PSO (FO-VPPSO) algorithm.
    Combines velocity pausing, two-swarm strategy, and fractional order update.
    """

    def __init__(
        self,
        dim,
        swarm_size=30,
        c1=1.5,
        c2=1.5,
        max_iter=100,
        bounds=None,
        fitness_func=None,
        alpha=0.1,
        b=5.0,
        min_beta=0.1,
    ):
        """
        dim: dimensionality of search space.
        swarm_size: total number of particles (will be split into two sub-swarms).
        c1, c2: cognitive and social coefficients.
        max_iter: maximum iterations.
        bounds: list of (min, max) for each dimension.
        fitness_func: evaluation function.
        alpha: velocity pausing probability
        b: parameter for adaptive factor a(t) = exp(-(b * t / T))
        min_beta: minimum fractional order value β
        """
        self.dim = dim
        self.swarm_size = swarm_size
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.bounds = bounds
        self.fitness_func = fitness_func
        self.alpha = alpha  # velocity pausing probability
        self.b = b  # parameter for adaptive factor a(t)
        self.min_beta = min_beta
        # Partition swarm into two groups (Group A and B)
        self.num_groupA = swarm_size // 2
        self.num_groupB = swarm_size - self.num_groupA
        # Initialize positions and velocities
        self.positions = []
        self.velocities = []
        for _ in range(swarm_size):
            pos = []
            vel = []
            for d in range(dim):
                if bounds and bounds[d] is not None:
                    min_d, max_d = bounds[d]
                    pd = random.uniform(min_d, max_d)
                    pos.append(pd)
                    v_range = (
                        (max_d - min_d)
                        if (max_d is not None and min_d is not None)
                        else 1.0
                    )
                    vd = random.uniform(-v_range, v_range)
                    vel.append(vd)
                else:
                    pos.append(random.random())
                    vel.append(random.uniform(-1.0, 1.0))
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
        # Fractional order β initial value
        self.beta = 1.0  # start with 1 (no fractional effect initially)
        # Initialize metrics for adaptive β
        self.weighted_avg_improvement = 0.0
        self.weighted_conv_rate = 0.0
        self.iteration = 0
        # print("Initialized FOVPPSO with two-swarm strategy.")

    def step(self):
        """Perform one iteration of FO-VPPSO updates."""
        T = self.max_iter
        t = self.iteration
        # Compute adaptive factor a(t) = exp(-(b * t / T))
        a_t = math.exp(-(self.b * float(t) / float(T)))
        # Update each particle
        for i in range(self.swarm_size):
            if i < self.num_groupA:
                # Group A: update velocity using fractional-order PSO (Equation 46)
                for d in range(self.dim):
                    r6 = random.random()
                    r7 = random.random()
                    base_velocity = self.velocities[i][d]
                    # Determine effective inertia scaling (velocity pausing)
                    if random.random() < self.alpha:
                        local_a = 1.0  # pause velocity damping (w=1 equivalent) # MATH_42: (velocity pausing)
                    else:
                        local_a = a_t
                    # Fractional velocity update: v_new = (v_old^β * a(t)) + c1*r6*(pbest_i - x_i) + c2*r7*(gbest - x_i)
                    # Apply sign-preserving power for v_old^β to handle negative velocities.
                    sign = 1.0 if base_velocity >= 0 else -1.0
                    vel_abs_beta = (abs(base_velocity) ** self.beta) * sign
                    self.velocities[i][d] = (
                        vel_abs_beta * local_a
                        + self.c1
                        * r6
                        * (self.pbest_positions[i][d] - self.positions[i][d])
                        + self.c2 * r7 * (self.gbest_position[d] - self.positions[i][d])
                    )
                # MATH_46: implemented above (velocity update for FO-VPPSO).
                for d in range(self.dim):
                    self.positions[i][d] += self.velocities[i][d]
                    # Clamp within bounds
                    if self.bounds and self.bounds[d] is not None:
                        min_d, max_d = self.bounds[d]
                        if self.positions[i][d] < min_d:
                            self.positions[i][d] = min_d
                            self.velocities[i][d] = 0.0
                        elif self.positions[i][d] > max_d:
                            self.positions[i][d] = max_d
                            self.velocities[i][d] = 0.0
                # MATH_47: implemented above (position update for FO-VPPSO).
            else:
                # Group B: update position based solely on global best (two-swarm strategy, Equation 44)
                new_pos = []
                for d in range(self.dim):
                    r8 = random.random()
                    r9 = random.random()
                    r10 = random.random()
                    if r9 < 0.5:
                        # X_i = gbest + a(t)*r8*|gbest|^{a(t)}
                        new_val = self.gbest_position[d] + a_t * r8 * (
                            (abs(self.gbest_position[d])) ** a_t
                        )
                    else:
                        # X_i = gbest - a(t)*r10*|gbest|^{a(t)}
                        new_val = self.gbest_position[d] - a_t * r10 * (
                            (abs(self.gbest_position[d])) ** a_t
                        )
                    # Clamp new position within bounds
                    if self.bounds and self.bounds[d] is not None:
                        min_d, max_d = self.bounds[d]
                        if new_val < min_d:
                            new_val = min_d
                        elif new_val > max_d:
                            new_val = max_d
                    new_pos.append(new_val)
                # Replace position directly; velocity is not used for group B
                self.positions[i] = new_pos
                self.velocities[i] = [0.0] * self.dim
        # Evaluate fitness for all particles
        fitness_vals = [
            self.fitness_func(self.positions[i]) if self.fitness_func else float("inf")
            for i in range(self.swarm_size)
        ]
        # Update personal best and global best
        for i in range(self.swarm_size):
            # Update Pbest_i if improved
            if fitness_vals[i] < self.pbest_values[i] and fitness_vals[i] != float(
                "inf"
            ):
                self.pbest_values[i] = fitness_vals[i]
                self.pbest_positions[i] = list(self.positions[i])
            # Update gbest if improved
            if fitness_vals[i] < self.gbest_value and fitness_vals[i] != float("inf"):
                self.gbest_value = fitness_vals[i]
                self.gbest_position = list(self.positions[i])

        if self.iteration > 0:
            prev_best = self.prev_gbest_value
            curr_best = self.gbest_value
            improvement = max(0.0, prev_best - curr_best)

            self.weighted_avg_improvement = (
                0.5 * self.weighted_avg_improvement + 0.5 * improvement
            )
            conv_change = abs(improvement - getattr(self, "prev_improvement", 0.0))
            self.weighted_conv_rate = 0.5 * self.weighted_conv_rate + 0.5 * conv_change
            self.prev_improvement = improvement

            threshold = 1e-3 * max(1.0, abs(self.gbest_value))
            raw_beta = (
                3.0
                - 0.5 * (self.weighted_avg_improvement / threshold)
                - 0.5 * self.weighted_conv_rate
            )
            raw_beta = max(self.min_beta, min(3.0, raw_beta))

            self.beta = 0.7 * self.beta + 0.3 * raw_beta

        # MATH_45 implemented above (adaptive fractional order β update).
        # Save current best for next iteration
        self.prev_gbest_value = self.gbest_value
        self.iteration += 1
        # print(
        #     f"Iteration {self.iteration}: BestFitness = {self.gbest_value}, beta = {self.beta}"
        # )

    def run(self):
        # print("Starting FO-VPPSO optimization...")
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
                # print("Early stopping triggered (no improvement)")
                break

        # print(f"FO-VPPSO completed. Best fitness = {self.gbest_value}")
