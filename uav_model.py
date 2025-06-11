import math


class UAV:
    """Represent a UAV with start and goal positions."""

    def __init__(self, uid, start, goal):
        self.id = uid
        self.start = tuple(start)
        self.goal = tuple(goal)

    def get_route_points(self, waypoints):
        """Get full route points (including start and goal) given intermediate waypoints."""
        route = [self.start]
        route.extend([tuple(wp) for wp in waypoints])
        route.append(self.goal)
        return route

    def compute_length_cost(self, waypoints):
        """Compute path length cost fLE as sum of distances of each leg"""
        route = self.get_route_points(waypoints)
        total_length = 0.0
        # Sum distance for each leg i (from point i to i+1)
        for i in range(len(route) - 1):
            x1, y1, z1 = route[i]
            x2, y2, z2 = route[i + 1]
            # Distance in 3D: sqrt(dx^2 + dy^2 + dz^2)
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            leg_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            total_length += leg_dist
        # fLE = sum_{i=1}^{n} l_i (Equation 4)
        return total_length  # This is the sum of leg lengths (cost proportional to route length).

    def compute_terrain_cost(self, waypoints, terrain, Q, epsilon, return_avgR=False):
        """
        Compute terrain cost fTR for this UAV's path
        fTR = Q/avgR if path does not go through mountains, otherwise epsilon
        avgR = (sum of distances from terrain at each leg) / n
        """
        route = self.get_route_points(waypoints)
        n_legs = len(route) - 1  # number of legs = n
        # Check if path goes through mountains (if UAV altitude goes below terrain height along any leg)
        path_through = False
        distances = []  # distances from terrain at waypoints (and perhaps midpoints)
        # We will sample each leg to ensure it does not dip below terrain.
        samples_per_leg = 10
        for i in range(len(route) - 1):
            x1, y1, z1 = route[i]
            x2, y2, z2 = route[i + 1]
            # Sample along segment to check clearance
            for t in range(samples_per_leg + 1):
                frac = t / samples_per_leg
                x = x1 + frac * (x2 - x1)
                y = y1 + frac * (y2 - y1)
                z = z1 + frac * (z2 - z1)
                terrain_h = terrain.get_height(x, y)
                if z <= terrain_h:
                    penetration = terrain_h - z
                    distances.append(-penetration)
                    path_through = True
                    break
            if path_through:
                break
            # Distance from terrain at end of this leg (point i+1)
            x_end, y_end, z_end = route[i + 1]
            dist_to_terrain = z_end - terrain.get_height(x_end, y_end)
            distances.append(dist_to_terrain)
        if path_through:
            penalty = epsilon + 1000.0 * abs(min(distances))
            if return_avgR:
                return float("inf"), None  # hard rejection
            else:
                return float("inf")
        # If no penetration, compute avgR = average vertical distance to terrain
        if distances:
            avg_R = sum(distances) / len(distances)
        else:
            avg_R = float("inf")
        if avg_R <= 0:
            # If somehow avg_R is 0 (which would imply UAV is at terrain level), treat as through (penalty).
            return epsilon
        # fTR = Q / avg_R for path not through mountains
        ftr = Q / avg_R

        if return_avgR:
            return ftr, avg_R
        else:
            return ftr
