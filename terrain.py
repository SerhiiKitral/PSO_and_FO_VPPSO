import math
import random
import json


class Terrain:
    """Terrain model generating 3D mountain terrain as sum of Gaussian peaks."""

    def __init__(
        self,
        random_terrain_mode=True,
        x_range=40,
        y_range=40,
        peaks=None,
        num_peaks=6,
        seed=None,
    ):
        self.x_min = 0.0
        self.x_max = float(x_range)
        self.y_min = 0.0
        self.y_max = float(y_range)
        self.peaks = []

        if not random_terrain_mode:
            if isinstance(peaks, str):

                with open(peaks, "r") as f:
                    data = json.load(f)
                    self.x_max = data.get("x_range", x_range)
                    self.y_max = data.get("y_range", y_range)
                    self.peaks = data["peaks"]
            elif peaks is not None:
                self.peaks = peaks
            else:
                self.peaks = [
                    (2.5, 10.0 * i, 10.0 * i, 3.0, 3.0) for i in range(1, 4)
                ] + [
                    (0.1, 20.0, 20.0, 1.5, 1.5)  # small pass
                ]
        else:
            if seed is not None:
                random.seed(seed)
            for _ in range(num_peaks):
                amp = random.uniform(0.5, 2.0)
                px = random.uniform(0.1 * self.x_max, 0.9 * self.x_max)
                py = random.uniform(0.1 * self.y_max, 0.9 * self.y_max)
                sx = random.uniform(2.0, 5.0)
                sy = random.uniform(2.0, 5.0)
                self.peaks.append((amp, px, py, sx, sy))

        self.terrain_max_alt = (
            max([amp for (amp, _, _, _, _) in self.peaks]) if self.peaks else 0.0
        )

    def get_height(self, x, y):
        """Compute terrain height at coordinate (x, y) by summing Gaussian peaks"""
        # Ensure (x,y) is within environment range (though extrapolating beyond is fine as exp will be tiny).
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            # Outside defined range, return 0 height (flat outside boundaries)
            return 0.0
        z = 0.0
        # Equation (2): Zi = Ai * exp( -(((X - Xi)^2)/(2σ_xi^2) + ((Y - Yi)^2)/(2σ_yi^2)) )
        # Summing contributions of all Gaussian peaks to get terrain height.
        for amp, px, py, sx, sy in self.peaks:
            z += amp * math.exp(
                -(((x - px) ** 2) / (2 * (sx**2)) + ((y - py) ** 2) / (2 * (sy**2)))
            )
        return z
