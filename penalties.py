from terrain import Terrain
import json
import numpy as np
from uav_model import UAV
from pso import PSO
from fo_vppso import FOVPPSO
from visualization import plot_swarm_paths
import math
import os
from datetime import datetime


def get_log_folder():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = f"logs/{timestamp}"
    os.makedirs(folder, exist_ok=True)
    return folder


def compute_fLE(uavs, vector, waypoints_per_uav):
    idx = 0
    fLE_total = 0.0
    for uav in uavs:
        wps = [
            (vector[idx + 3 * i], vector[idx + 3 * i + 1], vector[idx + 3 * i + 2])
            for i in range(waypoints_per_uav)
        ]
        fLE_total += uav.compute_length_cost(wps)
        idx += 3 * waypoints_per_uav
    return fLE_total


def compute_fTR(uavs, vector, terrain, Q, epsilon, waypoints_per_uav, A_max):
    idx = 0
    fTR_total = 0.0
    alt_cost_total = 0.0
    alt_penalty_const = 400.0
    alt_reward_const = 150.0
    delta = 0.01
    collision_penalty = 2e5
    routes = []

    for uav in uavs:
        wps = [
            (vector[idx + 3 * i], vector[idx + 3 * i + 1], vector[idx + 3 * i + 2])
            for i in range(waypoints_per_uav)
        ]
        fTR, avgR = uav.compute_terrain_cost(wps, terrain, Q, epsilon, return_avgR=True)

        # Strictly reject any terrain collision
        if fTR == float("inf"):
            return float("inf"), []

        avg_alt = sum(wp[2] for wp in wps) / len(wps)
        alt_penalty = alt_penalty_const * (avg_alt / A_max) ** 2
        if avg_alt > 2.5:
            alt_penalty += 400.0 * ((avg_alt - 2.5) ** 2)
        alt_reward = alt_reward_const * (1 / (avgR + delta)) if avgR else 0.0
        alt_cost = alt_penalty - alt_reward

        normalized_fTR = min(fTR / Q, 10.0)
        fTR_total += normalized_fTR + alt_cost
        routes.append(uav.get_route_points(wps))
        idx += 3 * waypoints_per_uav

    return fTR_total, routes


def compute_collision_penalty(routes, d_safe, P_co):
    fCO_list = [0.0] * len(routes)
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            for k in range(len(routes[0])):
                pi, pj = routes[i][k], routes[j][k]
                dist = math.dist(pi, pj)
                if dist <= d_safe:
                    fCO_list[i] += P_co
                    break
    return sum(fCO_list)


def compute_constraint_penalties(
    routes, l_min, l_max, psi_max, theta_max, iteration=0, max_iter=150
):
    penalty = 0.0
    scale_factor = min(2000.0, 500.0 * (1 + iteration / max_iter))

    for route in routes:
        wps_only = route[1:-1]
        for k in range(1, len(wps_only)):
            dx = wps_only[k][0] - wps_only[k - 1][0]
            backward_motion = max(-dx, 0.0)
            penalty += scale_factor * (backward_motion**2)

        for k in range(len(route) - 1):
            p1, p2 = route[k], route[k + 1]
            leg_dist = math.dist(p1, p2)
            short = max(l_min - leg_dist, 0.0)
            long = max(leg_dist - l_max, 0.0)
            penalty += scale_factor * (short**2 + long**2)

        for k in range(1, len(route) - 1):
            v1 = tuple(route[k][i] - route[k - 1][i] for i in range(3))
            v2 = tuple(route[k + 1][i] - route[k][i] for i in range(3))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(a * a for a in v2))
            if norm1 * norm2 == 0:
                continue
            dot_prod = sum(a * b for a, b in zip(v1, v2))
            angle = math.acos(max(-1.0, min(1.0, dot_prod / (norm1 * norm2))))
            turn_excess = max(angle - psi_max, 0.0)
            penalty += scale_factor * (turn_excess**2)

            horiz1 = math.hypot(v1[0], v1[1])
            horiz2 = math.hypot(v2[0], v2[1])
            theta1 = abs(math.atan2(v1[2], horiz1)) if horiz1 else math.pi / 2
            theta2 = abs(math.atan2(v2[2], horiz2)) if horiz2 else math.pi / 2
            delta_theta = abs(theta1 - theta2)
            theta_excess = max(delta_theta - theta_max, 0.0)
            penalty += 2 * scale_factor * (theta_excess**2)

    z_var_penalty = 25.0 * sum(np.var([p[2] for p in route]) for route in routes)

    # Smoothness penalty
    smooth_penalty = 0.0
    for route in routes:
        for k in range(1, len(route) - 1):
            prev = route[k - 1]
            curr = route[k]
            next = route[k + 1]

            dx1, dy1 = curr[0] - prev[0], curr[1] - prev[1]
            dx2, dy2 = next[0] - curr[0], next[1] - curr[1]
            dot = dx1 * dx2 + dy1 * dy2
            norm1 = math.hypot(dx1, dy1)
            norm2 = math.hypot(dx2, dy2)
            if norm1 > 0 and norm2 > 0:
                angle = math.acos(max(-1.0, min(1.0, dot / (norm1 * norm2))))
                smooth_penalty += 400.0 * (angle**2)

            dz1 = curr[2] - prev[2]
            dz2 = next[2] - curr[2]
            delta_dz = dz2 - dz1
            smooth_penalty += 100.0 * (delta_dz**2)

    return penalty + z_var_penalty + smooth_penalty


def evaluate_solution(vector):
    fTR, routes = compute_fTR(
        uavs, vector, terrain, Q, epsilon, waypoints_per_uav, A_max
    )
    if fTR == float("inf"):
        return float("inf")

    fLE = compute_fLE(uavs, vector, waypoints_per_uav)
    fCO = compute_collision_penalty(routes, d_safe, P_co)
    iteration = getattr(optimizer, "iteration", 0)
    penalty = compute_constraint_penalties(
        routes, l_min, l_max, psi_max, theta_max, iteration=iteration, max_iter=max_iter
    )
    fitness = fLE + fTR + fCO + penalty

    if (
        optimizer is not None
        and hasattr(evaluate_solution, "log_fd")
        and evaluate_solution.log_fd
    ):
        evaluate_solution.log_fd.write(
            f"{optimizer.iteration},{fitness},{fLE},{fTR},{fCO},{penalty}\n"
        )
        evaluate_solution.log_fd.flush()

    return fitness


def run_with_args_from_ui(values):
    global terrain, uavs, Q, epsilon, A_max, d_safe, P_co, l_min, l_max, psi_max, theta_max, waypoints_per_uav, max_iter, optimizer

    use_FO_VPPSO = values["opt_alg"] == "FO-VPPSO"
    terrain_mode = values["terrain_mode"] == "Random"

    num_uavs = values["num_uavs"]
    waypoints_per_uav = values["waypoints_per_uav"]
    max_iter = values["max_iter"]
    swarm_size = values["swarm_size"]
    A_max = values["a_max"]
    P_co = values["p_co"]
    Q = values["q"]
    epsilon = values["epsilon"]
    d_safe = values["d_safe"]
    l_min = values["l_min"]
    l_max = values["l_max"]
    psi_max = values["psi_max"]
    theta_max = values["theta_max"]
    all_paths_log = {}

    # Initialize terrain
    terrain = Terrain(random_terrain_mode=terrain_mode, x_range=40, y_range=40)
    # Define UAV start and goal positions
    uavs = []
    start_y_positions = [5.0 * (i + 1) for i in range(num_uavs)]
    end_y_positions = list(reversed(start_y_positions))
    for i in range(num_uavs):
        start = (0.0, start_y_positions[i], 0.5)
        goal = (40.0, end_y_positions[i], 0.5)
        uav = UAV(i, start, goal)
        uavs.append(uav)
    # Set up search space bounds for each waypoint coordinate
    dim = num_uavs * waypoints_per_uav * 3
    bounds = []
    for i in range(num_uavs):
        for j in range(waypoints_per_uav):
            bounds.append((terrain.x_min, terrain.x_max))
            bounds.append((terrain.y_min, terrain.y_max))
            bounds.append((0.0, A_max))  # MATH_11: (altitude constraint: A_i < A_max)

    # Choose optimizer and run optimization
    optimizer = None
    if not use_FO_VPPSO:
        optimizer = PSO(
            dim=dim,
            swarm_size=swarm_size,
            max_iter=max_iter,
            bounds=bounds,
            fitness_func=evaluate_solution,
        )
    else:
        optimizer = FOVPPSO(
            dim=dim,
            swarm_size=swarm_size,
            max_iter=max_iter,
            bounds=bounds,
            fitness_func=evaluate_solution,
        )
    log_folder = get_log_folder()

    evaluate_solution.log_fd = open(os.path.join(log_folder, "log.csv"), "w")
    evaluate_solution.log_fd.write("Iteration,TotalFitness,fLE,fTR,fCO,Penalty,Beta\n")
    # Run the optimization
    for i in range(optimizer.max_iter):

        paths_dict = {}
        idx = 0
        for uav_idx, uav in enumerate(uavs):
            x_list, y_list, z_list = [], [], []
            for _ in range(waypoints_per_uav):
                x = optimizer.gbest_position[idx]
                y = optimizer.gbest_position[idx + 1]
                z = optimizer.gbest_position[idx + 2]
                x_list.append(x)
                y_list.append(y)
                z_list.append(z)
                idx += 3
            full_route = uav.get_route_points(list(zip(x_list, y_list, z_list)))
            x_all = [pt[0] for pt in full_route]
            y_all = [pt[1] for pt in full_route]
            z_all = [pt[2] for pt in full_route]
            paths_dict[f"drone{uav_idx+1}"] = {"x": x_all, "y": y_all, "z": z_all}

        all_paths_log[f"iteration_{i}"] = paths_dict

        optimizer.step()

        fLE = compute_fLE(uavs, optimizer.gbest_position, waypoints_per_uav)
        fTR, routes = compute_fTR(
            uavs,
            optimizer.gbest_position,
            terrain,
            Q,
            epsilon,
            waypoints_per_uav,
            A_max,
        )
        fCO = compute_collision_penalty(routes, d_safe, P_co)
        penalty = compute_constraint_penalties(routes, l_min, l_max, psi_max, theta_max)
        total = fLE + fTR + fCO + penalty

        beta = getattr(optimizer, "beta", "")  # Empty if PSO

        evaluate_solution.log_fd.write(
            f"{i},{total},{fLE},{fTR},{fCO},{penalty},{beta}\n"
        )
        evaluate_solution.log_fd.flush()

    evaluate_solution.log_fd.close()
    # Retrieve best solution and display results
    best_vector = optimizer.gbest_position
    best_value = optimizer.gbest_value
    # print(f"Best solution fitness: {best_value}")
    # Construct paths for visualization and output
    best_paths = []
    idx = 0
    for ui, uav in enumerate(uavs):
        wps = []
        for w in range(waypoints_per_uav):
            x = best_vector[idx]
            y = best_vector[idx + 1]
            z = best_vector[idx + 2]
            wps.append((x, y, z))
            idx += 3
        best_paths.append(uav.get_route_points(wps))
        # print(f"UAV{ui+1} path: {best_paths[-1]}")
    # Visualize the terrain and paths
    plot_swarm_paths(terrain, best_paths, os.path.join(log_folder, "plot.html"))

    with open(os.path.join(log_folder, "path.csv"), "w") as f:
        num_waypoints = len(best_paths[0])
        header = []
        for i in range(len(best_paths)):
            header.extend([f"d{i+1}_x", f"d{i+1}_y", f"d{i+1}_z"])
        f.write(",".join(header) + "\n")

        for wp_idx in range(num_waypoints):
            row = []
            for drone_path in best_paths:
                x, y, z = drone_path[wp_idx]
                row.extend([f"{x}", f"{y}", f"{z}"])
            f.write(",".join(row) + "\n")

    terrain_data = {
        "x_range": terrain.x_max,
        "y_range": terrain.y_max,
        "peaks": terrain.peaks,
    }
    with open(os.path.join(log_folder, "terrain.json"), "w") as f:
        json.dump(terrain_data, f, indent=2)
    with open(os.path.join(log_folder, "run_parameters.json"), "w") as f:
        json.dump(values, f, indent=2)
    with open(os.path.join(log_folder, "paths_log.json"), "w") as pf:
        json.dump(all_paths_log, pf, indent=2)
