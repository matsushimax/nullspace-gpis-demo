import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obstacle_database.boundary_miner import BoundaryMiner, nearest_opposite_pairs
from obstacle_database.database_io import write_collision_database
from obstacle_database.spatial_hash import WorkspaceHash
from obstacle_database.workspace_grid import WorkspaceGrid
from robot.planar3r import Planar3RRobot


def build_self_collision_dataset(robot, q, labels, max_pairs, tolerance):
    miner = BoundaryMiner(tolerance=tolerance)
    rows = {
        "q": [],
        "label": [],
        "boundary_q": [],
        "signed_distance": [],
        "gradient_q": [],
    }
    pairs = nearest_opposite_pairs(q, labels, max_pairs=max_pairs)
    for free_id, collision_id in pairs:
        q_free = q[free_id]
        q_collision = q[collision_id]
        boundary = miner.find_boundary(q_free, q_collision, robot.self_collision)
        for sample_id in (free_id, collision_id):
            label = bool(labels[sample_id])
            signed_distance, gradient = miner.signed_distance_and_gradient(q[sample_id], boundary, label)
            rows["q"].append(q[sample_id])
            rows["label"].append(int(label))
            rows["boundary_q"].append(boundary)
            rows["signed_distance"].append(signed_distance)
            rows["gradient_q"].append(gradient)
    return finalize_dataset(rows)


def build_external_collision_dataset(robot, q, workspace_hash, args):
    rng = np.random.default_rng(args.seed + 1009)
    miner = BoundaryMiner(tolerance=args.boundary_tolerance)
    workspace_min, workspace_max = robot.workspace_bounds(margin=0.0)
    rows = {
        "q": [],
        "obstacle_center": [],
        "obstacle_radius": [],
        "label": [],
        "boundary_q": [],
        "signed_distance": [],
        "gradient_q": [],
    }
    attempts = 0
    while len(rows["q"]) < args.external_dataset_size and attempts < args.external_queries:
        attempts += 1
        center = rng.uniform(workspace_min, workspace_max)
        radius = rng.uniform(args.obstacle_radius_min, args.obstacle_radius_max)
        candidate_ids = workspace_hash.query_circle(center, radius + robot.circle_radius)
        if len(candidate_ids) == 0:
            continue

        collision_ids = [
            int(i)
            for i in candidate_ids
            if robot.external_collision(q[int(i)], center, radius)
        ]
        if not collision_ids:
            continue

        free_ids = [
            int(i)
            for i in candidate_ids
            if not robot.external_collision(q[int(i)], center, radius)
        ]
        random_trials = 0
        while len(free_ids) < min(len(collision_ids), 4) and random_trials < 200:
            rid = int(rng.integers(0, len(q)))
            if not robot.external_collision(q[rid], center, radius):
                free_ids.append(rid)
            random_trials += 1
        if not free_ids:
            continue

        for collision_id in collision_ids[: args.external_pairs_per_obstacle]:
            q_collision = q[collision_id]
            free_distances = np.linalg.norm(q[np.asarray(free_ids)] - q_collision[None, :], axis=1)
            free_id = free_ids[int(np.argmin(free_distances))]
            q_free = q[free_id]
            collision_fn = lambda q_test, c=center, r=radius: robot.external_collision(q_test, c, r)
            boundary = miner.find_boundary(q_free, q_collision, collision_fn)
            for sample_id in (free_id, collision_id):
                label = robot.external_collision(q[sample_id], center, radius)
                signed_distance, gradient = miner.signed_distance_and_gradient(q[sample_id], boundary, label)
                rows["q"].append(q[sample_id])
                rows["obstacle_center"].append(center)
                rows["obstacle_radius"].append(radius)
                rows["label"].append(int(label))
                rows["boundary_q"].append(boundary)
                rows["signed_distance"].append(signed_distance)
                rows["gradient_q"].append(gradient)
                if len(rows["q"]) >= args.external_dataset_size:
                    break
            if len(rows["q"]) >= args.external_dataset_size:
                break
    return finalize_dataset(rows)


def finalize_dataset(rows):
    out = {}
    for key, values in rows.items():
        if not values:
            out[key] = np.empty((0,), dtype=np.float32)
            continue
        arr = np.asarray(values)
        if key == "label":
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.float32)
        out[key] = arr
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate a Planar 3R collision dataset database.")
    parser.add_argument("--output", default="outputs/planar3r_collision_database.h5")
    parser.add_argument("--num-configurations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell-size", type=float, default=0.05)
    parser.add_argument("--link-radius", type=float, default=0.06)
    parser.add_argument("--circle-radius", type=float, default=0.06)
    parser.add_argument("--circle-spacing", type=float, default=0.08)
    parser.add_argument("--self-boundary-pairs", type=int, default=1000)
    parser.add_argument("--external-queries", type=int, default=1000)
    parser.add_argument("--external-dataset-size", type=int, default=1000)
    parser.add_argument("--external-pairs-per-obstacle", type=int, default=2)
    parser.add_argument("--obstacle-radius-min", type=float, default=0.0)
    parser.add_argument("--obstacle-radius-max", type=float, default=0.15)
    parser.add_argument("--boundary-tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    start = time.perf_counter()
    robot = Planar3RRobot(
        link_radius=args.link_radius,
        circle_radius=args.circle_radius,
        circle_spacing=args.circle_spacing,
    )
    q = robot.sample_configurations(args.num_configurations, seed=args.seed)
    labels = robot.self_collision_batch(q)
    circle_centers, circle_radii = robot.collision_circles_batch(q)
    workspace_min, workspace_max = robot.workspace_bounds(margin=args.circle_radius)
    grid = WorkspaceGrid(workspace_min, workspace_max, args.cell_size)
    workspace_hash = WorkspaceHash(grid).build(circle_centers, circle_radii)

    self_dataset = build_self_collision_dataset(
        robot,
        q,
        labels,
        max_pairs=args.self_boundary_pairs,
        tolerance=args.boundary_tolerance,
    )
    external_dataset = build_external_collision_dataset(robot, q, workspace_hash, args)

    metadata = {
        "version": "0.1",
        "robot_name": "Planar3RRobot",
        "link_lengths": robot.link_lengths.tolist(),
        "joint_limits": robot.joint_limits.tolist(),
        "link_radius": robot.link_radius,
        "circle_radius": robot.circle_radius,
        "circle_spacing": robot.circle_spacing,
        "num_configurations": int(len(q)),
        "workspace_min": workspace_min.tolist(),
        "workspace_max": workspace_max.tolist(),
        "workspace_cell_size": args.cell_size,
    }
    write_collision_database(
        args.output,
        metadata=metadata,
        q=q,
        self_labels=labels,
        circle_centers=circle_centers,
        circle_radii=circle_radii,
        workspace_hash=workspace_hash,
        self_dataset=self_dataset,
        external_dataset=external_dataset,
    )

    elapsed = time.perf_counter() - start
    print(f"database: {Path(args.output).resolve()}")
    print(f"configurations: {len(q)}")
    print(f"self collision: {int(np.sum(labels))} collision / {int(len(labels) - np.sum(labels))} free")
    print(f"circles per config: {robot.num_circles}")
    print(f"hash cells: {len(workspace_hash.cell_keys)}")
    print(f"hash links: {len(workspace_hash.configuration_ids)}")
    print(f"self dataset rows: {len(self_dataset.get('q', []))}")
    print(f"external dataset rows: {len(external_dataset.get('q', []))}")
    print(f"elapsed: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
