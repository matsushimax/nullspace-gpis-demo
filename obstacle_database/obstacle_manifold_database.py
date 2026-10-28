import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from database import compute_cell_indices, default_workspace_bounds, generate_target_grid
from obstacle_database.boundary_miner import BoundaryMiner, nearest_opposite_pairs
from planar_arm import PlanarArm3DOF
from robot.planar3r import Planar3RRobot


@dataclass(slots=True)
class ObstacleManifold:
    obstacle_id: int
    center: np.ndarray
    radius: float
    boundary_q: np.ndarray
    source_free_q: np.ndarray
    source_collision_q: np.ndarray


def load_nullspace_grid(nullspace_database_path):
    with h5py.File(nullspace_database_path, "r") as h5:
        return {
            "centers": h5["targets/positions"][:],
            "cell_indices": h5["targets/cell_indices"][:],
            "valid": h5["targets/valid"][:].astype(bool),
            "workspace_min": h5["meta"].attrs["workspace_min"].astype(float),
            "workspace_max": h5["meta"].attrs["workspace_max"].astype(float),
            "grid_shape": tuple(int(v) for v in h5["meta"].attrs["grid_shape"]),
            "cell_size": h5["meta"].attrs["cell_size"].astype(float),
        }


def make_grid_from_args(grid_shape=(20, 20), workspace_min=None, workspace_max=None):
    arm = PlanarArm3DOF(link_lengths=(1.0, 1.0, 0.7))
    if workspace_min is None or workspace_max is None:
        workspace_min, workspace_max = default_workspace_bounds(arm)
    else:
        workspace_min = np.asarray(workspace_min, dtype=float)
        workspace_max = np.asarray(workspace_max, dtype=float)
    centers = generate_target_grid(workspace_min, workspace_max, grid_shape)
    cell_size = (workspace_max - workspace_min) / np.asarray(grid_shape, dtype=float)
    cell_indices = compute_cell_indices(centers, workspace_min, cell_size, grid_shape)
    valid = np.linalg.norm(centers, axis=1) <= np.sum(arm.link_lengths)
    return {
        "centers": centers,
        "cell_indices": cell_indices,
        "valid": valid,
        "workspace_min": workspace_min,
        "workspace_max": workspace_max,
        "grid_shape": tuple(grid_shape),
        "cell_size": cell_size,
    }


class ObstacleManifoldDatabaseBuilder:
    def __init__(self, robot):
        self.robot = robot

    def build(
        self,
        output_path,
        grid,
        contact_epsilon=0.02,
        num_configurations=8000,
        max_boundary_pairs=80,
        seed=42,
        boundary_tolerance=1e-4,
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.perf_counter()

        q_pool = self.robot.sample_configurations(num_configurations, seed=seed)
        centers = np.asarray(grid["centers"], dtype=float)
        valid = np.asarray(grid["valid"], dtype=bool)
        cell_indices = np.asarray(grid["cell_indices"], dtype=np.int32)

        obstacle_valid = np.zeros(len(centers), dtype=bool)
        offsets = [0]
        all_boundary = []
        all_free = []
        all_collision = []
        miner = BoundaryMiner(tolerance=boundary_tolerance)

        for obstacle_id, center in enumerate(centers):
            if not valid[obstacle_id]:
                offsets.append(offsets[-1])
                continue
            labels = np.asarray([self.robot.link_point_distance(q, center) <= contact_epsilon for q in q_pool], dtype=np.uint8)
            if np.all(labels == 0) or np.all(labels == 1):
                offsets.append(offsets[-1])
                continue
            pairs = nearest_opposite_pairs(q_pool, labels, max_pairs=max_boundary_pairs)
            if not pairs:
                offsets.append(offsets[-1])
                continue
            boundary_q = []
            free_q = []
            collision_q = []
            collision_fn = lambda q_test, c=center: self.robot.link_point_distance(q_test, c) <= contact_epsilon
            for free_id, collision_id in pairs:
                q_free = q_pool[free_id]
                q_collision = q_pool[collision_id]
                boundary_q.append(miner.find_boundary(q_free, q_collision, collision_fn))
                free_q.append(q_free)
                collision_q.append(q_collision)
            obstacle_valid[obstacle_id] = True
            all_boundary.append(np.asarray(boundary_q, dtype=np.float32))
            all_free.append(np.asarray(free_q, dtype=np.float32))
            all_collision.append(np.asarray(collision_q, dtype=np.float32))
            offsets.append(offsets[-1] + len(boundary_q))

        boundary = stack_or_empty(all_boundary)
        source_free = stack_or_empty(all_free)
        source_collision = stack_or_empty(all_collision)
        offsets = np.asarray(offsets, dtype=np.int64)

        with h5py.File(output_path, "w") as h5:
            meta = h5.create_group("meta")
            meta.attrs["version"] = "0.1"
            meta.attrs["robot_name"] = "Planar3RRobot"
            meta.attrs["link_lengths"] = json.dumps(self.robot.link_lengths.tolist())
            meta.attrs["joint_limits"] = json.dumps(self.robot.joint_limits.tolist())
            meta.attrs["obstacle_radius"] = 0.0
            meta.attrs["contact_epsilon"] = float(contact_epsilon)
            meta.attrs["num_configurations"] = int(num_configurations)
            meta.attrs["max_boundary_pairs"] = int(max_boundary_pairs)
            meta.attrs["workspace_min"] = np.asarray(grid["workspace_min"], dtype=float)
            meta.attrs["workspace_max"] = np.asarray(grid["workspace_max"], dtype=float)
            meta.attrs["grid_shape"] = np.asarray(grid["grid_shape"], dtype=np.int32)
            meta.attrs["cell_size"] = np.asarray(grid["cell_size"], dtype=float)
            meta.attrs["created_at_unix"] = time.time()

            obstacles = h5.create_group("obstacles")
            obstacles.create_dataset("centers", data=centers.astype(np.float32))
            obstacles.create_dataset("cell_indices", data=cell_indices)
            obstacles.create_dataset("valid", data=obstacle_valid)
            obstacles.create_dataset("boundary_offsets", data=offsets)

            samples = h5.create_group("samples")
            samples.create_dataset("boundary_q", data=boundary, compression="gzip")
            samples.create_dataset("source_free_q", data=source_free, compression="gzip")
            samples.create_dataset("source_collision_q", data=source_collision, compression="gzip")

        elapsed = time.perf_counter() - start_time
        return {
            "output_path": output_path,
            "num_obstacles": len(centers),
            "num_valid_obstacles": int(np.sum(obstacle_valid)),
            "num_boundary_samples": len(boundary),
            "elapsed_seconds": elapsed,
        }


def stack_or_empty(chunks):
    if not chunks:
        return np.empty((0, 3), dtype=np.float32)
    return np.vstack(chunks).astype(np.float32)


class ObstacleManifoldDatabase:
    def __init__(self, path, h5):
        self.path = Path(path)
        self.h5 = h5
        self.workspace_min = self.h5["meta"].attrs["workspace_min"].astype(float)
        self.workspace_max = self.h5["meta"].attrs["workspace_max"].astype(float)
        self.grid_shape = tuple(int(v) for v in self.h5["meta"].attrs["grid_shape"])
        self.cell_size = self.h5["meta"].attrs["cell_size"].astype(float)
        self.obstacle_radius = float(self.h5["meta"].attrs["obstacle_radius"])
        self.contact_epsilon = float(self.h5["meta"].attrs.get("contact_epsilon", self.obstacle_radius))

    @classmethod
    def open(cls, path):
        return cls(path, h5py.File(path, "r"))

    def close(self):
        self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def query_nearest_obstacle(self, center):
        center = np.asarray(center, dtype=float)
        valid = self.h5["obstacles/valid"][:].astype(bool)
        if not np.any(valid):
            return None
        centers = self.h5["obstacles/centers"][:]
        valid_ids = np.flatnonzero(valid)
        distances = np.linalg.norm(centers[valid_ids] - center[None, :], axis=1)
        local = int(np.argmin(distances))
        return int(valid_ids[local]), centers[valid_ids[local]], float(distances[local])

    def load_manifold(self, obstacle_id):
        obstacle_id = int(obstacle_id)
        offsets = self.h5["obstacles/boundary_offsets"]
        start = int(offsets[obstacle_id])
        end = int(offsets[obstacle_id + 1])
        return ObstacleManifold(
            obstacle_id=obstacle_id,
            center=self.h5["obstacles/centers"][obstacle_id],
            radius=self.obstacle_radius,
            boundary_q=self.h5["samples/boundary_q"][start:end],
            source_free_q=self.h5["samples/source_free_q"][start:end],
            source_collision_q=self.h5["samples/source_collision_q"][start:end],
        )
