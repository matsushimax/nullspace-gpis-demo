import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from planar_arm import PlanarArm3DOF
from sampler import angular_distance, jacobian_guided_sampling, nullspace_basis


DATABASE_VERSION = "0.1"


@dataclass(slots=True)
class ManifoldSampleSet:
    target: np.ndarray
    q: np.ndarray
    tangents: np.ndarray
    residual_norms: np.ndarray
    arc_length: np.ndarray
    target_id: int | None = None


@dataclass(slots=True)
class DatabaseBuildConfig:
    grid_shape: tuple[int, int] = (20, 20)
    workspace_min: tuple[float, float] | None = None
    workspace_max: tuple[float, float] | None = None
    beta: float = 0.20
    max_samples: int = 220
    seed: int = 42


@dataclass(slots=True)
class BuildSummary:
    output_path: Path
    num_targets: int
    num_valid_targets: int
    total_samples: int
    elapsed_seconds: float
    num_added_targets: int = 0
    num_skipped_existing_targets: int = 0


@dataclass(slots=True)
class DatabaseArrays:
    targets: np.ndarray
    cell_indices: np.ndarray
    valid: np.ndarray
    manifold_offsets: np.ndarray
    q: np.ndarray
    target_id: np.ndarray
    local_index: np.ndarray
    residual_norm: np.ndarray
    tangent: np.ndarray
    arc_length: np.ndarray


@dataclass(slots=True)
class TargetQueryResult:
    target_id: int
    target: np.ndarray
    distance: float
    cell_index: np.ndarray
    num_samples: int


def default_workspace_bounds(arm):
    reach = float(np.sum(arm.link_lengths))
    return np.array([-reach, -reach], dtype=float), np.array([reach, reach], dtype=float)


def compute_cell_indices(positions, workspace_min, cell_size, grid_shape):
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    workspace_min = np.asarray(workspace_min, dtype=float)
    cell_size = np.asarray(cell_size, dtype=float)
    indices = np.floor((positions - workspace_min) / cell_size).astype(np.int32)
    grid_shape = np.asarray(grid_shape, dtype=np.int32)
    return np.clip(indices, 0, grid_shape - 1)


def flatten_cell_index(cell_index, grid_shape):
    cell_index = np.asarray(cell_index, dtype=np.int64)
    return int(cell_index[0] * grid_shape[1] + cell_index[1])


def unflatten_cell_index(flat_index, grid_shape):
    return np.array([flat_index // grid_shape[1], flat_index % grid_shape[1]], dtype=np.int32)


def generate_target_grid(workspace_min, workspace_max, grid_shape):
    workspace_min = np.asarray(workspace_min, dtype=float)
    workspace_max = np.asarray(workspace_max, dtype=float)
    xs = np.linspace(workspace_min[0], workspace_max[0], grid_shape[0])
    ys = np.linspace(workspace_min[1], workspace_max[1], grid_shape[1])
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()])


def compute_tangents(arm, q_samples):
    tangents = []
    prev_v = None
    for q in q_samples:
        basis = nullspace_basis(arm.jacobian(q))
        if basis.shape[1] == 0:
            v = np.zeros(arm.link_lengths.shape[0])
        else:
            v = basis[:, 0]
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            if prev_v is not None and np.dot(v, prev_v) < 0:
                v = -v
            prev_v = v
        tangents.append(v)
    return np.asarray(tangents, dtype=float)


def compute_arc_length(q_samples):
    arc = np.zeros(len(q_samples), dtype=float)
    for i in range(1, len(q_samples)):
        arc[i] = arc[i - 1] + angular_distance(q_samples[i], q_samples[i - 1])
    return arc


def sample_target_manifold(arm, target, target_id, beta, max_samples, seed):
    q = jacobian_guided_sampling(
        arm,
        target,
        beta=beta,
        max_samples=max_samples,
        seed=seed + target_id,
    )
    residuals = np.linalg.norm(np.array([arm.forward_kinematics(qi) - target for qi in q]), axis=1)
    return ManifoldSampleSet(
        target=np.asarray(target, dtype=float),
        q=q,
        tangents=compute_tangents(arm, q),
        residual_norms=residuals,
        arc_length=compute_arc_length(q),
        target_id=target_id,
    )


class NullspaceDatabaseBuilder:
    def __init__(self, arm):
        self.arm = arm

    def build(self, output_path, config=None, append=False):
        config = config or DatabaseBuildConfig()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.perf_counter()

        workspace_min, workspace_max = self._workspace_bounds(config)
        target_grid_shape = tuple(config.grid_shape)
        hash_grid_shape = target_grid_shape
        cell_size = (workspace_max - workspace_min) / np.asarray(hash_grid_shape, dtype=float)
        existing = None
        skipped_existing = 0
        target_id_offset = 0
        existing_positions = None

        if append and output_path.exists():
            existing, existing_meta = read_database_arrays(output_path)
            workspace_min = existing_meta["workspace_min"]
            workspace_max = existing_meta["workspace_max"]
            hash_grid_shape = existing_meta["grid_shape"]
            cell_size = existing_meta["cell_size"]
            target_id_offset = len(existing.targets)
            existing_positions = existing.targets

        targets = generate_target_grid(workspace_min, workspace_max, target_grid_shape)
        cell_indices = compute_cell_indices(targets, workspace_min, cell_size, hash_grid_shape)

        valid = np.zeros(len(targets), dtype=bool)
        manifold_offsets = [0]
        all_q = []
        all_target_id = []
        all_local_index = []
        all_residuals = []
        all_tangents = []
        all_arc_length = []

        new_target_positions = []
        new_cell_indices = []
        new_valid = []

        for local_target_id, target in enumerate(targets):
            if existing_positions is not None and target_exists(existing_positions, target):
                skipped_existing += 1
                continue

            target_id = target_id_offset + len(new_target_positions)
            new_target_positions.append(target)
            new_cell_indices.append(cell_indices[local_target_id])
            if not self._is_reachable(target):
                new_valid.append(False)
                manifold_offsets.append(manifold_offsets[-1])
                continue
            try:
                manifold = sample_target_manifold(
                    self.arm,
                    target,
                    target_id=target_id,
                    beta=config.beta,
                    max_samples=config.max_samples,
                    seed=config.seed,
                )
            except Exception as exc:
                print(f"skip target_id={target_id} target={target}: {exc}")
                new_valid.append(False)
                manifold_offsets.append(manifold_offsets[-1])
                continue

            new_valid.append(True)
            count = len(manifold.q)
            all_q.append(manifold.q)
            all_target_id.append(np.full(count, target_id, dtype=np.int32))
            all_local_index.append(np.arange(count, dtype=np.int32))
            all_residuals.append(manifold.residual_norms)
            all_tangents.append(manifold.tangents)
            all_arc_length.append(manifold.arc_length)
            manifold_offsets.append(manifold_offsets[-1] + count)

        targets = np.asarray(new_target_positions, dtype=float).reshape((-1, 2))
        cell_indices = np.asarray(new_cell_indices, dtype=np.int32).reshape((-1, 2))
        valid = np.asarray(new_valid, dtype=bool)
        generated = DatabaseArrays(
            targets=targets,
            cell_indices=cell_indices,
            valid=valid,
            manifold_offsets=np.asarray(manifold_offsets, dtype=np.int64),
            q=self._stack_or_empty(all_q, (0, 3), dtype=float),
            target_id=self._concat_or_empty(all_target_id, dtype=np.int32),
            local_index=self._concat_or_empty(all_local_index, dtype=np.int32),
            residual_norm=self._concat_or_empty(all_residuals, dtype=float),
            tangent=self._stack_or_empty(all_tangents, (0, 3), dtype=float),
            arc_length=self._concat_or_empty(all_arc_length, dtype=float),
        )
        arrays = merge_database_arrays(existing, generated) if existing is not None else generated

        hash_data = build_workspace_hash(
            valid=arrays.valid,
            cell_indices=arrays.cell_indices,
            manifold_offsets=arrays.manifold_offsets,
            grid_shape=hash_grid_shape,
        )

        with h5py.File(output_path, "w") as h5:
            meta = h5.create_group("meta")
            meta.attrs["version"] = DATABASE_VERSION
            meta.attrs["robot_name"] = "PlanarArm3DOF"
            meta.attrs["dof"] = 3
            meta.attrs["task_dim"] = 2
            meta.attrs["link_lengths"] = self.arm.link_lengths
            meta.attrs["workspace_min"] = workspace_min
            meta.attrs["workspace_max"] = workspace_max
            meta.attrs["grid_shape"] = np.asarray(hash_grid_shape, dtype=np.int32)
            meta.attrs["cell_size"] = cell_size
            meta.attrs["generator_config_json"] = json.dumps(config_to_dict(config))
            meta.attrs["created_at_unix"] = time.time()

            targets_group = h5.create_group("targets")
            targets_group.create_dataset("positions", data=arrays.targets)
            targets_group.create_dataset("cell_indices", data=arrays.cell_indices)
            targets_group.create_dataset("valid", data=arrays.valid)
            targets_group.create_dataset("manifold_offsets", data=arrays.manifold_offsets)

            samples_group = h5.create_group("samples")
            samples_group.create_dataset("q", data=arrays.q)
            samples_group.create_dataset("target_id", data=arrays.target_id)
            samples_group.create_dataset("local_index", data=arrays.local_index)
            samples_group.create_dataset("residual_norm", data=arrays.residual_norm)
            samples_group.create_dataset("tangent", data=arrays.tangent)
            samples_group.create_dataset("arc_length", data=arrays.arc_length)

            hash_group = h5.create_group("workspace_hash")
            for key, value in hash_data.items():
                hash_group.create_dataset(key, data=value)

        elapsed = time.perf_counter() - start_time
        return BuildSummary(
            output_path=output_path,
            num_targets=len(arrays.targets),
            num_valid_targets=int(np.sum(arrays.valid)),
            total_samples=len(arrays.q),
            elapsed_seconds=elapsed,
            num_added_targets=len(generated.targets),
            num_skipped_existing_targets=skipped_existing,
        )

    def _workspace_bounds(self, config):
        if config.workspace_min is None or config.workspace_max is None:
            return default_workspace_bounds(self.arm)
        return np.asarray(config.workspace_min, dtype=float), np.asarray(config.workspace_max, dtype=float)

    def _is_reachable(self, target):
        return np.linalg.norm(target) <= float(np.sum(self.arm.link_lengths)) + 1e-9

    @staticmethod
    def _stack_or_empty(values, shape, dtype):
        return np.vstack(values).astype(dtype) if values else np.empty(shape, dtype=dtype)

    @staticmethod
    def _concat_or_empty(values, dtype):
        return np.concatenate(values).astype(dtype) if values else np.empty((0,), dtype=dtype)


def build_workspace_hash(valid, cell_indices, manifold_offsets, grid_shape):
    num_cells = int(grid_shape[0] * grid_shape[1])
    target_bins = [[] for _ in range(num_cells)]
    sample_bins = [[] for _ in range(num_cells)]

    for target_id, ok in enumerate(valid):
        if not ok:
            continue
        flat_cell = flatten_cell_index(cell_indices[target_id], grid_shape)
        target_bins[flat_cell].append(target_id)
        start = int(manifold_offsets[target_id])
        end = int(manifold_offsets[target_id + 1])
        sample_bins[flat_cell].extend(range(start, end))

    cell_keys = np.asarray([unflatten_cell_index(i, grid_shape) for i in range(num_cells)], dtype=np.int32)
    target_id_offsets, target_ids = bins_to_offsets(target_bins, dtype=np.int32)
    sample_id_offsets, sample_ids = bins_to_offsets(sample_bins, dtype=np.int64)
    return {
        "cell_keys": cell_keys,
        "id_offsets": sample_id_offsets,
        "sample_ids": sample_ids,
        "target_id_offsets": target_id_offsets,
        "target_ids": target_ids,
        "sample_id_offsets": sample_id_offsets,
    }


def bins_to_offsets(bins, dtype):
    offsets = [0]
    flat = []
    for values in bins:
        flat.extend(values)
        offsets.append(len(flat))
    return np.asarray(offsets, dtype=np.int64), np.asarray(flat, dtype=dtype)


def config_to_dict(config):
    return {
        "grid_shape": list(config.grid_shape),
        "workspace_min": None if config.workspace_min is None else list(config.workspace_min),
        "workspace_max": None if config.workspace_max is None else list(config.workspace_max),
        "beta": config.beta,
        "max_samples": config.max_samples,
        "seed": config.seed,
    }


def target_exists(existing_positions, target, atol=1e-10):
    if len(existing_positions) == 0:
        return False
    return bool(np.any(np.all(np.isclose(existing_positions, target[None, :], atol=atol, rtol=0.0), axis=1)))


def read_database_arrays(path):
    with h5py.File(path, "r") as h5:
        arrays = DatabaseArrays(
            targets=h5["targets/positions"][:],
            cell_indices=h5["targets/cell_indices"][:],
            valid=h5["targets/valid"][:],
            manifold_offsets=h5["targets/manifold_offsets"][:],
            q=h5["samples/q"][:],
            target_id=h5["samples/target_id"][:],
            local_index=h5["samples/local_index"][:],
            residual_norm=h5["samples/residual_norm"][:],
            tangent=h5["samples/tangent"][:],
            arc_length=h5["samples/arc_length"][:],
        )
        meta = {
            "workspace_min": h5["meta"].attrs["workspace_min"].astype(float),
            "workspace_max": h5["meta"].attrs["workspace_max"].astype(float),
            "grid_shape": tuple(int(v) for v in h5["meta"].attrs["grid_shape"]),
            "cell_size": h5["meta"].attrs["cell_size"].astype(float),
        }
    return arrays, meta


def merge_database_arrays(existing, new):
    if existing is None:
        return new
    sample_offset = len(existing.q)
    merged_offsets = np.concatenate(
        [existing.manifold_offsets, existing.manifold_offsets[-1] + new.manifold_offsets[1:]]
    ).astype(np.int64)
    return DatabaseArrays(
        targets=np.vstack([existing.targets, new.targets]) if len(new.targets) else existing.targets,
        cell_indices=np.vstack([existing.cell_indices, new.cell_indices]) if len(new.cell_indices) else existing.cell_indices,
        valid=np.concatenate([existing.valid, new.valid]) if len(new.valid) else existing.valid,
        manifold_offsets=merged_offsets,
        q=np.vstack([existing.q, new.q]) if len(new.q) else existing.q,
        target_id=np.concatenate([existing.target_id, new.target_id]) if len(new.target_id) else existing.target_id,
        local_index=np.concatenate([existing.local_index, new.local_index]) if len(new.local_index) else existing.local_index,
        residual_norm=np.concatenate([existing.residual_norm, new.residual_norm])
        if len(new.residual_norm)
        else existing.residual_norm,
        tangent=np.vstack([existing.tangent, new.tangent]) if len(new.tangent) else existing.tangent,
        arc_length=np.concatenate([existing.arc_length, new.arc_length]) if len(new.arc_length) else existing.arc_length,
    )


class NullspaceDatabase:
    def __init__(self, path, h5):
        self.path = Path(path)
        self.h5 = h5
        self.workspace_min = self.h5["meta"].attrs["workspace_min"].astype(float)
        self.workspace_max = self.h5["meta"].attrs["workspace_max"].astype(float)
        self.grid_shape = tuple(int(v) for v in self.h5["meta"].attrs["grid_shape"])
        self.cell_size = self.h5["meta"].attrs["cell_size"].astype(float)

    @classmethod
    def open(cls, path):
        return cls(path, h5py.File(path, "r"))

    def close(self):
        self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def target_count(self):
        return int(self.h5["targets/positions"].shape[0])

    def valid_target_count(self):
        return int(np.sum(self.h5["targets/valid"][:]))

    def sample_count(self):
        return int(self.h5["samples/q"].shape[0])

    def query_cell(self, target):
        target = np.asarray(target, dtype=float)
        if np.any(target < self.workspace_min) or np.any(target > self.workspace_max):
            return np.empty((0,), dtype=np.int32)
        cell = compute_cell_indices(target, self.workspace_min, self.cell_size, self.grid_shape)[0]
        flat = flatten_cell_index(cell, self.grid_shape)
        offsets = self.h5["workspace_hash/target_id_offsets"]
        ids = self.h5["workspace_hash/target_ids"]
        return ids[int(offsets[flat]):int(offsets[flat + 1])]

    def query_nearest_target(self, target, search_radius_cells=1):
        results = self.query_k_nearest_targets(target, k=1, search_radius_cells=search_radius_cells)
        return results[0] if results else None

    def query_k_nearest_targets(self, target, k, search_radius_cells=2):
        target = np.asarray(target, dtype=float)
        center = compute_cell_indices(target, self.workspace_min, self.cell_size, self.grid_shape)[0]
        candidate_ids = []
        for dx in range(-search_radius_cells, search_radius_cells + 1):
            for dy in range(-search_radius_cells, search_radius_cells + 1):
                cell = center + np.array([dx, dy], dtype=np.int32)
                if np.any(cell < 0) or np.any(cell >= np.asarray(self.grid_shape)):
                    continue
                flat = flatten_cell_index(cell, self.grid_shape)
                offsets = self.h5["workspace_hash/target_id_offsets"]
                ids = self.h5["workspace_hash/target_ids"]
                candidate_ids.extend(ids[int(offsets[flat]):int(offsets[flat + 1])].tolist())

        if not candidate_ids:
            return []
        candidate_ids = np.asarray(sorted(set(candidate_ids)), dtype=np.int32)
        positions = self.h5["targets/positions"][candidate_ids]
        distances = np.linalg.norm(positions - target[None, :], axis=1)
        order = np.argsort(distances)[:k]
        offsets = self.h5["targets/manifold_offsets"]
        cell_indices = self.h5["targets/cell_indices"]
        return [
            TargetQueryResult(
                target_id=int(candidate_ids[i]),
                target=positions[i],
                distance=float(distances[i]),
                cell_index=cell_indices[int(candidate_ids[i])],
                num_samples=int(offsets[int(candidate_ids[i]) + 1] - offsets[int(candidate_ids[i])]),
            )
            for i in order
        ]

    def load_full_manifold(self, target_id):
        target_id = int(target_id)
        offsets = self.h5["targets/manifold_offsets"]
        start = int(offsets[target_id])
        end = int(offsets[target_id + 1])
        return ManifoldSampleSet(
            target=self.h5["targets/positions"][target_id],
            q=self.h5["samples/q"][start:end],
            tangents=self.h5["samples/tangent"][start:end],
            residual_norms=self.h5["samples/residual_norm"][start:end],
            arc_length=self.h5["samples/arc_length"][start:end],
            target_id=target_id,
        )
