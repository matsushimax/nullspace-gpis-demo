import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obstacle_database.spatial_hash import WorkspaceHash
from robot.planar3r import Planar3RRobot


def attr_json(attrs, key):
    value = attrs[key]
    return json.loads(value) if isinstance(value, str) else value


def main():
    parser = argparse.ArgumentParser(description="Validate and summarize a Planar 3R collision database.")
    parser.add_argument("--database", default="outputs/planar3r_collision_database.h5")
    parser.add_argument("--sample-checks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    with h5py.File(args.database, "r") as h5:
        meta = h5["metadata"].attrs
        q = h5["configuration_pool/q"][:]
        labels = h5["configuration_pool/self_collision_label"][:]
        centers = h5["configuration_pool/circle_centers"]
        radii = h5["configuration_pool/circle_radii"][:]

        robot = Planar3RRobot(
            link_lengths=attr_json(meta, "link_lengths"),
            joint_limits=attr_json(meta, "joint_limits"),
            link_radius=float(meta["link_radius"]),
            circle_radius=float(meta["circle_radius"]),
            circle_spacing=float(meta["circle_spacing"]),
        )
        workspace_hash = WorkspaceHash.from_arrays(
            attr_json(meta, "workspace_min"),
            attr_json(meta, "workspace_max"),
            float(meta["workspace_cell_size"]),
            h5["workspace_hash/cell_keys"][:],
            h5["workspace_hash/offsets"][:],
            h5["workspace_hash/configuration_ids"][:],
        )

        rng = np.random.default_rng(args.seed)
        ids = rng.choice(len(q), size=min(args.sample_checks, len(q)), replace=False)
        label_mismatch = 0
        hash_miss = 0
        start = time.perf_counter()
        for config_id in ids:
            expected_label = robot.self_collision(q[config_id])
            if int(expected_label) != int(labels[config_id]):
                label_mismatch += 1
            for center, radius in zip(centers[config_id], radii):
                candidates = workspace_hash.query_circle(center, radius)
                if int(config_id) not in set(candidates.tolist()):
                    hash_miss += 1
                    break
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        print(f"database: {Path(args.database).resolve()}")
        print(f"configurations: {len(q)}")
        print(f"self collision labels: {int(np.sum(labels))} collision / {int(len(labels) - np.sum(labels))} free")
        print(f"circles per config: {centers.shape[1]}")
        print(f"hash cells: {h5['workspace_hash/cell_keys'].shape[0]}")
        print(f"hash links: {h5['workspace_hash/configuration_ids'].shape[0]}")
        print(f"self dataset rows: {h5['self_collision_dataset/q'].shape[0] if 'q' in h5['self_collision_dataset'] else 0}")
        print(
            "external dataset rows: "
            f"{h5['external_collision_dataset/q'].shape[0] if 'q' in h5['external_collision_dataset'] else 0}"
        )
        print(f"sample checks: {len(ids)}")
        print(f"label mismatches: {label_mismatch}")
        print(f"hash misses: {hash_miss}")
        print(f"validation time: {elapsed_ms:.1f} ms")


if __name__ == "__main__":
    main()
