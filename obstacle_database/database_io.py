import json
import time
from pathlib import Path

import h5py
import numpy as np


def write_collision_database(
    output_path,
    metadata,
    q,
    self_labels,
    circle_centers,
    circle_radii,
    workspace_hash,
    self_dataset=None,
    external_dataset=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        meta = h5.create_group("metadata")
        for key, value in metadata.items():
            if isinstance(value, (dict, list, tuple)):
                meta.attrs[key] = json.dumps(value)
            else:
                meta.attrs[key] = value
        meta.attrs["created_at_unix"] = time.time()

        pool = h5.create_group("configuration_pool")
        pool.create_dataset("q", data=np.asarray(q, dtype=np.float32), compression="gzip")
        pool.create_dataset("self_collision_label", data=np.asarray(self_labels, dtype=np.uint8), compression="gzip")
        pool.create_dataset("circle_centers", data=np.asarray(circle_centers, dtype=np.float32), compression="gzip")
        pool.create_dataset("circle_radii", data=np.asarray(circle_radii, dtype=np.float32))
        pool.create_dataset("valid", data=np.ones(len(q), dtype=np.uint8))

        wh = h5.create_group("workspace_hash")
        wh.create_dataset("cell_keys", data=workspace_hash.cell_keys)
        wh.create_dataset("offsets", data=workspace_hash.offsets)
        wh.create_dataset("configuration_ids", data=workspace_hash.configuration_ids)

        write_dataset_group(h5, "self_collision_dataset", self_dataset)
        write_dataset_group(h5, "external_collision_dataset", external_dataset)


def write_dataset_group(h5, name, data):
    group = h5.create_group(name)
    if not data:
        return
    for key, value in data.items():
        group.create_dataset(key, data=value, compression="gzip" if np.asarray(value).ndim > 0 else None)
