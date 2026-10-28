from collections import defaultdict

import numpy as np

from .workspace_grid import WorkspaceGrid


class WorkspaceHash:
    def __init__(self, grid):
        self.grid = grid
        self.cell_keys = np.empty((0, 2), dtype=np.int32)
        self.offsets = np.zeros(1, dtype=np.int64)
        self.configuration_ids = np.empty((0,), dtype=np.int32)
        self._cell_to_row = {}

    def build(self, circle_centers, circle_radii):
        postings = defaultdict(set)
        circle_radii = np.asarray(circle_radii, dtype=float)
        for config_id, centers in enumerate(circle_centers):
            occupied = set()
            for center, radius in zip(centers, circle_radii):
                for cell in self.grid.cells_overlapping_circle(center, radius):
                    occupied.add(tuple(int(v) for v in cell))
            for cell in occupied:
                postings[cell].add(config_id)

        ordered_cells = sorted(postings)
        offsets = [0]
        flat_ids = []
        for cell in ordered_cells:
            ids = sorted(postings[cell])
            flat_ids.extend(ids)
            offsets.append(len(flat_ids))

        self.cell_keys = np.asarray(ordered_cells, dtype=np.int32).reshape((-1, 2))
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.configuration_ids = np.asarray(flat_ids, dtype=np.int32)
        self._cell_to_row = {tuple(cell): i for i, cell in enumerate(ordered_cells)}
        return self

    def query_cell(self, cell):
        row = self._cell_to_row.get(tuple(int(v) for v in cell))
        if row is None:
            return np.empty((0,), dtype=np.int32)
        start = int(self.offsets[row])
        end = int(self.offsets[row + 1])
        return self.configuration_ids[start:end]

    def query_circle(self, center, radius):
        ids = []
        for cell in self.grid.cells_overlapping_circle(center, radius):
            ids.extend(self.query_cell(cell).tolist())
        if not ids:
            return np.empty((0,), dtype=np.int32)
        return np.asarray(sorted(set(ids)), dtype=np.int32)

    @classmethod
    def from_arrays(cls, workspace_min, workspace_max, cell_size, cell_keys, offsets, configuration_ids):
        h = cls(WorkspaceGrid(workspace_min, workspace_max, cell_size))
        h.cell_keys = np.asarray(cell_keys, dtype=np.int32)
        h.offsets = np.asarray(offsets, dtype=np.int64)
        h.configuration_ids = np.asarray(configuration_ids, dtype=np.int32)
        h._cell_to_row = {tuple(cell): i for i, cell in enumerate(h.cell_keys.tolist())}
        return h
