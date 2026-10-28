import numpy as np


class WorkspaceGrid:
    def __init__(self, workspace_min, workspace_max, cell_size):
        self.workspace_min = np.asarray(workspace_min, dtype=float)
        self.workspace_max = np.asarray(workspace_max, dtype=float)
        self.cell_size = float(cell_size)
        shape = np.ceil((self.workspace_max - self.workspace_min) / self.cell_size).astype(int)
        self.shape = tuple(np.maximum(shape, 1))

    def cell_index(self, point):
        point = np.asarray(point, dtype=float)
        idx = np.floor((point - self.workspace_min) / self.cell_size).astype(int)
        return np.clip(idx, 0, np.asarray(self.shape) - 1)

    def flatten(self, cell):
        cell = np.asarray(cell, dtype=int)
        return int(cell[0] * self.shape[1] + cell[1])

    def unflatten(self, flat):
        return np.array([flat // self.shape[1], flat % self.shape[1]], dtype=np.int32)

    def cell_bounds(self, cell):
        cell = np.asarray(cell, dtype=float)
        low = self.workspace_min + cell * self.cell_size
        high = low + self.cell_size
        return low, high

    def cells_overlapping_circle(self, center, radius):
        center = np.asarray(center, dtype=float)
        radius = float(radius)
        lo = self.cell_index(center - radius)
        hi = self.cell_index(center + radius)
        cells = []
        for ix in range(int(lo[0]), int(hi[0]) + 1):
            for iy in range(int(lo[1]), int(hi[1]) + 1):
                cell = np.array([ix, iy], dtype=np.int32)
                if circle_intersects_aabb(center, radius, *self.cell_bounds(cell)):
                    cells.append(cell)
        return cells


def circle_intersects_aabb(center, radius, aabb_min, aabb_max):
    closest = np.minimum(np.maximum(center, aabb_min), aabb_max)
    return bool(np.sum((closest - center) ** 2) <= radius * radius)
