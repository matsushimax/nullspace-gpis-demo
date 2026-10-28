"""Small temporary-database integration tests and headless viewer smoke tests."""

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from database import DatabaseBuildConfig, NullspaceDatabase, NullspaceDatabaseBuilder
from deformation import interpolate_manifold_tps
from obstacle_database.obstacle_manifold_database import (
    ObstacleManifoldDatabase,
    ObstacleManifoldDatabaseBuilder,
    load_nullspace_grid,
)
from planar_arm import PlanarArm3DOF
from robot.planar3r import Planar3RRobot
from scripts.view_database_interpolation import DatabaseInterpolationViewer
from scripts.view_database_query import DatabaseQueryViewer


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "nullspace.h5"
        self.arm = PlanarArm3DOF()
        self.config = DatabaseBuildConfig(
            grid_shape=(2, 2), workspace_min=(1.2, 0.6), workspace_max=(1.4, 0.8),
            max_samples=20, seed=42,
        )
        self.builder = NullspaceDatabaseBuilder(self.arm)
        self.builder.build(self.path, self.config)

    def test_roundtrip_and_tps_projection(self):
        with NullspaceDatabase.open(self.path) as db:
            nearest = db.query_nearest_target([1.21, 0.61], search_radius_cells=2)
            self.assertIsNotNone(nearest)
            np.testing.assert_allclose(nearest.target, [1.2, 0.6])
            manifold = db.load_full_manifold(nearest.target_id)
            self.assertGreater(len(manifold.q), 4)
            target = np.array([1.21, 0.61])
            result = interpolate_manifold_tps(self.arm, manifold, target, post_project=True)
            self.assertTrue(result.post_projection_converged.all())
            actual = np.linalg.norm([self.arm.forward_kinematics(q) - target for q in result.estimated_q], axis=1)
            np.testing.assert_allclose(result.task_residuals, actual)
            self.assertLess(actual.max(), 1e-8)

    def test_append_preserves_samples_and_skips_duplicates(self):
        with NullspaceDatabase.open(self.path) as db:
            old_q = db.h5["samples/q"][:]
        summary = self.builder.build(self.path, self.config, append=True)
        self.assertEqual(summary.num_added_targets, 0)
        self.assertEqual(summary.num_skipped_existing_targets, 4)
        self.config.grid_shape = (3, 3)
        summary = self.builder.build(self.path, self.config, append=True)
        self.assertEqual(summary.num_added_targets, 5)
        with NullspaceDatabase.open(self.path) as db:
            self.assertEqual(db.target_count(), 9)
            np.testing.assert_array_equal(db.h5["samples/q"][:len(old_q)], old_q)

    def test_obstacle_boundary_roundtrip(self):
        path = Path(self.directory.name) / "obstacles.h5"
        robot = Planar3RRobot()
        epsilon = 0.05
        summary = ObstacleManifoldDatabaseBuilder(robot).build(
            path, load_nullspace_grid(self.path), contact_epsilon=epsilon,
            num_configurations=300, max_boundary_pairs=5, seed=42,
        )
        self.assertGreater(summary["num_boundary_samples"], 0)
        with ObstacleManifoldDatabase.open(path) as db:
            nearest = db.query_nearest_obstacle([1.2, 0.6])
            self.assertIsNotNone(nearest)
            manifold = db.load_manifold(nearest[0])
            distances = [robot.link_point_distance(q, manifold.center) for q in manifold.boundary_q]
            np.testing.assert_allclose(distances, epsilon, atol=3e-4)
            self.assertTrue(all(robot.link_point_distance(q, manifold.center) > epsilon for q in manifold.source_free_q))
            self.assertTrue(all(robot.link_point_distance(q, manifold.center) <= epsilon for q in manifold.source_collision_q))

    def test_viewers_render_and_update(self):
        for viewer_type in (DatabaseQueryViewer, DatabaseInterpolationViewer):
            with self.subTest(viewer=viewer_type.__name__):
                viewer = viewer_type(self.path, initial_target=(1.21, 0.61), k=2)
                try:
                    viewer.update_query(1.3, 0.7, force=True)
                    viewer.fig.canvas.draw()
                    self.assertEqual(len(viewer.results), 2)
                    if isinstance(viewer, DatabaseInterpolationViewer):
                        self.assertIsNone(viewer.failure_reason)
                        self.assertIsNotNone(viewer.estimate_time_ms)
                finally:
                    viewer.on_close(None)
                    plt.close(viewer.fig)


if __name__ == "__main__":
    unittest.main()
