"""Numerical regression tests; no stored datasets or GUI required."""

import unittest

import numpy as np

from deformation import configuration_difference, post_project_samples
from planar_arm import PlanarArm3DOF
from robot.planar3r import (
    Planar3RRobot,
    closest_point_on_segment,
    point_segment_distance,
    segment_segment_distance,
)
from viewer_utils import source_colors


class GeometryTests(unittest.TestCase):
    def test_segment_clamping_and_degenerate_segment(self):
        for point, a, b, closest in (
            ([0.5, 1], [0, 0], [1, 0], [0.5, 0]),
            ([2, 1], [0, 0], [1, 0], [1, 0]),
            ([-1, 1], [0, 0], [1, 0], [0, 0]),
            ([1, 1], [0, 0], [0, 0], [0, 0]),
        ):
            with self.subTest(point=point, a=a, b=b):
                np.testing.assert_allclose(closest_point_on_segment(point, a, b), closest)
                self.assertAlmostEqual(
                    point_segment_distance(point, a, b),
                    np.linalg.norm(np.asarray(point) - closest),
                )

    def test_segment_intersection_and_separation(self):
        self.assertEqual(segment_segment_distance([0, 0], [1, 1], [0, 1], [1, 0]), 0)
        self.assertEqual(segment_segment_distance([0, 0], [2, 0], [1, 0], [3, 0]), 0)
        self.assertAlmostEqual(segment_segment_distance([0, 0], [1, 0], [0, 2], [1, 2]), 2)

    def test_robot_models_agree(self):
        arm, robot = PlanarArm3DOF(), Planar3RRobot()
        for q in robot.sample_configurations(10, seed=7):
            np.testing.assert_allclose(arm.joint_positions(q), robot.forward_kinematics(q))
            np.testing.assert_allclose(arm.forward_kinematics(q), robot.forward_kinematics(q)[-1])

    def test_closest_link_distance(self):
        robot = Planar3RRobot()
        closest, distance, link = robot.closest_link_point([0, 0, 0], [1.5, 0.2])
        np.testing.assert_allclose(closest, [1.5, 0])
        self.assertEqual(link, 1)
        self.assertAlmostEqual(distance, 0.2)
        self.assertAlmostEqual(robot.link_point_distance([0, 0, 0], [1.5, 0.2]), distance)


class NumericalTests(unittest.TestCase):
    def test_jacobian_matches_finite_differences(self):
        arm = PlanarArm3DOF()
        q = np.array([0.4, -0.8, 1.2])
        delta = np.eye(3) * 1e-6
        numerical = np.column_stack([
            (arm.forward_kinematics(q + d) - arm.forward_kinematics(q - d)) / 2e-6
            for d in delta
        ])
        np.testing.assert_allclose(arm.jacobian(q), numerical, atol=1e-8)

    def test_periodic_difference(self):
        np.testing.assert_allclose(configuration_difference([np.pi - 0.01], [-np.pi + 0.01]), [-0.02])

    def test_projection_residual_is_fk_error(self):
        arm = PlanarArm3DOF()
        q = np.array([0.4, -0.8, 1.2])
        target = arm.forward_kinematics(q)
        projected, converged, residuals = post_project_samples(arm, [q + 0.02], target)
        self.assertTrue(converged.all())
        actual = np.linalg.norm([arm.forward_kinematics(p) - target for p in projected], axis=1)
        np.testing.assert_allclose(residuals, actual, atol=1e-12)
        self.assertLess(actual.max(), 1e-8)

    def test_source_colors_are_distinct_and_repeatable(self):
        self.assertEqual(source_colors(0), [])
        self.assertEqual(source_colors(1), ["tab:orange"])
        self.assertEqual(len(set(source_colors(10))), 10)
        self.assertEqual(source_colors(11)[10], source_colors(1)[0])


if __name__ == "__main__":
    unittest.main()
