"""Ensure documented entry points can display help without a database."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CommandLineTests(unittest.TestCase):
    def test_missing_nullspace_database_does_not_generate_another_grid(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "obstacles.h5"
            result = subprocess.run(
                [sys.executable, str(root / "scripts/build_obstacle_manifold_database.py"),
                 "--nullspace-database", str(Path(directory) / "missing.h5"),
                 "--output", str(output)],
                cwd=root, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("--no-nullspace-grid", result.stderr)
            self.assertFalse(output.exists())

    def test_help(self):
        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(root / ".matplotlib"))
        scripts = [root / "interactive_demo.py", *sorted((root / "scripts").glob("*.py"))]
        for script in scripts:
            with self.subTest(script=script.name):
                result = subprocess.run(
                    [sys.executable, str(script), "--help"], cwd=root, env=environment,
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
