import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obstacle_database.obstacle_manifold_database import (
    ObstacleManifoldDatabaseBuilder,
    load_nullspace_grid,
    make_grid_from_args,
)
from robot.planar3r import Planar3RRobot


def main():
    parser = argparse.ArgumentParser(description="Build obstacle collision-boundary manifolds on the null-space grid.")
    parser.add_argument("--output", default="data/obstacle_manifold_database.h5")
    parser.add_argument("--nullspace-database", default="data/nullspace_database.h5")
    parser.add_argument("--grid-x", type=int, default=20)
    parser.add_argument("--grid-y", type=int, default=20)
    parser.add_argument("--num-configurations", type=int, default=8000)
    parser.add_argument("--max-boundary-pairs", type=int, default=80)
    parser.add_argument("--contact-epsilon", type=float, default=0.02)
    parser.add_argument("--obstacle-radius", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-nullspace-grid", action="store_true", help="use --grid-x/--grid-y instead of nullspace DB grid")
    args = parser.parse_args()

    if not args.no_nullspace_grid:
        if not Path(args.nullspace_database).is_file():
            parser.error("nullspace database not found; build it first or use --no-nullspace-grid")
        grid = load_nullspace_grid(args.nullspace_database)
        grid_source = args.nullspace_database
    else:
        grid = make_grid_from_args(grid_shape=(args.grid_x, args.grid_y))
        grid_source = f"{args.grid_x}x{args.grid_y} generated grid"

    robot = Planar3RRobot()
    summary = ObstacleManifoldDatabaseBuilder(robot).build(
        output_path=args.output,
        grid=grid,
        contact_epsilon=args.contact_epsilon if args.obstacle_radius is None else args.obstacle_radius,
        num_configurations=args.num_configurations,
        max_boundary_pairs=args.max_boundary_pairs,
        seed=args.seed,
    )
    print(f"database: {summary['output_path'].resolve()}")
    print(f"grid source: {grid_source}")
    print(f"obstacles: {summary['num_valid_obstacles']}/{summary['num_obstacles']} valid")
    print(f"boundary samples: {summary['num_boundary_samples']}")
    print(f"elapsed: {summary['elapsed_seconds']:.2f} s")


if __name__ == "__main__":
    main()
