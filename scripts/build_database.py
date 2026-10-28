import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import DatabaseBuildConfig, NullspaceDatabaseBuilder
from planar_arm import PlanarArm3DOF


def main():
    parser = argparse.ArgumentParser(description="Build a null-space manifold HDF5 database.")
    parser.add_argument("--output", default="data/nullspace_database.h5", help="output HDF5 path")
    parser.add_argument("--grid-x", type=int, default=20, help="number of workspace grid points in x")
    parser.add_argument("--grid-y", type=int, default=20, help="number of workspace grid points in y")
    parser.add_argument("--max-samples", type=int, default=220, help="maximum samples per target")
    parser.add_argument("--beta", type=float, default=0.20, help="null-space tracing step")
    parser.add_argument("--seed", type=int, default=42, help="base random seed")
    parser.add_argument("--workspace-min", nargs=2, type=float, default=None, metavar=("X", "Y"))
    parser.add_argument("--workspace-max", nargs=2, type=float, default=None, metavar=("X", "Y"))
    parser.add_argument("--append", action="store_true", help="append missing targets to an existing database")
    args = parser.parse_args()

    arm = PlanarArm3DOF(link_lengths=(1.0, 1.0, 0.7))
    config = DatabaseBuildConfig(
        grid_shape=(args.grid_x, args.grid_y),
        workspace_min=None if args.workspace_min is None else tuple(args.workspace_min),
        workspace_max=None if args.workspace_max is None else tuple(args.workspace_max),
        beta=args.beta,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    summary = NullspaceDatabaseBuilder(arm).build(args.output, config, append=args.append)
    print(f"database: {summary.output_path.resolve()}")
    print(f"targets: {summary.num_valid_targets}/{summary.num_targets} valid")
    print(f"samples: {summary.total_samples}")
    print(f"added targets: {summary.num_added_targets}")
    if args.append:
        print(f"skipped existing targets: {summary.num_skipped_existing_targets}")
    print(f"elapsed: {summary.elapsed_seconds:.2f} s")


if __name__ == "__main__":
    main()
