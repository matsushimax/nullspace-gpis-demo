import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import NullspaceDatabase


def main():
    parser = argparse.ArgumentParser(description="Inspect and query a null-space manifold database.")
    parser.add_argument("--database", default="data/nullspace_database.h5", help="HDF5 database path")
    parser.add_argument("--target", nargs=2, type=float, metavar=("X", "Y"), help="target position to query")
    parser.add_argument("--radius-cells", type=int, default=2, help="neighbor cell search radius")
    parser.add_argument("--k", type=int, default=3, help="number of nearest targets to print")
    args = parser.parse_args()

    with NullspaceDatabase.open(args.database) as db:
        print(f"database: {Path(args.database).resolve()}")
        print(f"targets: {db.valid_target_count()}/{db.target_count()} valid")
        print(f"samples: {db.sample_count()}")
        print(f"workspace_min: {db.workspace_min}")
        print(f"workspace_max: {db.workspace_max}")
        print(f"grid_shape: {db.grid_shape}")
        print(f"cell_size: {db.cell_size}")

        if args.target is None:
            return

        print("")
        print(f"query target: {args.target}")
        results = db.query_k_nearest_targets(args.target, k=args.k, search_radius_cells=args.radius_cells)
        if not results:
            print("no nearby target found")
            return

        for result in results:
            print(
                "target_id={target_id} target=({x:.4f}, {y:.4f}) "
                "distance={distance:.4f} cell=({cx}, {cy}) samples={samples}".format(
                    target_id=result.target_id,
                    x=result.target[0],
                    y=result.target[1],
                    distance=result.distance,
                    cx=result.cell_index[0],
                    cy=result.cell_index[1],
                    samples=result.num_samples,
                )
            )

        manifold = db.load_full_manifold(results[0].target_id)
        print("")
        print(f"loaded nearest manifold target_id={manifold.target_id}")
        print(f"q shape: {manifold.q.shape}")
        print(f"max residual: {manifold.residual_norms.max():.3e}")
        print(f"arc length: {manifold.arc_length[-1]:.4f}")


if __name__ == "__main__":
    main()
