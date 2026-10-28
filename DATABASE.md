# Null-space manifold database

Build a planar 3R null-space manifold database:

```bash
python scripts/build_database.py --output data/nullspace_database.h5 --grid-x 20 --grid-y 20 --max-samples 220 --beta 0.20
```

Append missing targets to an existing database:

```bash
python scripts/build_database.py --output data/nullspace_database.h5 --grid-x 40 --grid-y 40 --max-samples 220 --beta 0.20 --append
```

Inspect the database and query nearby target manifolds:

```bash
python scripts/inspect_database.py --database data/nullspace_database.h5 --target 1.3 0.7 --k 3 --radius-cells 2
```

Interactively visualize the nearest source target selected by the workspace hash:

```bash
python scripts/view_database_query.py --database data/nullspace_database.h5 --target 1.3 0.7 --radius-cells 2
```

Drag the red query marker in the left panel. The nearest database source target is highlighted in
the workspace, and the right panel updates to show its null-space manifold.
Use `-k` to draw additional nearest manifolds for comparison:

```bash
python scripts/view_database_query.py --database data/nullspace_database.h5 --target 1.3 0.7 -k 3
```

Interactively visualize interpolation (default: TPS followed by projection):

```bash
python scripts/view_database_interpolation.py --database data/nullspace_database.h5 --target 1.3 0.7 --anchors 16 --show-gt
```

The right panel overlays the source manifold, projected anchors, estimated manifold, and
optionally direct self-motion samples for the query target.
The interpolation still uses only the nearest source manifold; `-k` only draws additional
neighboring manifolds for visual comparison.

Interpolation methods:

```bash
python scripts/view_database_interpolation.py --database data/nullspace_database.h5 --method rbf --show-gt
python scripts/view_database_interpolation.py --database data/nullspace_database.h5 --method rbf-project --show-gt
python scripts/view_database_interpolation.py --database data/nullspace_database.h5 --method tps --show-gt
python scripts/view_database_interpolation.py --database data/nullspace_database.h5 --method tps-project --show-gt
```

`*-project` first interpolates, then projects every estimated sample back to the query target
constraint with Gauss-Newton. This is usually the most accurate mode for task error.

The HDF5 file stores:

- `/meta`: robot, workspace, grid, and generator metadata
- `/targets`: target positions, cell indices, validity flags, and manifold offsets
- `/samples`: all joint-space samples in one contiguous table
- `/workspace_hash`: cell keys plus offset tables for target IDs and sample IDs

The main Python API is:

```python
from database import NullspaceDatabase

with NullspaceDatabase.open("data/nullspace_database.h5") as db:
    nearest = db.query_nearest_target([1.3, 0.7], search_radius_cells=2)
    if nearest is not None:
        manifold = db.load_full_manifold(nearest.target_id)
```

For quick smoke tests, use a smaller grid:

```bash
python scripts/build_database.py --output outputs/test_nullspace_database.h5 --grid-x 5 --grid-y 5 --max-samples 30
```

Queries only search the requested neighboring hash cells; they do not guarantee a global
nearest target. Append skips existing positions (including invalid targets) and rewrites
the file in place. Back up valuable databases before appending.

The displayed task error is `norm(FK(q) - target)`, not distance to direct GT samples.
Estimated timing excludes database lookup and plotting; `*-project` includes projection
of all output samples. Direct GT is a numerical trace, not an exhaustive ground truth.
