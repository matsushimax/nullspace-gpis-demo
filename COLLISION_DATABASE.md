# Planar 3R collision database

There are two distinct formats: the capsule/circle collision dataset and the point-obstacle
epsilon-boundary manifold database. Neither is a collision-free planning guarantee.

## Capsule/Circle Collision Dataset

Generate a Planar 3R collision dataset database:

```bash
python scripts/generate_collision_database.py --output outputs/planar3r_collision_database.h5 --num-configurations 10000
```

Larger pool example:

```bash
python scripts/generate_collision_database.py --output outputs/planar3r_collision_database.h5 --num-configurations 100000 --self-boundary-pairs 5000 --external-queries 10000 --external-dataset-size 10000
```

Validate the generated database:

```bash
python scripts/validate_collision_database.py --database outputs/planar3r_collision_database.h5
```

## Point-Obstacle Boundary Database

Build point-obstacle epsilon-boundary samples on the same workspace grid as the null-space manifold
database:

```bash
python scripts/build_obstacle_manifold_database.py --nullspace-database data/nullspace_database.h5 --output data/obstacle_manifold_database.h5 --contact-epsilon 0.02
```

If a null-space database is not available, generate a standalone grid:

```bash
python scripts/build_obstacle_manifold_database.py --no-nullspace-grid --grid-x 20 --grid-y 20 --output data/obstacle_manifold_database.h5 --contact-epsilon 0.02
```

Visualize obstacle manifolds:

```bash
python scripts/view_obstacle_manifold_database.py --database data/obstacle_manifold_database.h5 --center 0.8 0.4
```

The left panel selects a point obstacle in workspace. The right panel shows configurations near
`dist(point, link centerline) = contact_epsilon`, with nearby free/contact-side samples.
This is not exact zero-distance contact. Random sampling and bisection do not exhaustively
cover the boundary. Use a positive epsilon; setting it to zero normally yields no contact-side
samples. Rebuild older radius-based obstacle databases before using the point-obstacle viewers.
Several boundary samples are picked and drawn as arm postures in the left panel; their colors match
the highlighted boundary points in the right panel.

```bash
python scripts/view_obstacle_manifold_database.py --database data/obstacle_manifold_database.h5 --center 0.8 0.4 --arm-samples 8
```

Visualize null-space and obstacle manifolds together:

```bash
python scripts/view_combined_manifolds.py --nullspace-database data/nullspace_database.h5 --obstacle-database data/obstacle_manifold_database.h5 --show-gt
```

Controls:

- Left click or drag in the workspace panel to move the end-effector target.
- Right click or drag in the workspace panel to move the obstacle center.

The right panel overlays the estimated null-space manifold and the obstacle collision-boundary
manifold in the same joint-space coordinates.

## File Formats

The capsule/circle collision dataset HDF5 file contains:

- `/configuration_pool`: sampled `q`, self-collision labels, circle centers, circle radii
- `/workspace_hash`: CSR-style workspace cell to configuration-id postings
- `/self_collision_dataset`: free/collision samples, boundary configurations, signed distances, gradients
- `/external_collision_dataset`: obstacle-conditioned collision samples and boundary data
- `/metadata`: robot and generation parameters

The obstacle manifold database contains:

- `/obstacles`: obstacle centers, validity flags, and boundary sample offsets
- `/samples/boundary_q`: configuration-space samples near point-link contact
- `/samples/source_free_q`: samples with point-link distance greater than `contact_epsilon`
- `/samples/source_collision_q`: samples with point-link distance less than or equal to `contact_epsilon`

Capsule/circle collision dataset defaults (radii do not apply to the point-obstacle DB):

- joint limits: `[-pi, pi]` for all three joints
- link lengths: `(1.0, 1.0, 0.7)`
- link radius: `0.06`
- circle radius: `0.06`
- circle spacing: `0.08`
- workspace hash cell size: `0.05`

For fast smoke tests:

```bash
python scripts/generate_collision_database.py --output outputs/test_planar3r_collision_database.h5 --num-configurations 1000 --self-boundary-pairs 100 --external-queries 100 --external-dataset-size 100 --cell-size 0.08
python scripts/validate_collision_database.py --database outputs/test_planar3r_collision_database.h5 --sample-checks 100
```
