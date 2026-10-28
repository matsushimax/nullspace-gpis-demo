# Development

Use Python 3.10 or later in a virtual environment and install `requirements.txt`.
Run commands from the repository root.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests generate small databases in temporary directories and render with the Agg
backend. They do not require the datasets in `data/` or a desktop display.
For viewer changes, also check mouse interaction, labels, and legends in a GUI.

## Changes

- Keep numerical changes covered by focused regression tests.
- Record seeds, robot parameters, sample counts, and tolerances for experiments.
- Separate FK residuals from manifold coverage or joint-space distance metrics.
- Separate interpolation time from database lookup, offline generation, and rendering.
- Preserve existing HDF5 compatibility, or document a required rebuild.
- Do not commit generated databases, credentials, personal paths, or local agent files.
- Cite reference methods and retain notices when incorporating third-party code.

## Documentation Media

```bash
python scripts/render_readme_media.py
```

Review the generated files in `docs/images/` before committing. Timings shown in
the figures depend on the machine and are illustrative, not benchmark guarantees.
If a figure changes, update the corresponding example values in the README.

## Issue Reports

Include the command, Python and dependency versions, expected behavior, and the
error message. Use a small reproducible example rather than uploading a large
database. Remove private paths and identifying information from logs.
