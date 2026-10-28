# Interactive self-motion demo

Run the interactive target-position demo with:

```bash
python interactive_demo.py
```

Controls:

- Drag the red target marker in the left panel to move the desired end-effector position.
- Use the `target x` and `target y` sliders for precise target placement.
- Press `Reset` to return to the original target `(1.3, 0.7)`.

The left panel redraws sampled arm configurations that satisfy the target end-effector
position. The right panel redraws the corresponding null-space manifold samples in
joint space, matching the static `outputs/manifold_samples_3d.png` view.

Useful options:

```bash
python interactive_demo.py --x 1.3 --y 0.7 --max-samples 180 --beta 0.20
```

If updates feel slow while dragging, reduce `--max-samples`.
