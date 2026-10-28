"""Shared visual conventions for the database viewers."""


def source_colors(count):
    """Match workspace targets and manifolds; repeat only after ten sources."""
    palette = (
        "tab:orange", "tab:blue", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
    )
    return [palette[index % len(palette)] for index in range(max(0, count))]
