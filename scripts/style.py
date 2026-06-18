"""Shared academic plotting style for SciSci figures."""

import matplotlib as mpl
import matplotlib.pyplot as plt

PAL = {
    "primary": "#4B8BBE",
    "secondary": "#A33475",
    "tertiary": "#7A7A7A",
    "accent": "#2D8659",
    "light": "#D4E6F1",
}


def apply_style():
    """Apply the academic figure style globally."""
    plt.style.use("default")
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


def save_fig(fig, name: str, out_dir: str):
    """Save a figure to out_dir and close it."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name), dpi=300)
    plt.close(fig)


def wrap_label(s: str, mx: int = 35) -> str:
    """Wrap long labels for axis ticks."""
    s = str(s).strip().title()
    if len(s) <= mx:
        return s
    words, lines, cur = s.split(), [], ""
    for w in words:
        c = f"{cur} {w}".strip() if cur else w
        if len(c) > mx and cur:
            lines.append(cur)
            cur = w
        else:
            cur = c
    if cur:
        lines.append(cur)
    return "\n".join(lines)
