"""Bifocal-OR figure style: one palette and one look for Figs. 3-6, matching Fig. 2 (Master-Video).

Import it in a figure script:  from paper_style import PHASE_COLORS, apply, pill, ...
"""
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

# Phase palette of Fig. 2 (sampled from the paper PDF)
PHASES = ["Room_Prep", "Patient_Entry", "Sterile_Draping", "Microsurgery", "Emergence"]
PHASE_NAMES = ["Room Prep", "Patient Entry", "Sterile Draping", "Microsurgery", "Emergence"]
PHASE_COLORS = ["#2364AA", "#3DA5D9", "#73BFB8", "#FEC601", "#EA7317"]
DARK_BLUE, BLUE, TEAL, YELLOW, ORANGE = PHASE_COLORS

SPEECH = ORANGE            # speech segments (Fig. 2 speaker rows)
GT_GRAY = "#595959"        # manually recovered speech (Fig. 2 "GT (Dead Mic)")
INK = "#1F2A36"            # text, lanes, axes
MUTED = "#6B7680"          # secondary text
LANE = "#2B2B2B"           # thin lane lines as in Fig. 2
GRID = "#E3E7EB"

# Sequential map for matrices: white -> teal -> bright blue -> dark blue
SEQ = LinearSegmentedColormap.from_list("bifocal_seq", ["#FFFFFF", "#D9EFEC", TEAL, BLUE, DARK_BLUE, "#163F6C"])


def apply(size=7.5):
    """Fonts and line widths shared by all figures (Inter as in Fig. 2, else Arial/DejaVu)."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": size, "axes.labelsize": size, "xtick.labelsize": size - 0.5, "ytick.labelsize": size - 0.5,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })


def pill(ax, x, y, text, transform=None, size=7.0, fc="white", ec=INK, color=INK, ha="center"):
    """Rounded outlined label like "Phase" / "Speaker" in Fig. 2."""
    return ax.text(x, y, text, transform=transform or ax.transAxes, ha=ha, va="center", fontsize=size,
                   fontweight="bold", color=color,
                   bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.8", fc=fc, ec=ec, lw=0.9))


def bar(ax, t0, t1, y, color, lw=4.2, edge=None):
    """Rounded speech bar on a lane (round caps, thin darker outline), as in Fig. 2."""
    edge = edge or shade(color, 0.72)
    ax.plot([t0, t1], [y, y], color=edge, lw=lw + 1.0, solid_capstyle="round", zorder=3)
    ax.plot([t0, t1], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=4)


def shade(hex_color, f):
    """Darker (f < 1) or lighter (f > 1) version of a color."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    if f <= 1:
        r, g, b = (int(c * f) for c in (r, g, b))
    else:
        r, g, b = (int(c + (255 - c) * (f - 1)) for c in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"
