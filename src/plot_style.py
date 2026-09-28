"""One shared look for every figure in the report."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Colour-blind checked categorical order: always assign in this order, never cycle.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MAIN_COLOR = SERIES_COLORS[0]
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

MODEL_COLORS = {
    "Naive Bayes": SERIES_COLORS[0],
    "Logistic Regression": SERIES_COLORS[1],
    "BiLSTM + attention": SERIES_COLORS[2],
    "TextCNN": SERIES_COLORS[3],
    "AfriBERTa": SERIES_COLORS[4],
}

BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT,
            "lines.linewidth": 2,
            "legend.frameon": False,
        }
    )


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


apply_style()
