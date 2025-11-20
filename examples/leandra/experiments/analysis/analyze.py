"""
Analyze Computation Scaling Curves for MiniF2F Test
"""

# pyright: ignore

from pathlib import Path

import matplotlib
import matplotlib as mpl

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter

# Toggle display of curves, legend, and axis labels
SHOW_NO_TIPS = False
SHOW_LEGEND = True
SHOW_AXIS_LABELS = False
STARTING_LOG_PRICE = -2

# Define the data folder path
data_folder = Path(__file__).parent / "data"

# Load the three benchmark result files
df_init = pd.read_csv(data_folder / "test_init.csv")
df_post = pd.read_csv(data_folder / "test_post.csv")
df_post_no_tips = pd.read_csv(data_folder / "test_post_no_tips.csv")

# Get the maximum number of problems (use the largest dataset)
max_problems = max(len(df_init), len(df_post), len(df_post_no_tips))


# Function to compute cumulative success at different budget thresholds
def compute_cumulative_success(df, budget_thresholds):
    """
    For each budget threshold, count how many problems were solved
    with price <= threshold.
    """
    successes = []
    for budget in budget_thresholds:
        # Count problems where success=True and price <= budget
        # Missing rows are counted as failures (not in the dataframe)
        solved = df[(df["success"] == True) & (df["price"] <= budget)]
        successes.append(len(solved))
    return successes


# Create budget thresholds on a log scale
budget_thresholds = np.logspace(
    STARTING_LOG_PRICE, 0, 100
)  # From 0.0001 to 1.0


mpl.use("pgf")
mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{libertine}\usepackage[libertine]{newtxmath}",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


# Compute cumulative successes for each agent
success_init = compute_cumulative_success(df_init, budget_thresholds)
success_post = compute_cumulative_success(df_post, budget_thresholds)
success_post_no_tips = (
    compute_cumulative_success(df_post_no_tips, budget_thresholds)
    if SHOW_NO_TIPS
    else None
)

# Create the plot with paper-quality formatting
# Figure size: 2.8 inches width, 2.0 inches height (smaller, same fonts)
fig, ax = plt.subplots(figsize=(2.4, 1.4))


ax.plot(
    budget_thresholds,
    success_init,
    label="Before training",
    linewidth=1.2,
)
ax.plot(
    budget_thresholds,
    success_post,
    label="After training",
    linewidth=1.2,
)
if SHOW_NO_TIPS and success_post_no_tips is not None:
    ax.plot(
        budget_thresholds,
        success_post_no_tips,
        label="After (no tips)",
        linewidth=1.2,
        color="coral",
        linestyle="dotted",
    )

ax.set_xscale("log")

# Add finer graduations on the log axis
ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
minor_ticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ax.xaxis.set_minor_locator(
    LogLocator(base=10.0, subs=minor_ticks, numticks=100)
)
ax.xaxis.set_minor_formatter(NullFormatter())

if SHOW_AXIS_LABELS:
    ax.set_xlabel(r"Budget per Problem (\$)", labelpad=8)
    ax.set_ylabel(r"Problems Solved", labelpad=11)
if SHOW_LEGEND:
    ax.legend(frameon=False)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.grid(True, which="minor", alpha=0.1, linewidth=0.3)

plt.tight_layout(pad=0.3)

# Save as PDF for LaTeX
output_path_pdf = Path(__file__).parent / "scaling-curve.pdf"
plt.savefig(
    output_path_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02
)
print(f"PDF plot saved to: {output_path_pdf}")

# Also save as PNG for quick viewing
output_path_png = Path(__file__).parent / "scaling-curve.png"
plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
print(f"PNG plot saved to: {output_path_png}")

# Print summary statistics
print("\nSummary Statistics:")
print(
    f"Initial Agent: {len(df_init)} problems, {df_init['success'].sum()} solved"
)
print(
    f"Post-training (with tips): {len(df_post)} problems, {df_post['success'].sum()} solved"
)
if SHOW_NO_TIPS:
    print(
        f"Post-training (no tips): {len(df_post_no_tips)} problems, {df_post_no_tips['success'].sum()} solved"
    )
