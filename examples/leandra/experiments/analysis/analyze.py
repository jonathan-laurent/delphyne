"""
Analyze Computation Scaling Curves for MiniF2F Test
"""

# pyright: ignore

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Toggle display of the "no tips" curve
SHOW_NO_TIPS = False
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

# Compute cumulative successes for each agent
success_init = compute_cumulative_success(df_init, budget_thresholds)
success_post = compute_cumulative_success(df_post, budget_thresholds)
success_post_no_tips = (
    compute_cumulative_success(df_post_no_tips, budget_thresholds)
    if SHOW_NO_TIPS
    else None
)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(
    budget_thresholds,
    success_init,
    label="Initial Agent",
    marker="o",
    markersize=3,
    linewidth=2,
)
plt.plot(
    budget_thresholds,
    success_post,
    label="Post-training (with tips)",
    marker="s",
    markersize=3,
    linewidth=2,
)
if SHOW_NO_TIPS and success_post_no_tips is not None:
    plt.plot(
        budget_thresholds,
        success_post_no_tips,
        label="Post-training (no tips)",
        marker="^",
        markersize=3,
        linewidth=2,
    )

plt.xscale("log")
plt.xlabel("Budget (price)", fontsize=12)
plt.ylabel("Number of Problems Solved", fontsize=12)
plt.title("Proving Agent Performance: Problems Solved vs Budget", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
output_path = Path(__file__).parent / "scaling-curve.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {output_path}")

# Display the plot
plt.show()

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
