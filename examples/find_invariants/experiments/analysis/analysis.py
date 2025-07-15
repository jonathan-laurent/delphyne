# pyright: basic

from pathlib import Path

import pandas as pd  # type: ignore

DATA_DIR = Path(__file__).parent / "data"
BASELINE_DATA = DATA_DIR / "baseline.csv"
ABDUCTION_DATA = DATA_DIR / "abduction.csv"

RESULT_COLS = ["agent", "bench_name", "success", "price", "seed"]


def baseline_result_table():
    data = pd.read_csv(BASELINE_DATA)
    # Turn `model_name`, `temperature` and `max_feedback_cycles` into an
    # `agent` column
    data["agent"] = (
        "baseline-"
        + data["model_name"]
        + "-"
        + data["temperature"].astype(str)
        + "-"
        + data["max_feedback_cycles"].astype(str)
    )
    data = data[RESULT_COLS]
    return data


def abduction_result_table():
    data = pd.read_csv(ABDUCTION_DATA)
    # Ensure there is only one value for model_cycle
    assert data["model_cycle"].nunique() == 1
    # Turn `max_requests_per_attempt` and `temperature` into an `agent` column
    data["agent"] = (
        "abduction-"
        + data["max_requests_per_attempt"].astype(str)
        + "-"
        + data["temperature"].astype(str)
    )
    data = data[RESULT_COLS]
    return data


def results_table():
    # Concatenate baseline and abduction data
    baseline_data = baseline_result_table()
    abduction_data = abduction_result_table()
    data = pd.concat([baseline_data, abduction_data], ignore_index=True)
    return data


def apply_cutout_price(data, price_limit):
    # For every row where price exceeds the limit, set `success` to
    # False and `price` to the limit
    data.loc[data["price"] > price_limit, ["success", "price"]] = [
        False,
        price_limit,
    ]


def max_price_spent(data):
    """
    Returns the maximum price spent on a problem.
    """
    return data["price"].max()


def analyze_table(data):
    """
    Take a result table and return a table that says, for each agent, how many
    problems it solved, and what average and median price was spent per problem.
    For each reported quantity, also report the standard deviation across seeds.

    Also, put all price in cents for readability.
    """
    # Group by agent and seed to get per-seed statistics
    seed_stats = (
        data.groupby(["agent", "seed"])
        .agg(
            {
                "success": "sum",  # Number of problems solved per seed
                "price": [
                    "mean",
                    "median",
                ],  # Average and median price per seed
            }
        )
        .reset_index()
    )

    # Flatten column names
    seed_stats.columns = [
        "agent",
        "seed",
        "problems_solved",
        "avg_price",
        "median_price",
    ]

    # Now group by agent to get statistics across seeds
    result = (
        seed_stats.groupby("agent")
        .agg(
            {
                "problems_solved": ["mean", "std"],
                "avg_price": ["mean", "std"],
                "median_price": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # Flatten and rename columns for clarity
    result.columns = [
        "agent",
        "problems_solved_mean",
        "problems_solved_std",
        "avg_price_mean",
        "avg_price_std",
        "median_price_mean",
        "median_price_std",
    ]

    # Convert all price columns to cents for readability
    price_columns = [
        "avg_price_mean",
        "avg_price_std",
        "median_price_mean",
        "median_price_std",
    ]
    for col in price_columns:
        result[col] = result[col] * 100
    return result


def produce_latex_table(analysis_result):
    # Takes the data resulting from `analyze_table` and produces a LaTeX table,
    # with one row per agent. Use \pm to show standard deviations and report two
    # significant digits for every measure.

    def format_with_std(mean_val, std_val, is_price=False):
        """Format a value with its standard deviation."""
        if is_price:
            # Use two decimal places for prices in cents
            if pd.isna(std_val):
                std_val = 0.0
            return f"{mean_val:.2f} $\\pm$ {std_val:.2f}"
        else:
            # Use one decimal place for problems solved
            if pd.isna(std_val):
                std_val = 0.0
            return f"{mean_val:.1f} $\\pm$ {std_val:.1f}"

    latex_lines = []
    latex_lines.append("\\begin{tabular}{|l|c|c|c|}")
    latex_lines.append("\\hline")
    latex_lines.append(
        "Agent & Problems Solved & Avg Price (cents) & Median Price (cents) \\\\"
    )
    latex_lines.append("\\hline")

    for _, row in analysis_result.iterrows():
        agent = row["agent"].replace(
            "_", "\\_"
        )  # Escape underscores for LaTeX

        problems_solved = format_with_std(
            row["problems_solved_mean"],
            row["problems_solved_std"],
            is_price=False,
        )
        avg_price = format_with_std(
            row["avg_price_mean"], row["avg_price_std"], is_price=True
        )
        median_price = format_with_std(
            row["median_price_mean"], row["median_price_std"], is_price=True
        )

        latex_lines.append(
            f"{agent} & {problems_solved} & {avg_price} & {median_price} \\\\"
        )

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")

    return "\n".join(latex_lines)


def main():
    data = results_table()
    apply_cutout_price(data, 0.2)
    analysis_result = analyze_table(data)
    print("Analysis results:")
    print(analysis_result)
    print("\nLaTeX table:")
    print(produce_latex_table(analysis_result))


if __name__ == "__main__":
    main()
