import pandas as pd
import numpy as np
from itertools import combinations
from typing import List
from src.metrics.cosine import cosine_distance, calculate_dispersion 
from typing import List, Tuple

def stability_between_runs(master_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Measures the semantic shift between each prompt for each run using cosine distance.

    Args:
        master_df (pd.Dataframe):  of DataFrames with the main data points.
    Returns:
        pd.DataFrame: A dataframe where each row represents the stability of each prompt.
    '''
    dispersion_series = master_df.groupby('prompt_id')['embedding_response'].apply(
        lambda x : calculate_dispersion(np.stack(x.values))
    )

    master_df['dispersion'] = master_df['prompt_id'].map(dispersion_series)

    return master_df


def intra_case_shift_analysis(
    df: pd.DataFrame, 
    group_columns: List[str], 
    dimensions: List[str]
) -> pd.DataFrame:
    """
    Computes pairwise semantic shifts within defined groups to measure 
    consistency across experimental dimensions.

    This analysis creates a pairwise comparison for every prompt within 
    the same scenario and language, calculating the cosine distance and 
    tracking whether samples share the same categorical attributes.

    Args:
        df (pd.DataFrame): Consolidated dataset containing embeddings 
                           and experimental dimensions.
        group_columns (List[str]): Columns used to group the analysis 
                                   (e.g., ['scenario', 'lang']).
        dimensions (List[str]): Dimensions to track for attribute similarity.

    Returns:
        pd.DataFrame: A dataframe where each row represents a pairwise 
                      comparison (shift) between two prompts.
    """
    results = []
    grouped = df.groupby(group_columns)

    for (scenario, lang), case_df in grouped:
        case_df = case_df.reset_index(drop=True)
        
       
        for i, j in combinations(range(len(case_df)), 2):
            row_i = case_df.iloc[i]
            row_j = case_df.iloc[j]

            shift = cosine_distance(
                np.array(row_i["embedding_response"]),
                np.array(row_j["embedding_response"])
            )

            result = {
                "scenario": scenario,
                "language": lang,
                "shift": shift
            }

        
            for dim in dimensions:
                result[f"{dim}_same"] = (row_i[dim] == row_j[dim])

            results.append(result)

    return pd.DataFrame(results)



def controlled_dimension_shift_analysis(
    df: pd.DataFrame,
    dimensions: List[str],
    fixed_columns: Tuple[str, ...] = ("scenario", "lang")
) -> pd.DataFrame:
    """
    Measures the semantic shift induced by a specific dimension while 
    controlling for others.

    For each target dimension, the function groups data by fixed columns and 
    all other dimensions. It then calculates the cosine distance between pairs 
    that differ only in the target dimension value.

    Args:
        df (pd.DataFrame): Dataset containing embeddings and dimension columns.
        dimensions (List[str]): List of experimental dimensions to analyze.
        fixed_columns (Tuple[str, ...]): Base columns to group by (e.g., scenario, lang).

    Returns:
        pd.DataFrame: A comparison table of semantic shifts (cosine distance) 
                      attributed to changes in target dimension values.
    """
    results = []

    for target_dim in dimensions:
        
        other_dims = [d for d in dimensions if d != target_dim]
        grouping_cols = list(fixed_columns) + other_dims
        grouped = df.groupby(grouping_cols)

        for _, group_df in grouped:
            group_df = group_df.reset_index(drop=True)
            
            for i, j in combinations(range(len(group_df)), 2):
                row_i, row_j = group_df.iloc[i], group_df.iloc[j]

                val_i, val_j = row_i[target_dim], row_j[target_dim]

                if val_i == val_j:
                    continue

                shift = cosine_distance(
                    np.array(row_i["embedding_response"]),
                    np.array(row_j["embedding_response"])
                )

                results.append({
                    "scenario": row_i["scenario"],
                    "language": row_i["lang"],
                    "dimension": target_dim,
                    "value_i": val_i,
                    "value_j": val_j,
                    "comparison": " ↔ ".join(
                        sorted([str(val_i), str(val_j)])
                    ),
                    "shift": shift,
                    **{d: row_i[d] for d in other_dims}
                })

    return pd.DataFrame(results)






def build_pairwise_shift_matrix(
    results_df: pd.DataFrame,
    target_dimension: str,
    scenario: str,
    language: str
) -> pd.DataFrame:
    """
    Builds a pairwise semantic shift matrix for a given dimension, scenario,
    and language.

    Each cell (i, j) contains the mean semantic shift observed between the
    values i and j of the target dimension. Shifts are aggregated across all
    matching comparisons in the input dataframe. The resulting matrix is
    symmetric, and diagonal entries are set to zero.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing pairwise shift measurements. Expected columns
        include: 'dimension', 'scenario', 'language', 'value_i',
        'value_j', and 'shift'.

    target_dimension : str
        Dimension whose value-to-value semantic shifts will be analyzed.

    scenario : str
        Scenario to filter before constructing the matrix.

    language : str
        Language to filter before constructing the matrix.

    Returns
    -------
    pd.DataFrame
        Symmetric matrix where rows and columns correspond to dimension
        values and each entry contains the mean semantic shift between
        the corresponding pair of values.
    """

    subset = results_df[
        (results_df["dimension"] == target_dimension)
        & (results_df["scenario"] == scenario)
        & (results_df["language"] == language)
    ].copy()

    values = sorted(
        set(subset["value_i"]) | set(subset["value_j"])
    )

    mean_matrix = pd.DataFrame(
        np.nan,
        index=values,
        columns=values
    )

    for v1 in values:
        for v2 in values:

            if v1 == v2:
                mean_matrix.loc[v1, v2] = 0.0
                continue

            pair_subset = subset[
                (
                    (subset["value_i"] == v1)
                    & (subset["value_j"] == v2)
                )
                |
                (
                    (subset["value_i"] == v2)
                    & (subset["value_j"] == v1)
                )
            ]

            if not pair_subset.empty:
                mean_matrix.loc[v1, v2] = pair_subset["shift"].mean()

    return mean_matrix

def qualitative_examples(
    results_df: pd.DataFrame,
    original_df: pd.DataFrame,
    dimensions: list[str],
    response_column: str = "response",
    n_examples: int = 5,
    mode: str = "top"
):
    """
    Retrieves and displays qualitative examples associated with semantic shifts
    identified by the controlled dimension shift analysis.

    For each target dimension, the function selects a subset of pairwise
    comparisons according to one of three strategies:

    - "top":
        Comparisons exhibiting the largest semantic shifts.
    - "bottom":
        Comparisons exhibiting the smallest semantic shifts.
    - "representative":
        Comparisons whose shift magnitude is closest to the Expected Pairwise
        Shift (EPS) of the target dimension.


    Parameters
    ----------
    results_df : pd.DataFrame
        Output of ``controlled_dimension_shift_analysis``. Each row represents
        a pairwise comparison between two responses differing only in the
        target experimental dimension.

    original_df : pd.DataFrame
        Original dataset containing generated responses and experimental
        metadata.

    dimensions : list[str]
        List of experimental dimensions used in the study.

    response_column : str, default="response"
        Name of the column containing the textual response to display.

    n_examples : int, default=5
        Number of examples to retrieve for each target dimension.

    mode : {"top", "bottom", "representative"}, default="top"
        Strategy used to select comparisons:
            - "top": largest shifts
            - "bottom": smallest shifts
            - "representative": shifts closest to the dimension EPS

    Returns
    -------
    None
        Prints the selected comparisons together with their associated
        experimental conditions and responses.
    """

    for dimension in results_df["dimension"].unique():

        df_dim = results_df[
            results_df["dimension"] == dimension
        ].copy()

        if mode == "top":

            selected = (
                df_dim
                .sort_values("shift", ascending=False)
                .head(n_examples)
            )

        elif mode == "bottom":

            selected = (
                df_dim
                .sort_values("shift", ascending=True)
                .head(n_examples)
            )

        elif mode == "representative":

            eps = df_dim["shift"].mean()

            selected = (
                df_dim
                .assign(
                    deviation=lambda x:
                    (x["shift"] - eps).abs()
                )
                .sort_values("deviation")
                .head(n_examples)
            )

        else:

            raise ValueError(
                "mode must be one of: "
                "'top', 'bottom', 'representative'"
            )

        print("\n")
        print("=" * 120)
        print(f"TARGET DIMENSION: {dimension}")
        print(f"SELECTION MODE : {mode}")
        print("=" * 120)

        for _, row in selected.iterrows():

            target_dim = row["dimension"]

            filter_i = {
                "scenario": row["scenario"],
                "lang": row["language"],
                target_dim: row["value_i"]
            }

            filter_j = {
                "scenario": row["scenario"],
                "lang": row["language"],
                target_dim: row["value_j"]
            }

            for dim in dimensions:

                if dim == target_dim:
                    continue

                filter_i[dim] = row[dim]
                filter_j[dim] = row[dim]

            mask_i = pd.Series(
                True,
                index=original_df.index
            )

            mask_j = pd.Series(
                True,
                index=original_df.index
            )

            for col, val in filter_i.items():
                mask_i &= original_df[col] == val

            for col, val in filter_j.items():
                mask_j &= original_df[col] == val

            responses_i = (
                original_df
                .loc[mask_i, response_column]
                .tolist()
            )

            responses_j = (
                original_df
                .loc[mask_j, response_column]
                .tolist()
            )

            print("\n")
            print("-" * 120)

            print(
                f"Scenario   : {row['scenario']}"
            )

            print(
                f"Language   : {row['language']}"
            )

            print(
                f"Dimension  : {row['dimension']}"
            )

            print(
                f"Comparison : {row['comparison']}"
            )

            print(
                f"Shift      : {row['shift']:.4f}"
            )

            print("\nCONTROLLED CONDITIONS")
            print("=" * 60)

            for dim in dimensions:

                if dim == target_dim:
                    continue

                print(
                    f"{dim:<25}: {row[dim]}"
                )

            print("\nRESPONSES A")
            print("=" * 60)

            print(
                f"{target_dim}: {row['value_i']}"
            )

            for run_idx, response in enumerate(responses_i):

                print(f"\n[Run {run_idx}]")
                print(response)

            print("\nRESPONSES B")
            print("=" * 60)

            print(
                f"{target_dim}: {row['value_j']}"
            )

            for run_idx, response in enumerate(responses_j):

                print(f"\n[Run {run_idx}]")
                print(response)

            print("\nEND OF COMPARISON")
            print("-" * 120)

def compute_confidence_intervals(summary_df: pd.DataFrame, confidence_level: float = 0.95) -> pd.DataFrame:
    """
    Calculates the Standard Error (SE), Margin of Error, and Confidence Intervals (CI) 
    for the mean semantic shift based on the Central Limit Theorem.

    This function assumes a large sample size (count >> 30) per group, allowing the 
    use of the standard normal distribution (Z-score) approximation to estimate the 
    confidence interval bounds of the mean. Lower bounds are clipped at 0 to maintain 
    physical constraints of cosine distance.

    Args:
        summary_df (pd.DataFrame): Dataframe containing the descriptive statistics. 
            Must include 'mean', 'std', and 'count' columns.
        confidence_level (float): The target confidence level. Currently supports 0.95 
            (Z = 1.96). Defaults to 0.95.

    Returns:
        pd.DataFrame: A copy of the input dataframe with four additional columns:
            - 'se': Standard Error of the mean.
            - 'margin_of_error': The calculated margin of error.
            - 'ci_lower': The lower bound of the CI (clipped at 0).
            - 'ci_upper': The upper bound of the CI.
    """
  
    df = summary_df.copy()
    
   
    if confidence_level == 0.95:
        z_score = 1.96
    else:
     
        z_score = 1.96
        

    df['se'] = df['std'] / np.sqrt(df['count'])
  
    df['margin_of_error'] = z_score * df['se']
    
   
    df['ci_lower'] = (df['mean'] - df['margin_of_error']).clip(lower=0)
    df['ci_upper'] = df['mean'] + df['margin_of_error']
    
    return df

def dominant_dimension_per_scenario(
    results_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Identifies the dominant experimental dimension for each scenario based
    on the highest Expected Pairwise Shift (EPS).

    The function computes the mean semantic shift for every
    (scenario, dimension) pair, estimates 95% confidence intervals using
    `compute_confidence_intervals`, and reports the dimension with the
    highest mean shift within each scenario.

    Args:
        results_df (pd.DataFrame):
            Output of `controlled_dimension_shift_analysis`.
            Must contain at least:

            - 'scenario'
            - 'dimension'
            - 'shift'

    Returns:
        pd.DataFrame:
            DataFrame containing one row per scenario with:

            - 'scenario'
            - 'dimension' (dominant dimension)
            - 'mean' (EPS)
            - 'ci_lower'
            - 'ci_upper'
            - 'count'
    """

    summary = (
        results_df
        .groupby(["scenario", "dimension"])["shift"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary = compute_confidence_intervals(summary)

    winners = (
        summary
        .sort_values(
            ["scenario", "mean"],
            ascending=[True, False]
        )
        .groupby("scenario")
        .first()
        .reset_index()
    )

       

    return winners

def scenario_dimension_variability(
    results_df: pd.DataFrame
) -> pd.Series:
    """
    Computes the variability of dimension effects within each scenario.

    For every scenario, the function first computes the Expected Pairwise
    Shift (EPS) of each dimension and then calculates the standard deviation
    across dimensions. Higher values indicate that some dimensions produce
    substantially larger semantic shifts than others within the same scenario.

    Args:
        results_df (pd.DataFrame):
            Output of `controlled_dimension_shift_analysis`.
            Must contain:

            - 'scenario'
            - 'dimension'
            - 'shift'

    Returns:
        pd.Series:
            Standard deviation of EPS values across dimensions for each
            scenario, sorted in descending order.
    """

    pivot = (
        results_df
        .groupby(["scenario", "dimension"])["shift"]
        .mean()
        .unstack()
    )

    variability = (
        pivot
        .std(axis=1)
        .sort_values(ascending=False)
    )

   
    return variability

def scenario_dependence_by_dimension(
    results_df: pd.DataFrame
) -> pd.Series:
    """
    Measures how strongly the effect of each dimension varies across
    scenarios.

    For each (scenario, dimension) pair, the function computes the
    Expected Pairwise Shift (EPS). It then calculates the standard
    deviation of these EPS values across scenarios for every dimension.

    Higher values indicate that the impact of the dimension is highly
    dependent on the scenario. Lower values indicate that the dimension
    produces a relatively stable semantic shift across scenarios.

    Args:
        results_df (pd.DataFrame):
            Output of `controlled_dimension_shift_analysis`.
            Must contain:

            - 'scenario'
            - 'dimension'
            - 'shift'

    Returns:
        pd.Series:
            Standard deviation of EPS across scenarios for each
            dimension, sorted in descending order.
    """

    eps_table = (
        results_df
        .groupby(["scenario", "dimension"])["shift"]
        .mean()
        .unstack()
    )

    scenario_variability = (
        eps_table
        .std(axis=0)
        .sort_values(ascending=False)
    )

   
  

    return scenario_variability

def analyze_dimension_interactions(
    results_df: pd.DataFrame,
    dimensions: list[str]
) -> pd.DataFrame:
    """
    Measures how strongly the effect of one dimension depends on another.

    For each target dimension, the function computes the mean semantic
    shift (EPS) for every value of a moderator dimension and then
    calculates the standard deviation of those means.

    High values indicate that the effect of the target dimension varies
    substantially depending on the moderator. Low values indicate that
    the target dimension behaves consistently across moderator values.

    Args:
        results_df (pd.DataFrame):
            Output of `controlled_dimension_shift_analysis`.

            Must contain:

            - 'dimension'
            - 'shift'

            and one column for each dimension listed in `dimensions`.

        dimensions (list[str]):
            Names of the experimental dimensions.

    Returns:
        pd.DataFrame:
            Interaction table with the columns:

            - 'target'
            - 'moderator'
            - 'interaction_strength'
            - 'n_groups'

            sorted from strongest to weakest interaction.
    """

    interaction_results = []

    for target in dimensions:

        df_target = results_df[
            results_df["dimension"] == target
        ]

        for moderator in dimensions:

            if moderator == target:
                continue

            means = (
                df_target
                .groupby(moderator)["shift"]
                .mean()
            )

            interaction_results.append({
                "target": target,
                "moderator": moderator,
                "interaction_strength": means.std(),
                "n_groups": len(means)
            })

    interaction_df = pd.DataFrame(
        interaction_results
    ).sort_values(
        "interaction_strength",
        ascending=False
    )


    return interaction_df