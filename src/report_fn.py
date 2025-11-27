import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import os
import sys
from pathlib import Path
from scipy.stats import entropy

# Ensure project root is in path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
def eval_model_human(
    df: pd.DataFrame,
    problem_idx: str = "problem_idx",
    answer_idx: str = "idx_answer",
    model_name: str = "model_name",
    human_score: str = "points_judge_1",
    model_score: str = "model_answers",
    judge_model_name: str = "GPT-4",
    problem_set_name: str = "USAMO",
    save_figs: bool = False,
    output_dir: str = None,
    verbose: bool = False
) -> None:
    """
    Comprehensive evaluation of model–human scoring consistency.

    This function compares model-assigned and human-assigned scores
    across multiple problems, models, and attempts. It produces descriptive
    statistics, correlation analyses, and diagnostic plots to evaluate
    consistency between human and model scoring behavior.

    Sections Included
    -----------------
      I.  Global overview of human scores (model averages, per-problem summaries, heatmaps)
      II. Human = 0 subset analysis (agreement/disagreement statistics, calibration)
      III. Human > 0 subset analysis (correlation, bias, calibration)
      IV. Overall Analysis (correlation, bias, calibration)

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation dataset already loaded into memory.
        Must include columns such as `{problem_idx}`, `{model_name}`,
        `{human_score}`, and `{model_score}`.

    problem_idx : str, default="problem_idx"
        Column identifying the problem index or unique problem ID.

    answer_idx : str, default="idx_answer"
        Column identifying the attempt index (for multiple answers per problem).
        If missing, it will be created and sequentially numbered within each `{problem_idx}` group
        (starting from 0).

    model_name : str, default="model_name"
        Column identifying the model name or identifier (e.g., "GPT-4", "DeepSeek-R1-0528").

    human_score : str, default="points_judge_1"
        Column containing numeric human-assigned scores.

    model_score : str, default="model_answers"
        Column containing model-generated scores (e.g., "6/7" or numeric values).

    judge_model_name : str, default="GPT-4"
        Name of the *judge model* (the grading model assigning the scores).

    problem_set_name : str, default="USAMO"
        Identifier for the problem set or benchmark being analyzed (e.g., "USAMO", "AIME", "Putnam").

    save_figs : bool, default=True
        Whether to save all generated figures to the `figures/` directory.

    output_dir : str, default=None
        Custom output directory for reports and figures. If None, uses default 
        `summary/{judge_model_name}_{problem_set_name}` directory.

    verbose : bool, default=False
        Whether to print detailed progress information and analysis results.

    Expected Columns
    ----------------
    - {problem_idx}       : unique problem identifier (integer)
    - {answer_idx}        : attempt index for multiple answers (e.g., 0, 1, 2)
    - {model_name}        : model name or identifier (e.g., "GPT-4", "DeepSeek-R1-0528")
    - {human_score}       : numeric human-assigned score (e.g., 0–7)
    - {model_score}       : string model response or score text (e.g., "6/7")

    Optional Columns
    ----------------
    - {human_feedback}    : human qualitative feedback
    - {model_feedback}    : model qualitative feedback or answer content
    - judge_category      : label for the judging subset or evaluation type (e.g., "Judge_USAMO")
    - model_config        : model variant or configuration name
    - problem_statement   : full problem text or question content

    Output
    ------
    The function prints comprehensive summaries to the console and generates
    figures for each analysis section, providing a quantitative and visual
    evaluation of model–human scoring consistency.

    Notes
    -----
    - Figures are saved under the local directory `figures/` if `save_figs=True`.
    - Missing attempt index column (`{answer_idx}`) will be auto-numbered sequentially
      within each `{problem_idx}` (starting from 0).
    - The `{problem_statement}` column is optional and not directly used in statistical analysis,
      but can be retained for qualitative review or visualization.
    - `{model_score}` and `{human_score}` are required columns — the function will raise
      an error if either is missing.
    - Other columns will be auto-filled with empty strings or NaN if not present.
    - Handles both single- and multi-attempt datasets seamlessly.
    - Requires `{model_name}` and `{problem_idx}` for aggregation and visualization.
    """

    # === Helper function for structured printing ===
    def print_section(title: str, level: int = 1, show_separator: bool = True):
        """Print formatted section headers with optional separators."""
        if not verbose:
            return
        
        if level == 1:
            separator = "=" * 70
            if show_separator:
                print(separator)
            print(f"=== {title} ===")
            if show_separator:
                print(separator)
        elif level == 2:
            print(f"\n=== {title} ===")
        elif level == 3:
            print(f"\n--- {title} ---")
        else:
            print(f"\n{title}")

    def print_info(message: str, indent: int = 0):
        """Print informational messages with optional indentation."""
        if verbose:
            prefix = "  " * indent
            print(f"{prefix}{message}")

    def print_table(title: str, df_or_data, show_title: bool = True):
        """Print formatted tables with optional titles."""
        if not verbose:
            return
        
        if show_title:
            print(f"\n=== {title} ===")
        
        if hasattr(df_or_data, 'to_string'):
            print(df_or_data.to_string(index=False))
        else:
            print(df_or_data)

    # === MODEL–HUMAN CONSISTENCY EVALUATION PIPELINE ===
    print_section("MODEL–HUMAN CONSISTENCY EVALUATION PIPELINE")
    print_info(f"Data source          : DataFrame object")
    print_info(f"Total rows loaded    : {len(df):,}")
    print_info(f"Problem index column : {problem_idx}")
    print_info(f"Attempt index column : {answer_idx}")
    print_info(f"Model name column    : {model_name}")
    print_info(f"Human score column   : {human_score}")
    print_info(f"Model score column   : {model_score}")
    print_info(f"Save figures         : {'Yes' if save_figs else 'No'}")
    
    if verbose:
        print("-" * 70)
        print("This analysis includes:")
        print("  I.   Global overview of human scores")
        print("  II.  Overall Analysis")
        print("  III. Human = 0 subset analysis")
        print("  IV.  Human > 0 subset analysis")
        print("=" * 70, "\n")


    # --- Create output folder for summaries and figures ---
    if output_dir is not None:
        # Use analysis subfolder for reports and figures subfolder for figures
        # analysis_dir = os.path.join(output_dir, "analysis")
        analysis_dir = output_dir
        figures_dir = os.path.join(analysis_dir, "figures")
        summary_dir = analysis_dir  # Reports go in analysis/
        print_info(f"Analysis directory: {analysis_dir}")
        print_info(f"Figures directory: {figures_dir}")
        print_info(f"Summary directory: {summary_dir}")
    else:
        summary_dir = f"summary/{judge_model_name}_{problem_set_name}"
        figures_dir = summary_dir  # Default behavior for backward compatibility
    
    if save_figs:
        if output_dir is not None:
            os.makedirs(analysis_dir, exist_ok=True)
            os.makedirs(figures_dir, exist_ok=True)
        else:
            os.makedirs(summary_dir, exist_ok=True)

    # --------------------------------------------------------------
    # Helper Function: Save, Show, and Auto-Number Figures
    # --------------------------------------------------------------
    _fig_counter = 0
    saved_figs = {}

    def save_show_fig(*args):
        """
        Save figure with consistent naming (display removed).
        Supports:
            save_show_fig("Title")              → auto-number
            save_show_fig(3, "Title")           → manual-number
        """
        nonlocal _fig_counter, summary_dir, save_figs, saved_figs, figures_dir

        # --- Parse arguments ---
        if len(args) == 1:
            fig_num = None
            title = args[0]
        elif len(args) == 2:
            fig_num, title = args
        else:
            raise ValueError("Usage: save_show_fig('Title') or save_show_fig(fig_num, 'Title')")

        # --- Auto-increment ---
        if fig_num is None:
            _fig_counter += 1
            fig_num = _fig_counter
        else:
            _fig_counter = max(_fig_counter, fig_num)

        # --- Safe filename ---
        safe_name = (
            title.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )

        # --- Save figure ---
        if save_figs:
            # Use figures_dir if available, otherwise fall back to summary_dir
            fig_save_dir = figures_dir if 'figures_dir' in locals() else summary_dir
            os.makedirs(fig_save_dir, exist_ok=True)
            fig_path = os.path.join(fig_save_dir, f"fig_{fig_num:02d}_{safe_name}.png")
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            saved_figs[fig_num] = (title, fig_path)
            if verbose:
                print(f"[Saved] Figure {fig_num}: {title}")

        # Close the figure to free memory
        plt.close()

    def insert_figures(story, fig_nums=None, width=430, height=250, spacer=10):
        """
        Insert selected figures into PDF story using default 'Normal' style.
        -------------------------------------------------
        fig_nums : list[int] or None
            If None → insert all.
            If list → insert only specified figure numbers.
        """

        figs_to_insert = (
            sorted(saved_figs.items()) if fig_nums is None
            else [(i, saved_figs[i]) for i in fig_nums if i in saved_figs]
        )

        for fig_num, (title, fig_path) in figs_to_insert:
            if os.path.exists(fig_path):
                story.append(Paragraph(f"<b>Figure {fig_num}.</b> {title}", styles["Heading2"]))
                story.append(Image(fig_path, width=width, height=height))
                story.append(Spacer(1, spacer))

    # === Load and preprocess data ===
    df = df.copy()
    # Handle model_score column depending on its type
    if np.issubdtype(df[model_score].dtype, np.number):
      # Case 1: model_score is already numeric (e.g., judge scores like points_judge_2)
      # → Just copy it into a numeric column for consistency
      df["model_score_num"] = df[model_score].astype(float)
    else:
      # Case 2: model_score is a string (e.g., "5/7")
      # → Convert to string explicitly, then extract the numeric part before the slash
      df[model_score] = df[model_score].astype(str)
      df["model_score_num"] = (
          df[model_score]
          .str.extract(r"(\d+)/")[0]   # extract the first number (e.g., "5" from "5/7")
          .astype(float)
      )

    # Handle missing answer index column
    if answer_idx not in df.columns:
        print_info(f"'{answer_idx}' column missing — auto-generating sequential indices per problem (starting from 0).")
        # If the column is missing, assign sequential indices per problem starting from 0
        # Uses groupby(problem_idx).cumcount() if problem identifiers exist;
        # otherwise assigns global indices from 0 to len(df) - 1.
        df[answer_idx] = (
            df.groupby(problem_idx).cumcount()
            if problem_idx in df.columns else np.arange(len(df))
        )

    # Convert both human and model scores to numeric and drop invalid rows
    x = pd.to_numeric(df[human_score], errors="coerce")
    y = pd.to_numeric(df["model_score_num"], errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    df = df.loc[mask].copy()

    print_info(f"Data loaded successfully: {len(df)} valid rows.\n")

    # ==============================================================
    # === SECTION I: Global Overview of Human Scores ===
    # ==============================================================
    print_section("SECTION I: Global Overview of Human Scores")

    # --- Average Human Score by Model ---
    model_avg = (
        df.groupby(model_name)[human_score]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={human_score: "avg_human_score"})
    )

    print_table("Average Human Score by Model", model_avg.round(3))

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=model_name,
        y="avg_human_score",
        hue=model_name,
        data=model_avg,
        palette="Blues_d",
        legend=False
    )
    plt.xlabel("Model Name")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Human Score")
    plt.title("Average Human Score by Model")
    plt.tight_layout()
    save_show_fig("Average Human Score by Model")

    # --- Average Human Score per Problem ---
    best_attempts = (
        df.loc[df.groupby([model_name, problem_idx])[human_score].idxmax()]
        .reset_index(drop=True)
    )

    problem_avg = (
        best_attempts.groupby(problem_idx)[human_score]
        .mean()
        .reset_index()
        .rename(columns={human_score: "avg_human_score"})
    )

    print_table("Average Human Score per Problem", problem_avg.round(3))

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=problem_idx,
        y="avg_human_score",
        hue=problem_idx,
        data=problem_avg,
        palette="viridis",
        legend=False

    )
    plt.xlabel("Problem ID")
    plt.ylabel("Average Score (0–7)")
    plt.title("Average Human Score per Problem (Best Attempt per Model)")
    plt.ylim(0, 7)
    plt.tight_layout()
    save_show_fig("Average Human Score per Problem (Best Attempt per Model)")

    # --- Heatmap of Model × Problem ---
    heatmap_data = best_attempts.pivot(index=model_name, columns=problem_idx, values=human_score)
    print_table("Model × Problem Score Heatmap", heatmap_data.round(2).fillna("-").head(10))

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data, annot=True,
        cmap="viridis", cbar_kws={"label": "Score (0–7)"}
    )
    plt.title("Heatmap of Model Performance across Problems")
    plt.xlabel("Problem ID")
    plt.ylabel("Model Name")
    plt.tight_layout()
    save_show_fig("Heatmap of Model Performance across Problems")

    # ==============================================================
    # === SECTION II: Overall (Global) Analysis ===
    # ==============================================================

    # --- Descriptive Summary (Human vs Model) ---
    describe_points = df[human_score].describe()
    describe_answers = df["model_score_num"].describe()
    summary_stats = pd.DataFrame({
        "Metric": describe_points.index,
        "Human Score": describe_points.values,
        "Model Score": describe_answers.values
    }).round(3)
    print_table("Descriptive Summary (Human vs Model)", summary_stats)

    # === Alignment, Regression & Agreement Metrics ===
    pearson = x.corr(y, "pearson")
    spearman = x.corr(y, "spearman")
    kendall = x.corr(y, "kendall")
    xi, yi = x.round().astype(int), y.round().astype(int)
    kappa_linear = cohen_kappa_score(xi, yi, weights="linear")
    kappa_quadratic = cohen_kappa_score(xi, yi, weights="quadratic")

    slope, intercept = np.polyfit(x, y, 1)
    r2 = pearson ** 2

    align_df = pd.DataFrame({
        "Metric": [
            "Pearson Correlation",
            "Spearman Rank",
            "Kendall Tau",
            "Cohen's κ (Linear)",
            "Cohen's κ (Quadratic)",
            "Slope",
            "Intercept",
            "R²",
        ],
        "Value": [
            pearson, spearman, kendall,
            kappa_linear, kappa_quadratic,
            slope, intercept, r2,
        ],
    })
    print_table("Alignment, Regression & Agreement Metrics (Overall)", align_df.round(3))

    # === Error & Accuracy Metrics ===
    mae = np.mean(np.abs(y - x))
    rmse = np.sqrt(np.mean((y - x) ** 2))
    abs_diff = np.abs(y - x)
    acc_exact = (abs_diff == 0).mean()
    acc_within_1 = (abs_diff <= 1).mean()
    acc_within_2 = (abs_diff <= 2).mean()

    err_df = pd.DataFrame({
        "Metric": [
            "MAE",
            "RMSE",
            "Accuracy",
            "Accuracy |Δ| ≤ 1",
            "Accuracy |Δ| ≤ 2",
        ],
        "Value": [
            mae,
            rmse,
            acc_exact,
            acc_within_1,
            acc_within_2,
        ],
    })
    print_table("Error and Accuracy Metrics (Overall)", err_df.round(3))


    # === Distributional Metrics ===
    p_human = pd.Series(x).value_counts(normalize=True).sort_index()
    p_model = pd.Series(y).value_counts(normalize=True).reindex(p_human.index, fill_value=0)
    H_human = entropy(p_human, base=2)
    H_model = entropy(p_model, base=2)
    RJE = H_model / H_human if H_human > 0 else np.nan

    all_scores = sorted(set(p_human.index) | set(p_model.index))
    p_h = p_human.reindex(all_scores, fill_value=0)
    p_m = p_model.reindex(all_scores, fill_value=0)

    # 2) smooth + renormalize to avoid log(0) and floating issues
    eps = 1e-9
    p_h = (p_h + eps) / (p_h + eps).sum()
    p_m = (p_m + eps) / (p_m + eps).sum()

    # 3) compute JSD in bits
    M = 0.5 * (p_h + p_m)
    JSD = 0.5 * (entropy(p_h, M, base=2) + entropy(p_m, M, base=2))

    VCR = np.var(y) / np.var(x)

    dist_df = pd.DataFrame({
        "Metric": [
            "Relative Judge Entropy (RJE)",
            "Jensen–Shannon Divergence (JSD)",
            "Variance Collapse Ratio (VCR)"
        ],
        "Value": [RJE, JSD, VCR]
    })
    print_table("Distributional Metrics (Overall)", dist_df.round(4))

        # --- Scatter Plot ---
    plt.figure(figsize=(6, 6))
    size_series = (
        df.groupby([human_score, "model_score_num"])[model_score]
        .transform("count").fillna(1) * 20
    )
    df.plot.scatter(
        x=human_score, y="model_score_num",
        alpha=0.5, s=size_series, color="steelblue"
    )
    plt.xlabel("Human Score")
    plt.ylabel("Model Score (numerator)")
    plt.title("Distribution of Model Scores by Human Score")
    plt.tight_layout()
    save_show_fig("Distribution of Model Scores by Human Score")

    # --- Confusion Matrix ---
    plt.figure(figsize=(6, 5))
    x_cm = pd.to_numeric(df[human_score], errors="coerce")
    y_cm = pd.to_numeric(df["model_score_num"], errors="coerce")
    mask_cm = ~x_cm.isna() & ~y_cm.isna()
    x_cm, y_cm = x_cm[mask_cm], y_cm[mask_cm]
    x_cm = x_cm.round().astype(int)
    y_cm = y_cm.round().astype(int)
    score_labels = sorted(set(x_cm) | set(y_cm))
    cm = confusion_matrix(x_cm, y_cm, labels=score_labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums != 0)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=score_labels, yticklabels=score_labels)
    plt.xlabel("Model Score (y)")
    plt.ylabel("Human Score (x)")
    plt.title("Confusion Matrix: Human vs Model Scores (Integer Scale)")
    plt.tight_layout()
    save_show_fig("Confusion Matrix: Human vs Model Scores (Integer Scale)")

    # --- Bias & Calibration Plots ---
    bias_all = y - x
    plt.figure(figsize=(7, 4))
    sns.histplot(bias_all, bins=20, kde=True)
    plt.axvline(0, color="red", linestyle="--")
    plt.title("Bias Distribution (Model - Human) | Overall")
    plt.xlabel("Bias")
    plt.ylabel("Frequency")
    plt.tight_layout()
    save_show_fig("Bias Distribution (Model - Human) | Overall")

    calib_all = (
        pd.DataFrame({"human": x, "model": y})
        .groupby("human")
        .agg(count=("model", "size"),
             mean_model=("model", "mean"),
             median_model=("model", "median"))
        .reset_index()
    )
    print_table("Calibration by Human Score (Overall)", calib_all.round(3))

    plt.figure(figsize=(7, 5))
    # Ensure human column is numeric to avoid categorical plotting warnings
    calib_all["human"] = pd.to_numeric(calib_all["human"], errors="coerce")
    sns.lineplot(data=calib_all, x="human", y="mean_model", marker="o", label="Mean Model")
    sns.lineplot(data=calib_all, x="human", y="median_model", marker="s", label="Median Model")
    plt.plot(
        [calib_all["human"].min(), calib_all["human"].max()],
        [calib_all["human"].min(), calib_all["human"].max()],
        "k--", label="y = x (perfect)"
    )
    plt.xlabel("Human Score")
    plt.ylabel("Model Score")
    plt.title("Calibration of Model vs Human Scores (Overall)")
    plt.legend()
    plt.tight_layout()
    save_show_fig("Calibration of Model vs Human Scores (Overall)")
    # --- Mean Error by Human Score (within Calibration Section) ---
    mean_error_by_score = (
        pd.DataFrame({"human": x, "model": y})
        .groupby("human")[["human", "model"]]
        .apply(lambda g: np.mean(g["model"] - g["human"]))
        .rename("mean_error")
        .reset_index()
    )

    print_table("Mean Error by Human Score (Model - Human)", mean_error_by_score.round(3))

    plt.figure(figsize=(7, 4))
    # Ensure human column is numeric to avoid categorical plotting warnings
    mean_error_by_score["human"] = pd.to_numeric(mean_error_by_score["human"], errors="coerce")
    sns.barplot(
        data=mean_error_by_score,
        x="human",
        y="mean_error",
        hue="human",
        palette="coolwarm",
        legend=False
    )
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Human Score")
    plt.ylabel("Mean Error (Model - Human)")
    plt.title("Mean Error by Human Score (Overall)")
    plt.tight_layout()
    save_show_fig("Mean Error by Human Score (Overall)")

    # ==============================================================
    # === SECTION III-A: Layer I – Zero-Score Discrimination (Human = 0)
    # ==============================================================
    print_section("SECTION III-A: Layer I – Zero-Score Discrimination (Human = 0)")

    # --- Subset selection ---
    mask_h0 = (x == 0)
    x0, y0 = x[mask_h0], y[mask_h0]
    total_0 = len(x0)
    model_0 = int((y0 == 0).sum())
    model_pos = int((y0 > 0).sum())

    # --- Basic ratios ---
    p_model_0 = model_0 / total_0 if total_0 > 0 else np.nan
    p_model_pos = model_pos / total_0 if total_0 > 0 else np.nan

    print_info(f"Total Human=0 Cases : {total_0}")
    print_info(f"Model predicted 0   : {model_0} ({p_model_0:.2%})")
    print_info(f"Model predicted >0  : {model_pos} ({p_model_pos:.2%})")

    # --- Bias and dispersion ---
    bias_h0 = y0 - x0
    mean_pred = y0.mean()
    std_pred = y0.std(ddof=1) if total_0 > 1 else np.nan
    skew_pred = (pd.Series(y0).skew() if total_0 > 2 else np.nan)
    kurt_pred = (pd.Series(y0).kurtosis() if total_0 > 3 else np.nan)

    summary_h0 = pd.DataFrame({
        "Metric": [
            "Total Cases",
            "Model=0 Count", "Model>0 Count",
            "Model=0 Proportion", "Model>0 Proportion",
            "Mean Predicted Score", "Std of Predicted Score",
            "Skewness", "Kurtosis"
        ],
        "Value": [
            total_0, model_0, model_pos,
            p_model_0, p_model_pos,
            mean_pred, std_pred, skew_pred, kurt_pred
        ]
    })
    print_table("Summary Statistics (Human Score = 0)", summary_h0.round(3))

    # --- Agreement vs Disagreement bar chart ---
    plt.figure(figsize=(6, 4))
    agreement_data = pd.DataFrame({
        "Category": ["Model=0", "Model>0"],
        "Count": [model_0, model_pos]
    })
    sns.barplot(
        data=agreement_data,
        x="Category", y="Count",
        hue="Category", palette="Blues", legend=False
    )
    total = model_0 + model_pos
    for i, row in agreement_data.iterrows():
        percent = (row["Count"] / total) * 100 if total > 0 else 0
        plt.text(i, row["Count"] + 0.5, f"{row['Count']} ({percent:.1f}%)",
                 ha="center", va="bottom", fontsize=10, color="black")
    plt.ylabel("Count")
    plt.title("Layer I-A: Agreement vs Disagreement (Human = 0)")
    plt.tight_layout()
    save_show_fig("Layer I-A Agreement vs Disagreement (Human=0)")

    # --- Distribution of non-zero predictions (false positives) ---
    if model_pos > 0:
        nonzero_dist = y0[y0 > 0].value_counts().sort_index()
        mean_non0 = y0[y0 > 0].mean()
        print_table("Distribution of Non-Zero Predictions (False Positives)", nonzero_dist)
        print_info(f"Average Non-Zero Model Score: {mean_non0:.3f}")

        plt.figure(figsize=(6, 4))
        ax = sns.countplot(x=y0[y0 > 0].round().astype(int), color="#F28E2B")
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 0.5,
                    f"{int(height)}", ha="center", va="bottom", fontsize=9)
        plt.xlabel("Model Score (non-zero)")
        plt.ylabel("Count")
        plt.title("Layer I-A: Distribution of Non-Zero Predictions (Human=0)")
        plt.tight_layout()
        save_show_fig("Layer I-A Nonzero Distribution (Human=0)")

    # ==============================================================
    # === SECTION IV: Layer I-B – Valid-Answer Calibration (Human > 0)
    # ==============================================================
    print_section("SECTION IV: Layer I-B – Valid-Answer Calibration (Human > 0)")

    mask_hpos = x > 0
    x_pos, y_pos = x[mask_hpos], y[mask_hpos]
    bias_pos = y_pos - x_pos

    # --------------------------------------------------------------
    # (1) Alignment, Agreement & Linear Fit Metrics
    # --------------------------------------------------------------
    pearson_pos = x_pos.corr(y_pos, "pearson")
    spearman_pos = x_pos.corr(y_pos, "spearman")
    kendall_pos = x_pos.corr(y_pos, "kendall")

    xi, yi = x_pos.round().astype(int), y_pos.round().astype(int)
    kappa_lin_pos = cohen_kappa_score(xi, yi, weights="linear")
    kappa_quad_pos = cohen_kappa_score(xi, yi, weights="quadratic")

    slope_pos, intercept_pos = np.polyfit(x_pos, y_pos, 1)
    r2_pos = pearson_pos ** 2

    align_pos_df = pd.DataFrame({
        "Metric": [
            "Pearson",
            "Spearman",
            "Kendall",
            "Cohen's κ (Linear)",
            "Cohen's κ (Quadratic)",
            "Slope",
            "Intercept",
            "R²"
        ],
        "Value": [
            pearson_pos,
            spearman_pos,
            kendall_pos,
            kappa_lin_pos,
            kappa_quad_pos,
            slope_pos,
            intercept_pos,
            r2_pos
        ]
    })
    print_table("Alignment & Linear Fit Metrics (Human > 0)", align_pos_df.round(3))

    # --------------------------------------------------------------
    # (2) Error & Accuracy Metrics
    # --------------------------------------------------------------
    mae_pos = np.mean(np.abs(bias_pos))
    rmse_pos = np.sqrt(np.mean(bias_pos ** 2))
    acc0_pos = (bias_pos == 0).mean()
    acc1_pos = (np.abs(bias_pos) <= 1).mean()
    acc2_pos = (np.abs(bias_pos) <= 2).mean()

    err_pos_df = pd.DataFrame({
        "Metric": [
            "MAE",
            "RMSE",
            "Accuracy",
            "Accuracy |Δ|≤1",
            "Accuracy |Δ|≤2",
        ],
        "Value": [
            mae_pos,
            rmse_pos,
            acc0_pos,
            acc1_pos,
            acc2_pos,
        ],
    })
    print_table("Error and Accuracy Metrics (Human > 0)", err_pos_df.round(3))


    # --------------------------------------------------------------
    # (3) Bias Descriptive Statistics (Human > 0)
    # --------------------------------------------------------------
    desc_bias_pos = pd.DataFrame({
        "Metric": [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
            "Accuracy",
            "Accuracy |Δ|≤1",
            "Accuracy |Δ|≤2",
        ],
        "Bias (Model - Human)": [
            len(bias_pos),
            bias_pos.mean(),
            bias_pos.median(),
            bias_pos.std(ddof=1),
            bias_pos.min(),
            bias_pos.max(),
            (bias_pos == 0).mean(),
            (bias_pos.abs() <= 1).mean(),
            (bias_pos.abs() <= 2).mean(),
        ],
    })

    print_table("Descriptive Statistics: Bias (Model - Human) | Human > 0", desc_bias_pos.round(3))


    # --------------------------------------------------------------
    # (4) Bias Distribution Plot
    # --------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    sns.histplot(bias_pos, bins=20, kde=True)
    plt.axvline(0, color="red", linestyle="--")
    plt.title("Layer I-B: Bias Distribution (Human > 0)")
    plt.xlabel("Bias (Model - Human)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    save_show_fig("Layer I-B Bias Distribution (Human>0)")

    # --------------------------------------------------------------
    # (5) Calibration by Human Score (Human > 0)
    # --------------------------------------------------------------
    calib_pos = (
        pd.DataFrame({"human": x_pos, "model": y_pos})
        .groupby("human")
        .agg(
            count=("model", "size"),
            mean_model=("model", "mean"),
            median_model=("model", "median")
        )
        .reset_index()
        .sort_values("human")
    )

    print_table("Calibration by Human Score (Human > 0)", calib_pos.round(3))

    plt.figure(figsize=(7, 5))
    # Ensure human column is numeric to avoid categorical plotting warnings
    calib_pos["human"] = pd.to_numeric(calib_pos["human"], errors="coerce")
    sns.lineplot(data=calib_pos, x="human", y="mean_model", marker="o", label="Mean Model")
    sns.lineplot(data=calib_pos, x="human", y="median_model", marker="s", label="Median Model")
    plt.plot(
        [calib_pos["human"].min(), calib_pos["human"].max()],
        [calib_pos["human"].min(), calib_pos["human"].max()],
        "k--", label="y = x (perfect)"
    )
    plt.xlabel("Human Score")
    plt.ylabel("Model Score")
    plt.title("Layer I-B: Calibration Curve (Human > 0)")
    plt.legend()
    plt.tight_layout()
    save_show_fig("Layer I-B Calibration Curve (Human>0)")

    # --------------------------------------------------------------
    # (6) Conditional Bias by Human Score
    # --------------------------------------------------------------
    cond_bias_pos = (
        pd.DataFrame({"human": x_pos, "bias": bias_pos})
        .groupby("human")["bias"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_bias", "std": "std_bias"})
    )

    print_table("Conditional Bias by Human Score", cond_bias_pos.round(3))

    plt.figure(figsize=(7, 4))
    # Ensure human column is numeric to avoid categorical plotting warnings
    cond_bias_pos["human"] = pd.to_numeric(cond_bias_pos["human"], errors="coerce")
    sns.barplot(
        data=cond_bias_pos,
        x="human",
        y="mean_bias",
        hue="human",
        palette="coolwarm",
        legend=False
    )
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Human Score")
    plt.ylabel("Mean Bias (Model - Human)")
    plt.title("Layer I-B: Mean Bias by Human Score")
    plt.tight_layout()
    save_show_fig("Layer I-B Mean Bias by Human Score")

    # ==============================================================
    # === SECTION V: Layer II – Band-wise Calibration Analysis ===
    # ==============================================================
    print_section("SECTION V: Layer II – Band-wise Calibration Analysis")

    bands = {
        "Low (1–2)": (1, 2),
        "Mid (3–5)": (3, 5),
        "High (6–7)": (6, 7),
    }

    band_rows = []

    for band_name, (lo_band, hi_band) in bands.items():
        mask_band = (x >= lo_band) & (x <= hi_band)
        xb_band, yb_band = x[mask_band], y[mask_band]
        if len(xb_band) < 3:
            continue
        if verbose:
            print(np.unique(yb_band, return_counts=True))
        # --- Alignment Metrics ---
        pearson_band = xb_band.corr(yb_band, "pearson")
        spearman_band = xb_band.corr(yb_band, "spearman")
        kendall_band = xb_band.corr(yb_band, "kendall")
        xi_band, yi_band = xb_band.round().astype(int), yb_band.round().astype(int)
        kappa_lin_band = cohen_kappa_score(xi_band, yi_band, weights="linear")
        kappa_quad_band = cohen_kappa_score(xi_band, yi_band, weights="quadratic")

        # --- Error & Bias Metrics ---
        slope_band, intercept_band = np.polyfit(xb_band, yb_band, 1)
        r2_band = pearson_band ** 2
        mae_band = np.mean(np.abs(yb_band - xb_band))
        rmse_band = np.sqrt(np.mean((yb_band - xb_band) ** 2))
        abs_diff_band = np.abs(yb_band - xb_band)
        acc1_band = (abs_diff_band <= 1).mean()
        acc2_band = (abs_diff_band <= 2).mean()

        bias_band = yb_band - xb_band
        bias_mean_band = bias_band.mean()
        bias_median_band = bias_band.median()
        bias_std_band = bias_band.std(ddof=1)
        bias_min_band = bias_band.min()
        bias_max_band = bias_band.max()

        # --- Distributional Metrics ---
        # Compute normalized frequency distributions (probabilities)
        p_h_band = pd.Series(xb_band).value_counts(normalize=True).sort_index()

        # Model’s own score distribution (do NOT align with human scores)
        p_m_band_raw = pd.Series(yb_band).value_counts(normalize=True).sort_index()

        # Compute Shannon entropies (in bits)
        Hh_band = entropy(p_h_band, base=2)      # Human score entropy
        Hm_band = entropy(p_m_band_raw, base=2)  # Model score entropy

        # Relative Judge Entropy (RJE): diversity ratio of model vs. human
        RJE_band = Hm_band / Hh_band if Hh_band > 0 else np.nan

        # Variance and Variance Collapse Ratio (sample variance, ddof=1)
        xb_arr = np.asarray(xb_band, dtype=float)
        yb_arr = np.asarray(yb_band, dtype=float)
        varx_band = np.var(xb_arr, ddof=1)
        vary_band = np.var(yb_arr, ddof=1)
        VCR_band = vary_band / varx_band if varx_band > 0 else np.nan

        # --- Append Row ---
        band_rows.append({
            "Band": band_name,
            "Count": len(xb_band),

            # Alignment metrics
            "Pearson": pearson_band,
            "Spearman": spearman_band,
            "Kendall": kendall_band,
            "κ (Linear)": kappa_lin_band,
            "κ (Quadratic)": kappa_quad_band,

            # Error & Bias metrics
            "Slope": slope_band,
            "Intercept": intercept_band,
            "R²": r2_band,
            "MAE": mae_band,
            "RMSE": rmse_band,
            "Mean Bias": bias_mean_band,
            "Median Bias": bias_median_band,
            "Std Bias": bias_std_band,
            "Min Bias": bias_min_band,
            "Max Bias": bias_max_band,
            "Acc |Δ|≤1": acc1_band,
            "Acc |Δ|≤2": acc2_band,

            # Distributional metrics
            "Var(X)": varx_band,
            "Var(Y)": vary_band,
            "RJE": RJE_band,
            "VCR": VCR_band,
        })

    band_df = pd.DataFrame(band_rows)


    # --------------------------------------------------------------
    # (A) Alignment Metrics
    # --------------------------------------------------------------
    print_table("Alignment Metrics (Band-wise)", 
                band_df[["Band", "Count", "Pearson", "Spearman", "Kendall",
                        "κ (Linear)", "κ (Quadratic)"]].round(3))

    # --------------------------------------------------------------
    # (B) Error & Bias Metrics
    # --------------------------------------------------------------
    print_table("Error & Bias Metrics (Band-wise)",
                band_df[["Band", "Count", "Slope", "Intercept", "R²", "MAE", "RMSE",
                        "Mean Bias", "Median Bias", "Std Bias",
                        "Min Bias", "Max Bias", "Acc |Δ|≤1", "Acc |Δ|≤2"]].round(3))

    # --------------------------------------------------------------
    # (C) Distributional Metrics
    # --------------------------------------------------------------
    print_table("Distributional Metrics (Band-wise)",
                band_df[["Band", "Count", "Var(X)", "Var(Y)", "RJE", "VCR"]].round(3))

    # --------------------------------------------------------------
    # (D) Visualization: Calibration Curves
    # --------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    for band_name, (lo_band, hi_band) in bands.items():
        mask_band = (x >= lo_band) & (x <= hi_band)
        xb_band, yb_band = x[mask_band], y[mask_band]
        if len(xb_band) < 2:
            continue
        calib_band = (
            pd.DataFrame({"human": xb_band, "model": yb_band})
            .groupby("human")[["human", "model"]]
            .agg(mean_model=("model", "mean"))
            .reset_index()
        )
        # Ensure human column is numeric to avoid categorical plotting warnings
        calib_band["human"] = pd.to_numeric(calib_band["human"], errors="coerce")
        sns.lineplot(
            data=calib_band,
            x="human",
            y="mean_model",
            marker="o",
            label=band_name,
        )

    plt.plot([x.min(), x.max()], [x.min(), x.max()], "k--", label="y = x (perfect)")
    plt.xlabel("Human Score")
    plt.ylabel("Mean Model Score")
    plt.title("Layer II: Band-wise Calibration Curves")
    plt.legend()
    plt.tight_layout()
    save_show_fig("Layer II Band-wise Calibration Curves")

    # --------------------------------------------------------------
    # (E) Visualization: Mean Bias by Band
    # --------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    sns.barplot(
        data=band_df,
        x="Band",
        y="Mean Bias",
        hue="Band",
        palette="coolwarm",
        legend=False,
    )
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Layer II: Mean Bias by Band")
    plt.ylabel("Mean Bias (Model – Human)")
    plt.tight_layout()
    save_show_fig("Layer II Mean Bias by Band")


    # ==============================================================
    # === LAYER III: Problem-wise Consistency Analysis ===
    # ==============================================================
    print_section("LAYER III: Problem-wise Consistency Analysis")

    PROB = problem_idx
    HUMAN = human_score
    MODEL = "model_score_num"

    results = []

    # ----------------------------------------------------------
    # --- Per-Problem Metric Computation ---
    # ----------------------------------------------------------
    for pid, g in df.groupby(PROB):
        x_prob = pd.to_numeric(g[HUMAN], errors="coerce")
        y_prob = pd.to_numeric(g[MODEL], errors="coerce")
        mask_prob = ~x_prob.isna() & ~y_prob.isna()
        x_prob, y_prob = x_prob[mask_prob], y_prob[mask_prob]
        if len(x_prob) < 3:
            continue

        # === Alignment Metrics ===
        pearson_prob = x_prob.corr(y_prob, method="pearson")
        spearman_prob = x_prob.corr(y_prob, method="spearman")
        kendall_prob = x_prob.corr(y_prob, method="kendall")

        xi_prob, yi_prob = x_prob.round().astype(int), y_prob.round().astype(int)
        kappa_lin_prob = cohen_kappa_score(xi_prob, yi_prob, weights="linear")
        kappa_quad_prob = cohen_kappa_score(xi_prob, yi_prob, weights="quadratic")

        # === Linear regression (slope, intercept, R²) ===
        if np.std(x_prob) > 0:
            slope_prob, intercept_prob = np.polyfit(x_prob, y_prob, 1)
            r2_prob = pearson_prob ** 2 if pearson_prob is not None else np.nan
        else:
            slope_prob, intercept_prob, r2_prob = np.nan, np.nan, np.nan

        # === Error & Accuracy Metrics ===
        bias_prob = (y_prob - x_prob).mean()
        mae_prob = np.mean(np.abs(y_prob - x_prob))
        rmse_prob = np.sqrt(np.mean((y_prob - x_prob) ** 2))
        acc_pm1_prob = np.mean(np.abs(y_prob - x_prob) <= 1)
        acc_pm2_prob = np.mean(np.abs(y_prob - x_prob) <= 2)

        # === Distributional Metrics ===
        var_h_prob, var_m_prob = np.var(x_prob), np.var(y_prob)
        vcr_prob = var_m_prob / var_h_prob if var_h_prob > 0 else np.nan

        # --- Entropy-based Diversity (independent supports) ---
        p_human_prob = pd.Series(x_prob).value_counts(normalize=True).sort_index()
        p_model_prob = pd.Series(y_prob).value_counts(normalize=True).sort_index()
        H_human_prob = entropy(p_human_prob, base=2)
        H_model_prob = entropy(p_model_prob, base=2)
        rje_prob = H_model_prob / H_human_prob if H_human_prob > 1e-9 else np.nan

        # --- Jensen–Shannon Divergence (union support + smoothing) ---
        eps = 1e-9
        all_scores = sorted(set(p_human_prob.index) | set(p_model_prob.index))
        p_h = p_human_prob.reindex(all_scores, fill_value=0)
        p_m = p_model_prob.reindex(all_scores, fill_value=0)
        p_h = (p_h + eps) / (p_h + eps).sum()
        p_m = (p_m + eps) / (p_m + eps).sum()

        m = 0.5 * (p_h + p_m)
        jsd_prob = 0.5 * (entropy(p_h, m, base=2) + entropy(p_m, m, base=2))
        rje_js_prob = 1 - jsd_prob

        # --- Collect Results ---
        results.append({
            "problem_idx": pid,
            "n": len(x_prob),
            "Pearson Correlation": pearson_prob,
            "Spearman Rank": spearman_prob,
            "Kendall Tau": kendall_prob,
            "Cohen’s κ (Linear)": kappa_lin_prob,
            "Cohen’s κ (Quadratic)": kappa_quad_prob,
            "Slope": slope_prob,
            "Intercept": intercept_prob,
            "R²": r2_prob,
            "Bias": bias_prob,
            "MAE": mae_prob,
            "RMSE": rmse_prob,
            "Accuracy |Δ| ≤ 1": acc_pm1_prob,
            "Accuracy |Δ| ≤ 2": acc_pm2_prob,
            "VCR": vcr_prob,
            "RJE": rje_prob,
            "JS": jsd_prob,
        })

    # --- Convert Results to DataFrame ---
    prob_metrics = pd.DataFrame(results)
    print_info(f"Total problems analyzed: {len(prob_metrics)}")

    # ==============================================================
    # === (A) Alignment Metrics ===
    # ==============================================================
    align_cols = [
        "problem_idx", "n",
        "Pearson Correlation", "Spearman Rank", "Kendall Tau",
        "Cohen’s κ (Linear)", "Cohen’s κ (Quadratic)",
        "Slope", "Intercept", "R²"
    ]
    print_table("(A) Alignment Metrics (Problem-wise)", prob_metrics[align_cols].round(3))

    # ==============================================================
    # === (B) Error & Accuracy Metrics ===
    # ==============================================================
    err_cols = [
        "problem_idx", "n",
        "Bias", "MAE", "RMSE",
        "Accuracy |Δ| ≤ 1", "Accuracy |Δ| ≤ 2"
    ]
    print_table("(B) Error & Accuracy Metrics (Problem-wise)", prob_metrics[err_cols].round(3))

    # ==============================================================
    # === (C) Distributional Metrics ===
    # ==============================================================
    dist_cols = [
        "problem_idx", "n",
        "VCR", "RJE", "JS"
    ]
    print_table("(C) Distributional Metrics (Problem-wise)", prob_metrics[dist_cols].round(3))


    # ==============================================================
    # === Visualization Section (by Problem Index) ===
    # ==============================================================
    sns.set(style="whitegrid", font_scale=1.1)

    # --- (1) Pearson Correlation by Problem ---
    plt.figure(figsize=(8, 4))
    # Ensure problem_idx column is numeric to avoid categorical plotting warnings
    prob_metrics_sorted = prob_metrics.sort_values("problem_idx").copy()
    prob_metrics_sorted["problem_idx"] = pd.to_numeric(prob_metrics_sorted["problem_idx"], errors="coerce")
    sns.barplot(
        data=prob_metrics_sorted,
        x="problem_idx",
        y="Pearson Correlation",
        hue="problem_idx",
        palette="Blues_d",
        legend=False
    )
    plt.title("Pearson Correlation by Problem Index")
    plt.xlabel("Problem Index")
    plt.ylabel("Pearson Correlation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_show_fig("Pearson Correlation by Problem Index")

    # --- (2) MAE and Bias by Problem ---
    plt.figure(figsize=(8, 4))
    # Ensure problem_idx column is numeric to avoid categorical plotting warnings
    prob_metrics_sorted = prob_metrics.sort_values("problem_idx").copy()
    prob_metrics_sorted["problem_idx"] = pd.to_numeric(prob_metrics_sorted["problem_idx"], errors="coerce")
    sns.lineplot(
        data=prob_metrics_sorted,
        x="problem_idx",
        y="MAE",
        marker="o",
        label="MAE"
    )
    sns.lineplot(
        data=prob_metrics_sorted,
        x="problem_idx",
        y="Bias",
        marker="s",
        label="Bias"
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Problem-wise MAE and Bias")
    plt.xlabel("Problem Index")
    plt.ylabel("Value")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_show_fig("Problem-wise MAE and Bias")

    # --- (3a) RJE by Problem ---
    plt.figure(figsize=(8, 4))
    # Ensure problem_idx column is numeric to avoid categorical plotting warnings
    prob_metrics_sorted = prob_metrics.sort_values("problem_idx").copy()
    prob_metrics_sorted["problem_idx"] = pd.to_numeric(prob_metrics_sorted["problem_idx"], errors="coerce")
    sns.barplot(
        data=prob_metrics_sorted,
        x="problem_idx",
        y="RJE",
        color="tab:blue"
    )
    plt.title("Relative Judge Entropy (RJE) by Problem Index")
    plt.xlabel("Problem Index")
    plt.ylabel("RJE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_show_fig("RJE by Problem Index")

    # --- (3b) VCR by Problem ---
    plt.figure(figsize=(8, 4))
    # Ensure problem_idx column is numeric to avoid categorical plotting warnings
    prob_metrics_sorted = prob_metrics.sort_values("problem_idx").copy()
    prob_metrics_sorted["problem_idx"] = pd.to_numeric(prob_metrics_sorted["problem_idx"], errors="coerce")
    sns.barplot(
        data=prob_metrics_sorted,
        x="problem_idx",
        y="VCR",
        color="tab:green"
    )
    plt.title("Variance Collapse Ratio (VCR) by Problem Index")
    plt.xlabel("Problem Index")
    plt.ylabel("VCR")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_show_fig("VCR by Problem Index")

    # --- (3c) JS Divergence by Problem ---
    plt.figure(figsize=(8, 4))
    # Ensure problem_idx column is numeric to avoid categorical plotting warnings
    prob_metrics_sorted = prob_metrics.sort_values("problem_idx").copy()
    prob_metrics_sorted["problem_idx"] = pd.to_numeric(prob_metrics_sorted["problem_idx"], errors="coerce")
    sns.barplot(
        data=prob_metrics_sorted,
        x="problem_idx",
        y="JS",
        color="tab:orange"
    )
    plt.title("Jensen-Shannon Divergence by Problem Index")
    plt.xlabel("Problem Index")
    plt.ylabel("Jensen-Shannon Divergence")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_show_fig("JS Divergence by Problem Index")


    # ==============================================================
    # === FINAL VISUAL REPORT: GAUSS Evaluation (Full Version) ===
    # ==============================================================

    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image,
        PageBreak,
    )
    from datetime import datetime

    print_section("GENERATING COMPREHENSIVE VISUAL PDF REPORT (MATCHING PRINT ORDER)")
    summary_dir = os.path.abspath(summary_dir)

    # === Generate timestamped PDF name ===
    date_str = datetime.now().strftime("%Y-%m-%d")
    pdf_filename = f"report_{judge_model_name}_{problem_set_name}_{date_str}.pdf"
    pdf_path = os.path.join(summary_dir, pdf_filename)

    # === Create document ===
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=11))
    styles.add(ParagraphStyle(name="TableTitle", fontSize=10, leading=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="NormalCenter", alignment=1, fontSize=10))
    story = []

    # ==============================================================
    # COVER PAGE
    # ==============================================================
    story.append(Paragraph("<b>GAUSS Evaluation Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Judge Model:</b> {judge_model_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Problem Set:</b> {problem_set_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Samples:</b> {len(df):,}", styles["Normal"]))
    story.append(
        Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"])
    )
    story.append(Spacer(1, 18))

    story.append(
        Paragraph(
            "The <b>GAUSS Evaluation Framework</b> provides a comprehensive statistical assessment of "
            "human–LLM grading consistency in mathematical reasoning tasks. "
            "It integrates three major metric families to capture complementary aspects of alignment and reliability.",
            styles["Small"],
        )
    )

    story.append(
        Paragraph(
            "<b>Alignment metrics</b> (Pearson, Spearman, Kendall τ, linear and quadratic-weighted Cohen’s κ, "
            "slope, intercept, and R²) quantify overall ranking, agreement fidelity, and linear regression consistency.",
            styles["Small"],
        )
    )

    story.append(
        Paragraph(
            "<b>Error & Accuracy metrics</b> (bias, MAE, MSE, and accuracy within ±1 / ±2 score tolerance) "
            "measure the magnitude and direction of score deviations.",
            styles["Small"],
        )
    )

    story.append(
        Paragraph(
            "<b>Distributional metrics</b> (Relative Judge Entropy, Kullback–Leibler divergence, and Variance Collapse Ratio) "
            "capture how well models preserve the diversity and variance structure of human score distributions.",
            styles["Small"],
        )
    )

    story.append(
        Paragraph(
            "<b>Layer I</b> – Zero-Score Discrimination: distinguishes invalid responses (human = 0) "
            "from valid ones (human > 0) to assess validity enforcement and zero-score calibration.",
            styles["Small"],
        )
    )

    story.append(
        Paragraph(
            "<b>Layer II</b> – Band-wise Calibration: examines calibration within low (1–2), mid (3–5), "
            "and high (6–7) score bands to reveal systematic leniency or harshness across scoring ranges.",
            styles["Small"],
        )
    )

    story.append(
        Paragraph(
            "<b>Layer III</b> – Problem-wise Consistency: analyzes alignment, bias, and entropy-based divergence "
            "on a per-problem basis to identify localized deviations between model and human grading behavior.",
            styles["Small"],
        )
    )



    story.append(Spacer(1, 24))
    story.append(PageBreak())


    # ==============================================================
    # SECTION I — Global Overview of Human Scores
    # ==============================================================
    story.append(Paragraph("<b>SECTION I – Global Overview of Human Scores</b>", styles["Heading1"]))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "This section provides a global overview of human-assigned scores across all evaluated models and problems. "
            "It first summarizes the average human scores associated with each model’s graded responses, offering a high-level "
            "view of cross-model scoring tendencies. It then reports per-problem averages computed from the best-performing "
            "model attempts to capture variation in human evaluation across different tasks. Finally, a heatmap visualization "
            "illustrates model–problem interactions, highlighting regions of strong agreement and systematic divergence.",
            styles["Small"],
        )
    )

    story.append(Spacer(1, 10))
    # --------------------------------------------------------------
    # Table 1 + Figure 1: Average Human Score by Model
    # --------------------------------------------------------------
    story.append(Paragraph("<b>Table 1.</b> Average Human Score by Model", styles["Heading2"]))
    data = [model_avg.columns.tolist()] + model_avg.round(3).values.tolist()
    t = Table(data, hAlign="LEFT")
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ])
    )
    story.append(t)
    story.append(Spacer(1, 6))

    insert_figures(story, fig_nums=[1])

    # --------------------------------------------------------------
    # Table 2 + Figure 2: Average Human Score per Problem
    # --------------------------------------------------------------
    story.append(Paragraph("<b>Table 2.</b> Average Human Score per Problem (Best Attempt per Model)", styles["Heading2"]))
    data = [problem_avg.columns.tolist()] + problem_avg.round(3).values.tolist()
    t = Table(data, hAlign="LEFT")
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ])
    )
    story.append(t)
    story.append(Spacer(1, 6))

    insert_figures(story, fig_nums=[2])
    # --------------------------------------------------------------
    # Table 3 + Figure 3: Model × Problem Score Heatmap
    # --------------------------------------------------------------
    story.append(Paragraph("<b>Table 3.</b> Model × Problem Score Heatmap", styles["Heading2"]))
    hm_preview = heatmap_data.round(2).fillna("-").head(10)
    data = [hm_preview.columns.tolist()] + hm_preview.values.tolist()
    t = Table(data, hAlign="LEFT")
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ])
    )
    story.append(t)
    story.append(Spacer(1, 6))

    insert_figures(story, fig_nums=[3])

    story.append(PageBreak())

    # ==============================================================
    # SECTION II — Overall (Global) Analysis
    # ==============================================================

    story.append(Paragraph("<b>SECTION II – Overall (Global) Analysis</b>", styles["Heading1"]))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Evaluates global human–model alignment through multi-perspective statistical measures. "
            "This section reports alignment metrics (Pearson, Spearman, Kendall τ, and Cohen’s κ) capturing "
            "overall correlation and ranking consistency; error and accuracy metrics (bias, MAE, MSE, and "
            "accuracy within ±1 and ±2 score tolerance) quantifying the magnitude of deviation; and "
            "distributional metrics (Relative Judge Entropy, Kullback–Leibler divergence, and Variance Collapse Ratio) "
            "assessing score diversity, calibration, and variance retention. "
            "Together, these measures summarize how closely the model reproduces human grading behavior across the entire score range.",
            styles["Small"],
        )
    )


    story.append(Spacer(1, 8))

    # --------------------------------------------------------------
    # (1) Descriptive Summary (Human vs Model)
    # --------------------------------------------------------------
    if "summary_stats" in locals():
        story.append(Paragraph("<b>Descriptive Summary (Human vs Model)</b>", styles["TableTitle"]))
        table = Table(
            [summary_stats.columns.tolist()] + summary_stats.round(3).astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[70] * len(summary_stats.columns),
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

        # fig_04: Distribution of Model Scores by Human Score
        insert_figures(story, fig_nums=[4])

    # --------------------------------------------------------------
    # (2) Alignment Metrics (Overall)
    # --------------------------------------------------------------
    if "align_df" in locals():
        story.append(Paragraph("<b>Alignment Metrics (Overall)</b>", styles["TableTitle"]))
        table = Table(
            [align_df.columns.tolist()] + align_df.round(3).astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[120, 50],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

        # fig_05: Confusion Matrix
        insert_figures(story, fig_nums=[5])

    # --------------------------------------------------------------
    # (3) Error and Accuracy Metrics (Overall)
    # --------------------------------------------------------------
    if "err_df" in locals():
        story.append(Paragraph("<b>Error and Accuracy Metrics (Overall)</b>", styles["TableTitle"]))
        table = Table(
            [err_df.columns.tolist()] + err_df.round(3).astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[130, 50],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

        # fig_06: Bias Distribution
        insert_figures(story, fig_nums=[6])
        # fig_07: Calibration Curve
        insert_figures(story, fig_nums=[7])
        # fig_08: Mean Error by Human Score
        insert_figures(story, fig_nums=[8])

    # --------------------------------------------------------------
    # (4) Distributional Metrics (Overall)
    # --------------------------------------------------------------
    if "dist_df" in locals():
        story.append(Paragraph("<b>Distributional Metrics (Overall)</b>", styles["TableTitle"]))
        table = Table(
            [dist_df.columns.tolist()] + dist_df.round(4).astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[150, 60],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

    story.append(PageBreak())


    # ==============================================================
    # SECTION III-A — Layer I: Zero-Score Discrimination (Human = 0)
    # ==============================================================

    story.append(Paragraph("<b>SECTION III-A – Layer I: Zero-Score Discrimination (Human = 0)</b>", styles["Heading1"]))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Evaluates the model’s validity enforcement by testing its ability to correctly assign zero scores "
            "when human judges also assigned zero. "
            "This section quantifies zero-score discrimination through alignment and bias metrics, "
            "reporting the false-positive rate and bias distributions that reveal how models over-credit invalid answers.",
            styles["Small"],
        )
    )

    story.append(Spacer(1, 8))

    # --------------------------------------------------------------
    # (1) Summary Statistics (Human=0) + Agreement vs Disagreement
    # --------------------------------------------------------------
    if "summary_h0" in locals():
        story.append(Paragraph("<b>Summary Statistics (Human = 0)</b>", styles["TableTitle"]))
        table = Table(
            [summary_h0.columns.tolist()] + summary_h0.round(3).astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[160, 70],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

        # fig_09: Agreement vs Disagreement
        insert_figures(story, fig_nums=[9])

    # --------------------------------------------------------------
    # (2) Distribution of Non-Zero Predictions (False Positives)
    # --------------------------------------------------------------
    insert_figures(story, fig_nums=[10])


    # ==============================================================
    # === SECTION IV: Layer I-B – Valid-Answer Calibration (Human > 0)
    # ==============================================================
    story.append(Paragraph("<b>SECTION IV – Layer I-B: Valid-Answer Calibration (Human > 0)</b>", styles["Heading1"]))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Analyzes the subset of responses where human-assigned scores are greater than zero, "
            "evaluating model behavior on valid answers only. "
            "This section reports alignment and agreement metrics, bias distributions, and calibration curves, "
            "quantifying how accurately each model reproduces human scoring patterns within the valid-response regime.",
            styles["Small"],
        )
    )

    story.append(Spacer(1, 8))

    # --------------------------------------------------------------
    # (1) Alignment, Regression & Agreement Metrics (Human > 0)
    # --------------------------------------------------------------
    if "pearson_pos" in locals() and "spearman_pos" in locals():
        align_pos_df = pd.DataFrame({
            "Metric": [
                "Pearson Correlation",
                "Spearman Rank",
                "Kendall Tau",
                "Cohen’s κ (Linear)",
                "Cohen’s κ (Quadratic)",
                "Slope",
                "Intercept",
                "R²",
            ],
            "Value": [
                pearson_pos, spearman_pos, kendall_pos,
                kappa_lin_pos, kappa_quad_pos,
                slope_pos, intercept_pos, r2_pos,
            ],
        })
        story.append(Paragraph("<b>Alignment, Regression & Agreement Metrics (Human > 0)</b>", styles["TableTitle"]))
        table = Table(
            [align_pos_df.columns.tolist()] + align_pos_df.round(3).astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[160, 60],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

    # --------------------------------------------------------------
    # (2) Error & Accuracy Metrics (Human > 0)
    # --------------------------------------------------------------
    if "mae_pos" in locals():
        err_pos_df = pd.DataFrame({
            "Metric": [
                "MAE",
                "RMSE",
                "Accuracy",
                "Accuracy |Δ| ≤ 1",
                "Accuracy |Δ| ≤ 2",
            ],
            "Value": [
                mae_pos,
                rmse_pos,
                acc0_pos,
                acc1_pos,
                acc2_pos,
            ],
        })
        story.append(Paragraph("<b>Error & Accuracy Metrics (Human > 0)</b>", styles["TableTitle"]))
        table = Table(
            [err_pos_df.columns.tolist()] + err_pos_df.round(3).astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[160, 60],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))


    # --------------------------------------------------------------
    # (3) Bias Descriptive Statistics + Bias Distribution Plot
    # --------------------------------------------------------------
    if "desc_bias_pos" in locals():
        story.append(Paragraph("<b>Descriptive Statistics: Bias (Model - Human) | Human > 0</b>", styles["TableTitle"]))
        table = Table(
            [desc_bias_pos.columns.tolist()] + desc_bias_pos.round(3).astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[160, 70],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

        # fig_11: Bias Distribution (Human>0)
        insert_figures(story, fig_nums=[11])

    # --------------------------------------------------------------
    # (4) Calibration by Human Score + Calibration Curve
    # --------------------------------------------------------------
    insert_figures(story, fig_nums=[12])

    # --------------------------------------------------------------
    # (5) Conditional Bias by Human Score + Mean Bias Barplot
    # --------------------------------------------------------------
    if "cond_bias_pos" in locals():
        story.append(Paragraph("<b>Conditional Bias by Human Score</b>", styles["TableTitle"]))
        table = Table(
            [cond_bias_pos.columns.tolist()] + cond_bias_pos.round(3).astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[140, 60, 60, 60],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

        # fig_13: Mean Bias by Human Score
        insert_figures(story, fig_nums=[13])

    story.append(PageBreak())


    # ==============================================================
    # SECTION V — Layer II: Band-wise Calibration Analysis
    # ==============================================================

    story.append(Paragraph("<b>SECTION V – Layer II: Band-wise Calibration Analysis</b>", styles["Heading1"]))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Examines calibration within low (1–2), mid (3–5), and high (6–7) score bands, "
            "evaluating how models allocate partial credit across valid answers. "
            "Each band is analyzed using alignment and bias measures, along with distributional metrics "
            "including Relative Judge Entropy (RJE), Kullback–Leibler divergence (KL), and Variance Collapse Ratio (VCR), "
            "which quantify the preservation of spread and human-like variance within each scoring regime.",
            styles["Small"],
        )
    )

    story.append(Spacer(1, 8))
    # --------------------------------------------------------------
    # (1) Alignment Metrics (Band-wise)
    # --------------------------------------------------------------
    if "band_df" in locals():
        story.append(Paragraph("<b>Alignment Metrics (Band-wise)</b>", styles["Heading2"]))
        table = Table(
            [band_df[[
                "Band", "Count", "Slope", "Intercept", "R²",
                "Pearson", "Spearman", "Kendall",
                "κ (Linear)", "κ (Quadratic)"
            ]].columns.tolist()]
            + band_df[[
                "Band", "Count", "Slope", "Intercept", "R²",
                "Pearson", "Spearman", "Kendall",
                "κ (Linear)", "κ (Quadratic)"
            ]].round(3).astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[60, 45, 50, 50, 45, 55, 55, 55, 60, 60],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.0),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

    # --------------------------------------------------------------
    # (2a) Error Metrics (Band-wise)
    # --------------------------------------------------------------
    if "band_df" in locals():
        story.append(Paragraph("<b>Error Metrics (Band-wise)</b>", styles["Heading2"]))
        err_cols = ["Band", "Count", "MAE", "RMSE", "Acc |Δ|≤1", "Acc |Δ|≤2"]
        table = Table(
            [band_df[err_cols].columns.tolist()] +
            band_df[err_cols].round(3).astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[60, 50, 60, 60, 70, 70],
        )
        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(table)
        story.append(Spacer(1, 8))
    # --------------------------------------------------------------
    # (2b) Bias Metrics (Band-wise)
    # --------------------------------------------------------------
    if "band_df" in locals():
        story.append(Paragraph("<b>Bias Metrics (Band-wise)</b>", styles["Heading2"]))
        bias_cols = [
            "Band", "Mean Bias", "Median Bias",
            "Std Bias", "Min Bias", "Max Bias"
        ]
        table = Table(
            [band_df[bias_cols].columns.tolist()] +
            band_df[bias_cols].round(3).astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[60, 65, 65, 65, 65, 65],
        )
        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(table)
        story.append(Spacer(1, 8))



    # --------------------------------------------------------------
    # (3) Distributional Metrics (Band-wise)
    # --------------------------------------------------------------
    if "band_df" in locals():
        story.append(Paragraph("<b>Distributional Metrics (Band-wise)</b>", styles["TableTitle"]))
        table = Table(
            [band_df[[
                "Band", "Count", "Var(X)", "Var(Y)", "RJE", "VCR"
            ]].columns.tolist()]
            + band_df[[
                "Band", "Count", "Var(X)", "Var(Y)", "RJE", "VCR"
            ]].round(3).astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[90, 45, 60, 60, 60, 60],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.0),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

    # --------------------------------------------------------------
    # (4) Band-wise Calibration Curves
    # --------------------------------------------------------------
    insert_figures(story, fig_nums=[14])

    # --------------------------------------------------------------
    # (5) Mean Bias by Band
    # --------------------------------------------------------------
    insert_figures(story, fig_nums=[15])

    story.append(PageBreak())


    # ==============================================================
    # LAYER III — Problem-wise Consistency Analysis
    # ==============================================================

    story.append(Paragraph("<b>LAYER III – Problem-wise Consistency Analysis</b>", styles["Heading1"]))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Aggregates all previous layers at the per-problem level to evaluate how model–human alignment "
            "varies across different mathematical tasks. "
            "For each problem, correlation coefficients, bias patterns, and calibration metrics are computed "
            "to identify whether certain problems systematically amplify or suppress model–human agreement, "
            "revealing localized sources of divergence in scoring behavior.",
            styles["Small"],
        )
    )

    story.append(Spacer(1, 8))

    # --------------------------------------------------------------
    # (A1) Correlation & Agreement Metrics (Problem-wise)
    # --------------------------------------------------------------
    if "prob_metrics" in locals():
        story.append(Paragraph("<b>(A1) Correlation & Agreement Metrics (Problem-wise)</b>", styles["TableTitle"]))
        align_corr_cols = [
            "problem_idx", "n",
            "Pearson Correlation", "Spearman Rank", "Kendall Tau",
            "Cohen’s κ (Linear)", "Cohen’s κ (Quadratic)"
        ]
        table = Table(
            [align_corr_cols] +
            prob_metrics[align_corr_cols].round(3).astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[65, 40, 70, 70, 65, 70, 70],
        )
        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8.0),
            ])
        )
        story.append(table)
        story.append(Spacer(1, 8))

    # --------------------------------------------------------------
    # (A2) Regression Metrics (Problem-wise)
    # --------------------------------------------------------------
    if "prob_metrics" in locals():
        story.append(Paragraph("<b>(A2) Regression Metrics (Problem-wise)</b>", styles["TableTitle"]))
        align_reg_cols = [
            "problem_idx", "n", "Slope", "Intercept", "R²"
        ]
        table = Table(
            [align_reg_cols] +
            prob_metrics[align_reg_cols].round(3).astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[65, 40, 65, 65, 65],
        )
        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8.0),
            ])
        )
        story.append(table)
        story.append(Spacer(1, 8))

    # --- Figure 16: Pearson Correlation by Problem Index ---
    insert_figures(story, fig_nums=[16])


    # --------------------------------------------------------------
    # (B) Error & Accuracy Metrics (Problem-wise)
    # --------------------------------------------------------------
    if "prob_metrics" in locals():
        story.append(Paragraph("<b>(B) Error & Accuracy Metrics (Problem-wise)</b>", styles["TableTitle"]))
        err_cols = [
            "problem_idx", "n",
            "Bias", "MAE", "RMSE",
            "Accuracy |Δ| ≤ 1", "Accuracy |Δ| ≤ 2"
        ]
        table = Table(
            [err_cols] + prob_metrics[err_cols].round(3).astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[70, 40, 55, 55, 55, 70, 70],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.0),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

        # fig_2: Problem-wise MAE and Bias
        insert_figures(story, fig_nums=[17])

    # --------------------------------------------------------------
    # (C) Distributional Metrics (Problem-wise)
    # --------------------------------------------------------------
    if "prob_metrics" in locals():
        story.append(Paragraph("<b>(C) Distributional Metrics (Problem-wise)</b>", styles["TableTitle"]))
        dist_cols = ["problem_idx", "n", "VCR", "RJE", "JS"]
        table = Table(
            [dist_cols] + prob_metrics[dist_cols].round(3).astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[70, 40, 60, 60, 60],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.0),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

        # fig_3a: RJE by Problem Index
        insert_figures(story, fig_nums=[18])
        # fig_3b: VCR by Problem Index
        insert_figures(story, fig_nums=[19])
        # fig_3c: JS Divergence by Problem Index
        insert_figures(story, fig_nums=[20])

    story.append(PageBreak())


    # ==============================================================
    # BUILD PDF
    # ==============================================================
    doc.build(story)
    print_info(f"📄 Visual report generated: {pdf_path}")