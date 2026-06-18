"""Shared data loading and feature engineering for SciSci modules."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_engineer(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply feature engineering to a raw DataFrame based on config.

    Args:
        df: Raw DataFrame (from dataset_a.csv or synthetic fixture).
        cfg: Config dict matching scisci_config.yaml structure.

    Returns:
        DataFrame with engineered features appended.
    """
    current_year = cfg["project"].get("current_year", 2026)
    df = df.copy()
    df["citations"] = df["citations"].fillna(0).astype(int)
    df["log_citations"] = np.log1p(df["citations"])

    # Build theme indicators from keyword+abstract text matching
    combined = (
        df["keywords"].str.lower().fillna("")
        + " "
        + df["abstract"].str.lower().fillna("")
    )

    for key in ("theme_a", "theme_b"):
        v = cfg["variables"][key]
        col_name = v["name"]
        terms = v["terms"]
        df[col_name] = combined.apply(lambda x: int(any(t in x for t in terms)))

    ta = cfg["variables"]["theme_a"]["name"]
    tb = cfg["variables"]["theme_b"]["name"]
    df["interaction"] = df[ta] * df[tb]

    # Control variables
    for ctrl in cfg["variables"].get("controls", []):
        col = ctrl["col"]
        compute = ctrl.get("compute", "")
        if col == "is_international" and compute == "affiliations":
            df[col] = df["affiliations"].apply(
                lambda a: int(
                    len(set(c.split(",")[-1].strip() for c in str(a).split(";"))) > 1
                )
                if pd.notna(a)
                else 0
            )
        elif col == "author_count" and compute == "authors":
            df[col] = df["authors"].apply(
                lambda x: len(str(x).split(";")) if pd.notna(x) else 1
            )
        elif col == "norm_degree_z" and compute == "degree":
            df[col] = StandardScaler().fit_transform(df[["degree"]].fillna(0))
        elif col == "paper_age" and compute == "year":
            df[col] = current_year - df["year"]

    # Author prestige proxy — controls for senior-author selection bias
    # Metric: for each paper, max(author's total papers in this corpus)
    if "authors" in df.columns:
        author_paper_count = {}
        for _, row in df.iterrows():
            if pd.notna(row.get("authors")):
                for author in str(row["authors"]).split(";"):
                    name = author.strip().lower()
                    if name:
                        author_paper_count[name] = author_paper_count.get(name, 0) + 1

        def _max_author_prestige(authors_str):
            if pd.isna(authors_str):
                return 1
            names = [a.strip().lower() for a in str(authors_str).split(";") if a.strip()]
            counts = [author_paper_count.get(n, 1) for n in names]
            return max(counts) if counts else 1

        df["author_prestige"] = df["authors"].apply(_max_author_prestige)
    else:
        df["author_prestige"] = 1

    df["is_top_10"] = (df["citations"] >= df["citations"].quantile(0.90)).astype(int)

    # Document type control: Review papers structurally accumulate higher citations
    # (endogeneity fix — isolates boundary-spanning effect from review-paper premium)
    if "entry_type" in df.columns:
        df["is_review"] = df["entry_type"].isin(["Review", "Short Survey"]).astype(int)
    else:
        df["is_review"] = 0

    # Cluster name mapping
    if "cluster_id" in df.columns and "cluster_label" in df.columns:
        df["_cname"] = df["cluster_id"].map(
            {
                r["cluster_id"]: str(r["cluster_label"]).strip()
                for _, r in df[["cluster_id", "cluster_label"]]
                .drop_duplicates()
                .dropna()
                .iterrows()
            }
        )

    return df


def get_labels(cfg: dict) -> dict:
    """Build variable -> label mapping from config."""
    labels = {
        cfg["variables"]["theme_a"]["name"]: cfg["variables"]["theme_a"]["label"],
        cfg["variables"]["theme_b"]["name"]: cfg["variables"]["theme_b"]["label"],
        "interaction": f"{cfg['variables']['theme_a']['label']} × {cfg['variables']['theme_b']['label']}",
    }
    for ctrl in cfg["variables"].get("controls", []):
        labels[ctrl["col"]] = ctrl["label"]
    return labels


def get_model_vars(cfg: dict) -> list:
    """Get the list of model independent variable column names."""
    ta = cfg["variables"]["theme_a"]["name"]
    tb = cfg["variables"]["theme_b"]["name"]
    ctrl_cols = [c["col"] for c in cfg["variables"].get("controls", [])]
    return [ta, tb] + ctrl_cols


def get_formula(cfg: dict, dv: str = "log_citations") -> str:
    """Build a statsmodels formula string."""
    ta = cfg["variables"]["theme_a"]["name"]
    tb = cfg["variables"]["theme_b"]["name"]
    ctrl_cols = [c["col"] for c in cfg["variables"].get("controls", [])]
    ivs = [ta, tb, "interaction"] + ctrl_cols
    return f"{dv} ~ " + " + ".join(ivs)


def check_terminal_year(df: pd.DataFrame, threshold: float = 0.75) -> bool:
    """Check if the terminal year in the dataset appears truncated.

    A year is considered truncated if its paper count is less than
    `threshold` (default 75%) of the penultimate year's count. This
    is a standard bibliometric hygiene check for partial-year Scopus
    data (Golden Rule #9, Lessons-Learned Cookbook).

    Args:
        df: DataFrame with a 'year' column.
        threshold: Fraction below which the terminal year is flagged.

    Returns:
        True if the terminal year appears truncated, False otherwise.
    """
    if "year" not in df.columns:
        return False

    year_counts = df["year"].value_counts().sort_index()
    if len(year_counts) < 2:
        return False

    terminal_year = year_counts.index[-1]
    penultimate_year = year_counts.index[-2]
    terminal_count = year_counts.iloc[-1]
    penultimate_count = year_counts.iloc[-2]

    is_truncated = bool(terminal_count < (threshold * penultimate_count))

    if is_truncated:
        print(
            f"  ⚠️  TERMINAL YEAR TRUNCATION DETECTED: "
            f"{terminal_year} has {terminal_count} papers "
            f"({terminal_count / penultimate_count:.0%} of "
            f"{penultimate_year}'s {penultimate_count}). "
            f"Consider excluding {terminal_year} from temporal models."
        )
    else:
        print(
            f"  ✅ Terminal year check: {terminal_year} has {terminal_count} "
            f"papers ({terminal_count / penultimate_count:.0%} of "
            f"{penultimate_year}'s {penultimate_count}) — OK."
        )

    return is_truncated
