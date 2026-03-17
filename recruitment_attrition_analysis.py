from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PANDAS_NO_IMPORT_PYARROW", "1")

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


EXPECTED_COLUMNS = [
    "Candidate Name",
    "Offer Date",
    "Reporting Date",
    "Degree",
    "School Name",
    "Major",
    "Hiring Team",
    "Hiring Position",
    "Hiring Location",
    "Final Result",
    "Reject Reason",
]

RENAME_MAP = {
    "Candidate Name": "candidate_id",
    "Offer Date": "offer_date",
    "Reporting Date": "reporting_date",
    "Degree": "degree",
    "School Name": "school_name",
    "Major": "major",
    "Hiring Team": "hiring_team",
    "Hiring Position": "hiring_position",
    "Hiring Location": "hiring_location",
    "Final Result": "final_result",
    "Reject Reason": "reject_reason",
}

RESULT_MAP = {
    "accept": "Accept",
    "reject": "Reject",
    "withdrawn": "withdrawn",
}

LEAD_BUCKETS = [-np.inf, 30, 60, 90, 120, 180, np.inf]
LEAD_BUCKET_LABELS = ["<30d", "30-59d", "60-89d", "90-119d", "120-179d", "180d+"]


sns.set_theme(style="whitegrid", context="notebook")


def configure_chinese_font() -> None:
    # Try common CJK fonts to prevent garbled Chinese labels in charts.
    font_candidates = [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "PingFang TC",
        "Heiti TC",
        "Noto Sans CJK TC",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "SimHei",
    ]
    plt.rcParams["font.sans-serif"] = font_candidates + plt.rcParams.get("font.sans-serif", [])
    plt.rcParams["axes.unicode_minus"] = False


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def parse_tw_datetime(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.replace("上午", "AM", regex=False)
        .str.replace("下午", "PM", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )
    parsed = pd.to_datetime(
        normalized,
        format="%Y/%m/%d %p %I:%M:%S",
        errors="coerce",
    )
    if parsed.isna().any():
        fallback = pd.to_datetime(normalized, errors="coerce")
        parsed = parsed.fillna(fallback)
    return parsed


def load_raw_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    data = df.rename(columns=RENAME_MAP).copy()

    text_cols = [
        "candidate_id",
        "degree",
        "school_name",
        "major",
        "hiring_team",
        "hiring_position",
        "hiring_location",
        "final_result",
        "reject_reason",
    ]
    for col in text_cols:
        data[col] = clean_text(data[col])

    data["offer_date"] = parse_tw_datetime(data["offer_date"])
    data["reporting_date"] = parse_tw_datetime(data["reporting_date"])

    data["final_result"] = (
        data["final_result"].str.lower().map(RESULT_MAP).fillna(data["final_result"])
    )
    data["is_attrition"] = data["final_result"].isin(["Reject", "withdrawn"]).astype(int)

    data["lead_days"] = (data["reporting_date"] - data["offer_date"]).dt.days
    data["lead_days_bucket"] = pd.cut(
        data["lead_days"], bins=LEAD_BUCKETS, labels=LEAD_BUCKET_LABELS, right=False
    )

    data["offer_month"] = data["offer_date"].dt.to_period("M").astype("string")
    data["offer_quarter"] = data["offer_date"].dt.to_period("Q").astype("string")
    data["major_clean"] = clean_text(data["major"])
    data["school_code"] = clean_text(data["school_name"])

    data = data.sort_values(["candidate_id", "offer_date", "reporting_date"]).reset_index(drop=True)
    return data


def build_candidate_view(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_cand = (
        df_raw.groupby("candidate_id", as_index=False, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return df_cand


def summarize_outcome(df: pd.DataFrame, grain_name: str) -> dict[str, Any]:
    total = len(df)
    attrition = int(df["is_attrition"].sum())
    accept = total - attrition
    return {
        "grain": grain_name,
        "rows": total,
        "accept": accept,
        "attrition": attrition,
        "accept_rate": accept / total if total else np.nan,
        "attrition_rate": attrition / total if total else np.nan,
        "offer_start": df["offer_date"].min(),
        "offer_end": df["offer_date"].max(),
    }


def build_quality_report(df_raw: pd.DataFrame, df_cand: pd.DataFrame) -> pd.DataFrame:
    quality = pd.DataFrame(
        {
            "missing_count": df_raw.isna().sum(),
            "missing_ratio": (df_raw.isna().mean() * 100).round(2),
            "nunique": df_raw.nunique(dropna=True),
        }
    ).sort_values("missing_count", ascending=False)

    neg_lead_rows = int((df_raw["lead_days"] < 0).fillna(False).sum())
    if neg_lead_rows != 0:
        raise ValueError(f"Found lead_days < 0 rows: {neg_lead_rows}")

    if df_cand["candidate_id"].nunique() != len(df_cand):
        raise ValueError("candidate_id is not unique in candidate-level view")

    return quality


def attrition_by_dimension(df: pd.DataFrame, col: str, min_n: int) -> pd.DataFrame:
    baseline = df["is_attrition"].mean()
    out = (
        df.groupby(col, dropna=False, observed=False)
        .agg(n=("candidate_id", "count"), attrition_rate=("is_attrition", "mean"))
        .reset_index()
    )
    out = out[out["n"] >= min_n].copy()
    out["risk_lift_pp"] = (out["attrition_rate"] - baseline) * 100
    out["excess_attritions"] = (out["attrition_rate"] - baseline) * out["n"]
    return out.sort_values(["risk_lift_pp", "n"], ascending=[False, False])


def interaction_risk(df: pd.DataFrame, cols: list[str], min_n: int) -> pd.DataFrame:
    baseline = df["is_attrition"].mean()
    out = (
        df.groupby(cols, dropna=False, observed=False)
        .agg(n=("candidate_id", "count"), attrition_rate=("is_attrition", "mean"))
        .reset_index()
    )
    out = out[out["n"] >= min_n].copy()
    if out.empty:
        return out
    out["risk_lift_pp"] = (out["attrition_rate"] - baseline) * 100
    out["segment"] = out[cols].astype(str).apply(lambda r: " | ".join(r.values), axis=1)
    return out.sort_values("risk_lift_pp", ascending=False)


def build_risk_segments(df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    baseline = df["is_attrition"].mean()
    dims = [
        ["hiring_team"],
        ["hiring_position"],
        ["hiring_location"],
        ["degree"],
        ["lead_days_bucket"],
        ["hiring_team", "hiring_position"],
        ["hiring_team", "hiring_location"],
        ["hiring_position", "lead_days_bucket"],
    ]

    tables = []
    for cols in dims:
        part = (
            df.groupby(cols, dropna=False, observed=False)
            .agg(n=("candidate_id", "count"), attrition_rate=("is_attrition", "mean"))
            .reset_index()
        )
        part = part[part["n"] >= min_n].copy()
        if part.empty:
            continue

        part["baseline"] = baseline
        part["risk_lift_pp"] = (part["attrition_rate"] - baseline) * 100
        part["excess_attritions"] = (part["attrition_rate"] - baseline) * part["n"]
        part["dimension"] = " + ".join(cols)
        part["segment"] = part[cols].astype(str).apply(lambda r: " | ".join(r.values), axis=1)
        tables.append(part)

    if not tables:
        return pd.DataFrame(columns=["dimension", "segment", "n", "attrition_rate", "risk_lift_pp", "excess_attritions"])

    out = pd.concat(tables, ignore_index=True)
    return out.sort_values(["risk_lift_pp", "n"], ascending=[False, False])

def plot_result_distribution(df_cand: pd.DataFrame) -> None:
    result_dist = (
        df_cand["final_result"]
        .value_counts(dropna=False)
        .reindex(["Accept", "Reject", "withdrawn"])
        .fillna(0)
        .astype(int)
        .rename_axis("final_result")
        .reset_index(name="count")
    )
    result_dist["pct"] = result_dist["count"] / result_dist["count"].sum()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=result_dist, x="final_result", y="count", palette="Set2", ax=ax)
    ax.set_title("Candidate-level Final Result Distribution")
    ax.set_xlabel("Final Result")
    ax.set_ylabel("Candidates")
    for i, row in result_dist.iterrows():
        ax.text(i, row["count"] + 2, f"{row['pct']:.1%}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.show()


def monthly_profile(df_cand: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df_cand.dropna(subset=["offer_date"])
        .assign(offer_month_dt=lambda d: d["offer_date"].values.astype("datetime64[M]"))
        .groupby("offer_month_dt", as_index=False)
        .agg(offers=("candidate_id", "count"), attrition_rate=("is_attrition", "mean"))
        .sort_values("offer_month_dt")
    )
    return monthly


def plot_monthly_profile(monthly: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()

    ax1.plot(monthly["offer_month_dt"], monthly["offers"], marker="o", color="#1f77b4", label="Offers")
    ax2.plot(monthly["offer_month_dt"], monthly["attrition_rate"], marker="s", color="#d62728", label="Attrition Rate")

    ax1.set_title("Monthly Offer Volume and Attrition Rate")
    ax1.set_xlabel("Offer Month")
    ax1.set_ylabel("Offer Count", color="#1f77b4")
    ax2.set_ylabel("Attrition Rate", color="#d62728")
    ax2.set_ylim(0, max(0.35, monthly["attrition_rate"].max() * 1.2))

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.tight_layout()
    plt.show()


def lead_days_profile(df_cand: pd.DataFrame) -> pd.DataFrame:
    lead_df = df_cand.dropna(subset=["lead_days"]).copy()
    lead_bucket = (
        lead_df.groupby("lead_days_bucket", dropna=False, observed=False)
        .agg(n=("candidate_id", "count"), attrition_rate=("is_attrition", "mean"))
        .reset_index()
        .sort_values("lead_days_bucket")
    )
    baseline_attr = lead_df["is_attrition"].mean()
    lead_bucket["risk_lift_pp"] = (lead_bucket["attrition_rate"] - baseline_attr) * 100
    return lead_df, lead_bucket


def plot_lead_days(lead_df: pd.DataFrame, lead_bucket: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(data=lead_df, x="lead_days", hue="final_result", bins=30, kde=True, ax=axes[0])
    axes[0].set_title("Lead Days Distribution by Final Result")
    axes[0].set_xlabel("Lead Days")

    sns.boxplot(data=lead_df, x="is_attrition", y="lead_days", ax=axes[1], palette="Set3")
    axes[1].set_xticklabels(["Accept", "Attrition"])
    axes[1].set_title("Lead Days vs Attrition")
    axes[1].set_xlabel("Outcome Group")

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 4))
    baseline = lead_df["is_attrition"].mean()
    sns.barplot(data=lead_bucket, x="lead_days_bucket", y="attrition_rate", palette="flare", ax=ax)
    ax.axhline(baseline, color="black", linestyle="--", linewidth=1, label=f"Baseline={baseline:.1%}")
    ax.set_title("Attrition Rate by Lead Days Bucket")
    ax.set_xlabel("Lead Days Bucket")
    ax.set_ylabel("Attrition Rate")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_dimension_bars(risk_team: pd.DataFrame, risk_position: pd.DataFrame, risk_location: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    sns.barplot(data=risk_team.sort_values("attrition_rate", ascending=False), x="hiring_team", y="attrition_rate", ax=axes[0], palette="Blues_d")
    sns.barplot(data=risk_position.sort_values("attrition_rate", ascending=False), x="hiring_position", y="attrition_rate", ax=axes[1], palette="Greens_d")
    sns.barplot(data=risk_location.sort_values("attrition_rate", ascending=False), x="hiring_location", y="attrition_rate", ax=axes[2], palette="Oranges_d")

    axes[0].set_title("Attrition by Team")
    axes[1].set_title("Attrition by Position")
    axes[2].set_title("Attrition by Location")

    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("Attrition Rate")
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.show()


def plot_interaction_heatmaps(team_pos: pd.DataFrame, team_loc: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    p1 = team_pos.pivot(index="hiring_team", columns="hiring_position", values="attrition_rate")
    sns.heatmap(p1, annot=True, fmt=".2%", cmap="YlOrRd", ax=axes[0])
    axes[0].set_title("Team × Position Attrition")

    p2 = team_loc.pivot(index="hiring_team", columns="hiring_location", values="attrition_rate")
    sns.heatmap(p2, annot=True, fmt=".2%", cmap="YlOrRd", ax=axes[1])
    axes[1].set_title("Team × Location Attrition")

    plt.tight_layout()
    plt.show()


def reject_reason_profile(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    non_accept = df_raw[df_raw["is_attrition"] == 1].copy()
    non_accept["reject_reason_clean"] = non_accept["reject_reason"].replace("", "<NULL>")

    reason_counts = (
        non_accept["reject_reason_clean"]
        .value_counts()
        .rename_axis("reject_reason")
        .reset_index(name="count")
    )
    reason_counts["cum_pct"] = reason_counts["count"].cumsum() / reason_counts["count"].sum()

    reason_mix = (
        non_accept.groupby(["hiring_team", "reject_reason_clean"])
        .size()
        .rename("count")
        .reset_index()
    )
    reason_mix["share_in_team"] = reason_mix["count"] / reason_mix.groupby("hiring_team")["count"].transform("sum")
    reason_mix = reason_mix.sort_values(["hiring_team", "count"], ascending=[True, False])

    return reason_counts, reason_mix


def plot_reject_reason_pareto(reason_counts: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=reason_counts, x="reject_reason", y="count", ax=ax1, color="#4c78a8")
    ax1.set_title("Reject Reason Pareto")
    ax1.set_xlabel("Reject Reason")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=20)

    ax2 = ax1.twinx()
    ax2.plot(reason_counts["reject_reason"], reason_counts["cum_pct"], color="#f58518", marker="o")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Cumulative %")
    plt.tight_layout()
    plt.show()


def cramers_v(conf_mat: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(conf_mat)[0]
    n = conf_mat.to_numpy().sum()
    if n == 0:
        return np.nan
    r, k = conf_mat.shape
    denom = max(1, min(r - 1, k - 1))
    return np.sqrt((chi2 / n) / denom)


def run_stat_tests(df_cand: pd.DataFrame, min_n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    categorical_cols = [
        "hiring_team",
        "hiring_position",
        "hiring_location",
        "degree",
        "lead_days_bucket",
    ]

    chi_rows = []
    for col in categorical_cols:
        temp = df_cand[[col, "is_attrition"]].dropna()
        ctab = pd.crosstab(temp[col], temp["is_attrition"])
        if ctab.shape[0] < 2 or ctab.shape[1] < 2:
            continue

        chi2, pval, dof, _ = stats.chi2_contingency(ctab)
        chi_rows.append(
            {
                "feature": col,
                "chi2": chi2,
                "dof": dof,
                "p_value": pval,
                "cramers_v": cramers_v(ctab),
            }
        )

    chi_df = pd.DataFrame(chi_rows).sort_values("cramers_v", ascending=False)

    lead_non_attr = df_cand.loc[df_cand["is_attrition"] == 0, "lead_days"].dropna()
    lead_attr = df_cand.loc[df_cand["is_attrition"] == 1, "lead_days"].dropna()
    mw_u, mw_p = stats.mannwhitneyu(lead_non_attr, lead_attr, alternative="two-sided")
    mw_df = pd.DataFrame(
        [
            {
                "test": "Mann-Whitney U (lead_days)",
                "u_stat": mw_u,
                "p_value": mw_p,
                "accept_median": float(np.median(lead_non_attr)),
                "attrition_median": float(np.median(lead_attr)),
            }
        ]
    )

    def two_prop_test(df: pd.DataFrame, segment_col: str, a: str, b: str) -> dict[str, Any]:
        sub = df[df[segment_col].isin([a, b])].copy()
        g = sub.groupby(segment_col)["is_attrition"].agg(["sum", "count"])
        if a not in g.index or b not in g.index:
            return {"comparison": f"{segment_col}: {a} vs {b}", "error": "group not found"}

        x = np.array([g.loc[a, "sum"], g.loc[b, "sum"]], dtype=float)
        n = np.array([g.loc[a, "count"], g.loc[b, "count"]], dtype=float)
        if np.any(n < min_n):
            return {"comparison": f"{segment_col}: {a} vs {b}", "error": f"n < {min_n}"}

        z_stat, p_val = proportions_ztest(x, n)
        rate_a = x[0] / n[0]
        rate_b = x[1] / n[1]
        return {
            "comparison": f"{segment_col}: {a} vs {b}",
            "n_a": int(n[0]),
            "n_b": int(n[1]),
            "attr_rate_a": rate_a,
            "attr_rate_b": rate_b,
            "rate_diff_pp": (rate_a - rate_b) * 100,
            "z_stat": z_stat,
            "p_value": p_val,
        }

    ztest_rows = []
    ztest_rows.append(two_prop_test(df_cand, "hiring_position", "Junior IT Engineer", "Senior IT Engineer"))
    ztest_rows.append(
        two_prop_test(
            df_cand.assign(loc_grp=np.where(df_cand["hiring_location"] == "HSN", "HSN", "Non-HSN")),
            "loc_grp",
            "HSN",
            "Non-HSN",
        )
    )
    ztest_rows.append(
        two_prop_test(
            df_cand.assign(lead_grp=np.where(df_cand["lead_days"] >= 180, "180d+", "<180d")),
            "lead_grp",
            "180d+",
            "<180d",
        )
    )

    ztest_df = pd.DataFrame(ztest_rows)
    return chi_df, mw_df, ztest_df


def plot_risk_segments(risk_segments: pd.DataFrame) -> None:
    top_plot = risk_segments.head(20).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=top_plot,
        x="n",
        y="risk_lift_pp",
        size="excess_attritions",
        hue="dimension",
        sizes=(60, 600),
        ax=ax,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Top Risk Segments: Risk Lift vs Sample Size")
    ax.set_xlabel("Sample Size (n)")
    ax.set_ylabel("Risk Lift (pp vs baseline)")

    for _, row in top_plot.head(10).iterrows():
        ax.text(row["n"] + 1, row["risk_lift_pp"] + 0.2, row["segment"], fontsize=8)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.show()


def build_insights(
    df_cand: pd.DataFrame,
    reason_counts: pd.DataFrame,
    risk_segments: pd.DataFrame,
    monthly: pd.DataFrame,
    lead_bucket: pd.DataFrame,
    risk_position: pd.DataFrame,
) -> pd.DataFrame:
    baseline = df_cand["is_attrition"].mean()
    top_reason = reason_counts.iloc[0]
    top_seg = risk_segments.iloc[0]
    worst_month = monthly.sort_values("attrition_rate", ascending=False).iloc[0]
    best_month = monthly.sort_values("attrition_rate", ascending=True).iloc[0]

    long_wait = lead_bucket.loc[lead_bucket["lead_days_bucket"] == "180d+"]
    long_wait = long_wait.iloc[0] if len(long_wait) else None

    junior = risk_position.loc[risk_position["hiring_position"] == "Junior IT Engineer"]

    insights = [
        {
            "priority": 1,
            "insight": "先鎖定最高風險區段做定向挽留",
            "evidence": f"segment={top_seg['segment']}，risk_lift={top_seg['risk_lift_pp']:.2f}pp",
            "action": "offer 後第7/14/30天固定關懷節點，要求 hiring team 回報候選人狀態。",
            "kpi": "該區段 attrition_rate 下降 >= 3pp",
        },
        {
            "priority": 2,
            "insight": "縮短等待期，優先處理長等待候選人",
            "evidence": (
                f"180d+ attrition={long_wait['attrition_rate']:.1%}，vs baseline {baseline:.1%}"
                if long_wait is not None
                else "未找到 180d+ bucket"
            ),
            "action": "建立長等待候選人預警清單與流程加速機制。",
            "kpi": "180d+ 族群 attrition_rate 下降 >= 4pp",
        },
        {
            "priority": 3,
            "insight": "聚焦 Top 拒絕原因做制度改善",
            "evidence": f"top reason={top_reason['reject_reason']}，占比={top_reason['count'] / reason_counts['count'].sum():.1%}",
            "action": "提前對齊職務內容、薪資競爭力與候選人期待。",
            "kpi": "Top1 拒絕原因占比下降 >= 5pp",
        },
        {
            "priority": 4,
            "insight": "在歷史高風險月份前置部署",
            "evidence": f"best={best_month['offer_month_dt']:%Y-%m}({best_month['attrition_rate']:.1%}), worst={worst_month['offer_month_dt']:%Y-%m}({worst_month['attrition_rate']:.1%})",
            "action": "高風險月份前一月啟動候選人關係維護與競業條件盤點。",
            "kpi": "高風險月份 attrition_rate 年比下降 >= 3pp",
        },
        {
            "priority": 5,
            "insight": "主力職缺建立標準化挽留腳本",
            "evidence": f"Junior attrition={float(junior['attrition_rate'].iloc[0]):.1%}" if len(junior) else "Junior 樣本不足",
            "action": "針對主力職缺強化職務邊界、成長路徑與報到節奏溝通。",
            "kpi": "主力職缺 attrition_rate 下降 >= 2pp",
        },
    ]

    return pd.DataFrame(insights).sort_values("priority")


def run_acceptance_checks(
    df_raw: pd.DataFrame,
    df_cand: pd.DataFrame,
    min_n: int,
    risk_tables: list[pd.DataFrame],
    chi_df: pd.DataFrame,
    mw_df: pd.DataFrame,
    ztest_df: pd.DataFrame,
    insight_df: pd.DataFrame,
) -> pd.DataFrame:
    tests = []

    tests.append(("11 columns loaded", len(EXPECTED_COLUMNS) == 11 and set(RENAME_MAP.values()).issubset(df_raw.columns)))
    tests.append(("row-level vs candidate-level differs", len(df_raw) != len(df_cand)))
    tests.append(("candidate-level unique id", df_cand["candidate_id"].nunique() == len(df_cand)))
    tests.append(("is_attrition logic valid", set(df_cand["is_attrition"].dropna().unique()).issubset({0, 1})))
    tests.append(("lead_days >= 0", int((df_raw["lead_days"] < 0).fillna(False).sum()) == 0))

    threshold_ok = True
    for tbl in risk_tables:
        if len(tbl) > 0 and (tbl["n"] < min_n).any():
            threshold_ok = False
            break
    tests.append((f"all grouped tables n >= {min_n}", threshold_ok))

    stats_ok = (len(chi_df) >= 3) and (len(mw_df) == 1) and (len(ztest_df) >= 2)
    tests.append(("stat tests generated", stats_ok))
    tests.append(("at least 5 actionable insights", len(insight_df) >= 5))

    return pd.DataFrame(tests, columns=["test_case", "passed"])


def run_analysis(
    csv_path: str | Path = "Task_to candidates_Final.csv",
    min_n: int = 20,
    show_plots: bool = True,
) -> dict[str, Any]:
    configure_chinese_font()
    raw_source = load_raw_csv(csv_path)
    df_raw = preprocess(raw_source)
    df_cand = build_candidate_view(df_raw)

    quality = build_quality_report(df_raw, df_cand)

    summary = pd.DataFrame(
        [
            summarize_outcome(df_raw, "Process row-level"),
            summarize_outcome(df_cand, "Candidate-level final state"),
        ]
    )

    monthly = monthly_profile(df_cand)
    lead_df, lead_bucket = lead_days_profile(df_cand)

    risk_team = attrition_by_dimension(df_cand, "hiring_team", min_n=min_n)
    risk_position = attrition_by_dimension(df_cand, "hiring_position", min_n=min_n)
    risk_location = attrition_by_dimension(df_cand, "hiring_location", min_n=min_n)
    risk_degree = attrition_by_dimension(df_cand, "degree", min_n=min_n)
    risk_major = attrition_by_dimension(df_cand, "major_clean", min_n=min_n)

    team_pos = interaction_risk(df_cand, ["hiring_team", "hiring_position"], min_n=min_n)
    team_loc = interaction_risk(df_cand, ["hiring_team", "hiring_location"], min_n=min_n)
    pos_lead = interaction_risk(df_cand.dropna(subset=["lead_days_bucket"]), ["hiring_position", "lead_days_bucket"], min_n=min_n)

    reason_counts, reason_mix = reject_reason_profile(df_raw)

    chi_df, mw_df, ztest_df = run_stat_tests(df_cand, min_n=min_n)

    risk_segments = build_risk_segments(df_cand, min_n=min_n)
    insight_df = build_insights(df_cand, reason_counts, risk_segments, monthly, lead_bucket, risk_position)

    checks_df = run_acceptance_checks(
        df_raw,
        df_cand,
        min_n,
        [risk_team, risk_position, risk_location, risk_degree, risk_major, risk_segments],
        chi_df,
        mw_df,
        ztest_df,
        insight_df,
    )

    compare_grain = pd.DataFrame(
        {
            "grain": ["Process row-level", "Candidate-level"],
            "rows": [len(df_raw), len(df_cand)],
            "attrition_rate": [df_raw["is_attrition"].mean(), df_cand["is_attrition"].mean()],
            "accept_rate": [1 - df_raw["is_attrition"].mean(), 1 - df_cand["is_attrition"].mean()],
        }
    )

    sensitivity_rows = []
    for threshold in [10, 20, 30]:
        seg = build_risk_segments(df_cand, min_n=threshold)
        top = seg.head(1)
        sensitivity_rows.append(
            {
                "min_n": threshold,
                "segment_count": len(seg),
                "top_segment": top["segment"].iloc[0] if len(top) else None,
                "top_risk_lift_pp": float(top["risk_lift_pp"].iloc[0]) if len(top) else np.nan,
            }
        )
    sensitivity_df = pd.DataFrame(sensitivity_rows)

    lead_q = df_cand.dropna(subset=["lead_days"]).copy()
    lead_q["lead_q_bucket"] = pd.qcut(lead_q["lead_days"], q=4, duplicates="drop")
    lead_q_summary = (
        lead_q.groupby("lead_q_bucket", observed=False)
        .agg(n=("candidate_id", "count"), attrition_rate=("is_attrition", "mean"))
        .reset_index()
    )

    if show_plots:
        plot_result_distribution(df_cand)
        plot_monthly_profile(monthly)
        plot_lead_days(lead_df, lead_bucket)
        plot_dimension_bars(risk_team, risk_position, risk_location)
        plot_interaction_heatmaps(team_pos, team_loc)
        plot_reject_reason_pareto(reason_counts)
        plot_risk_segments(risk_segments)

    outputs = {
        "df_raw": df_raw,
        "df_cand": df_cand,
        "quality": quality,
        "summary": summary,
        "monthly": monthly,
        "lead_bucket": lead_bucket,
        "risk_team": risk_team,
        "risk_position": risk_position,
        "risk_location": risk_location,
        "risk_degree": risk_degree,
        "risk_major": risk_major,
        "team_pos": team_pos,
        "team_loc": team_loc,
        "pos_lead": pos_lead,
        "reason_counts": reason_counts,
        "reason_mix": reason_mix,
        "chi_df": chi_df,
        "mw_df": mw_df,
        "ztest_df": ztest_df,
        "risk_segments": risk_segments,
        "insight_df": insight_df,
        "checks_df": checks_df,
        "compare_grain": compare_grain,
        "sensitivity_df": sensitivity_df,
        "lead_q_summary": lead_q_summary,
    }
    return outputs


if __name__ == "__main__":
    result = run_analysis(csv_path="Task_to candidates_Final.csv", min_n=20, show_plots=False)
    print("Summary:")
    print(result["summary"])
    print("\nAcceptance checks:")
    print(result["checks_df"])
