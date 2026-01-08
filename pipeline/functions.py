from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import io
import json
import zipfile

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


@dataclass(frozen=True)
class HW2Config:
    # Columns
    date_col: str = "date"
    target_col: str = "import_clv_qna_sa"

    # Time windows (aligned with RAP_HW2.py)
    analysis_end: pd.Timestamp = pd.Timestamp("2025-04-01")
    cutoff_date: pd.Timestamp = pd.Timestamp("2021-07-01")
    test_start: pd.Timestamp = pd.Timestamp("2021-10-01")
    test_end: pd.Timestamp = pd.Timestamp("2025-04-01")
    last_origin: pd.Timestamp = pd.Timestamp("2025-01-01")

    # Cleaning / screening
    col_na_thresh_ratio: float = 0.8
    top_k: int = 30

    # VAR config
    maxlags_select: int = 8  # RAP_HW2.py uses 8
    irf_horizon: int = 10


# -------------------------
# Pure compute helpers
# -------------------------
def _attach_inferred_freq(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a freq attribute to a DatetimeIndex if it can be inferred.
    Does NOT insert new rows (unlike asfreq). Pure with respect to data values.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    if df.index.freq is not None:
        return df
    freq = pd.infer_freq(df.index)
    if freq:
        out = df.copy()
        out.index = pd.DatetimeIndex(out.index.values, freq=freq)
        return out
    return df

def build_master(
    df_y: pd.DataFrame,
    df_x: pd.DataFrame,
    date_col: str,
    target_col: str,
    analysis_end: pd.Timestamp,
    col_na_thresh_ratio: float = 0.8,
) -> pd.DataFrame:
    """Pure: align y to analysis_end, left-join x, drop sparse x-columns."""
    df_y2 = df_y.copy()
    df_x2 = df_x.copy()

    df_y2[date_col] = pd.to_datetime(df_y2[date_col])
    df_x2[date_col] = pd.to_datetime(df_x2[date_col])

    df_y2 = df_y2[df_y2[date_col] <= analysis_end].copy()

    df_master = (
        pd.merge(df_y2[[date_col, target_col]], df_x2, on=date_col, how="left")
        .set_index(date_col)
        .sort_index()
    )

    thresh = int(len(df_master) * float(col_na_thresh_ratio))
    df_master = df_master.dropna(axis=1, thresh=thresh)
    return df_master


def diff_df(df: pd.DataFrame) -> pd.DataFrame:
    """Pure: 1st difference (keeps NaNs)."""
    return df.diff()


def abs_corr_with_target(df_diff: pd.DataFrame, target_col: str) -> pd.Series:
    """Pure: pairwise correlation with target diff (NA-safe)."""
    corr = df_diff.corrwith(df_diff[target_col])
    return corr.abs().sort_values(ascending=False)


def has_required_coverage(
    s: pd.Series,
    cutoff_date: pd.Timestamp,
    last_origin: pd.Timestamp,
) -> bool:
    """Pure: require no missing values within [cutoff_date, last_origin]."""
    needed = s.loc[cutoff_date:last_origin]
    return bool(needed.notna().all())


def select_best_x_by_corr_and_coverage(
    df_master: pd.DataFrame,
    target_col: str,
    cutoff_date: pd.Timestamp,
    last_origin: pd.Timestamp,
    top_k: int = 30,
) -> Tuple[str, pd.Series, List[str]]:
    """Pure: correlation screening on diffs + coverage filter."""
    df_diff = diff_df(df_master)
    abs_corr = abs_corr_with_target(df_diff, target_col)

    top_candidates = abs_corr.drop(index=target_col, errors="ignore").head(top_k)

    valid_vars: List[str] = []
    for v in list(top_candidates.index):
        if v in df_master.columns and has_required_coverage(df_master[v], cutoff_date, last_origin):
            valid_vars.append(v)

    if not valid_vars:
        raise ValueError(
            "No candidate has full coverage up to last_origin. "
            "Increase top_k or relax NA threshold in build_master()."
        )

    return valid_vars[0], top_candidates, valid_vars


def build_pair_level(df_master: pd.DataFrame, target_col: str, x_col: str) -> pd.DataFrame:
    """Pure: two-column level dataset."""
    return df_master[[target_col, x_col]].copy()


def select_var_lag_bic(df_diff_train: pd.DataFrame, maxlags: int = 8) -> Tuple[int, pd.DataFrame]:
    """Pure: select lag by BIC using statsmodels VAR.select_order."""
    df_use = _attach_inferred_freq(df_diff_train.dropna())   
    model = VAR(df_use)
    sel = model.select_order(maxlags=maxlags)

    chosen = sel.selected_orders.get("bic", None)
    if chosen is None:
        raise ValueError("BIC lag selection returned None.")

    ic_table = pd.DataFrame(sel.ics).T
    ic_table.index.name = "lag"
    return int(chosen), ic_table


def fit_var(df_diff_train: pd.DataFrame, lag: int):
    """Deterministic given inputs."""
    df_use = _attach_inferred_freq(df_diff_train.dropna())  
    model = VAR(df_use)
    return model.fit(lag)


def rolling_recursive_1step_level_forecast(
    df_pair_level: pd.DataFrame,
    target_col: str,
    x_col: str,
    lag: int,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> Tuple[pd.Series, pd.Series]:
    """Pure: expanding-window 1-step-ahead forecast on LEVEL, VAR fit on DIFF."""
    if not isinstance(df_pair_level.index, pd.DatetimeIndex):
        raise TypeError("df_pair_level must have a DatetimeIndex.")
    df_pair_level = df_pair_level.sort_index()

    # test dates come from TARGET (like RAP_HW2.py)
    y_level = df_pair_level[target_col].loc[:test_end]
    test_dates = y_level.loc[test_start:test_end].index

    actual_level: List[float] = []
    pred_level: List[float] = []
    pred_index: List[pd.Timestamp] = []

    idx = df_pair_level.index
    for t_next in test_dates:
        pos = idx.get_loc(t_next)
        if isinstance(pos, slice) or pos == 0:
            raise ValueError(f"Cannot set origin for t_next={t_next} (pos={pos}).")
        t_origin = idx[pos - 1]

        pair_upto = df_pair_level.loc[:t_origin, [target_col, x_col]].dropna()
        diff_upto = pair_upto.diff().dropna()
        diff_upto = _attach_inferred_freq(diff_upto)

        if len(diff_upto) <= lag:
            raise ValueError(
                f"Not enough observations to forecast with lag={lag} at {t_next}. "
                f"Have {len(diff_upto)} differenced rows."
            )

        res = VAR(diff_upto).fit(lag)
        fc_diff = res.forecast(diff_upto.values[-lag:], steps=1)[0]

        # column order is [target_col, x_col]
        yhat_level = float(pair_upto.loc[t_origin, target_col] + fc_diff[0])
        y_true = float(df_pair_level.loc[t_next, target_col])

        actual_level.append(y_true)
        pred_level.append(yhat_level)
        pred_index.append(pd.Timestamp(t_next))

    actual_s = pd.Series(actual_level, index=pd.DatetimeIndex(pred_index), name="actual_level")
    pred_s = pd.Series(pred_level, index=pd.DatetimeIndex(pred_index), name="forecast_level")
    return actual_s, pred_s


def rmse(a: pd.Series, b: pd.Series) -> float:
    """Pure RMSE with alignment."""
    aligned = pd.concat([a, b], axis=1).dropna()
    err = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(np.sqrt(np.mean(err**2)))


def run_hw2_pipeline(df_y: pd.DataFrame, df_x: pd.DataFrame, cfg: Optional[HW2Config] = None) -> pd.DataFrame:
    """Pure: DataFrames in -> predictions DataFrame out (no disk I/O)."""
    cfg = cfg or HW2Config()

    df_master = build_master(
        df_y=df_y,
        df_x=df_x,
        date_col=cfg.date_col,
        target_col=cfg.target_col,
        analysis_end=cfg.analysis_end,
        col_na_thresh_ratio=cfg.col_na_thresh_ratio,
    )

    best_x_var, _, _ = select_best_x_by_corr_and_coverage(
        df_master=df_master,
        target_col=cfg.target_col,
        cutoff_date=cfg.cutoff_date,
        last_origin=cfg.last_origin,
        top_k=cfg.top_k,
    )

    df_pair_level = build_pair_level(df_master, target_col=cfg.target_col, x_col=best_x_var)

    # Lag selection uses training sample up to cutoff_date (like RAP_HW2.py)
    train_pair_level = df_pair_level.loc[:cfg.cutoff_date].dropna()
    train_diff = _attach_inferred_freq(train_pair_level.diff().dropna())

    selected_lag, _ = select_var_lag_bic(train_diff, maxlags=cfg.maxlags_select)

    actual_s, forecast_s = rolling_recursive_1step_level_forecast(
        df_pair_level=df_pair_level,
        target_col=cfg.target_col,
        x_col=best_x_var,
        lag=selected_lag,
        test_start=cfg.test_start,
        test_end=cfg.test_end,
    )

    out = pd.DataFrame(
        {
            "date": actual_s.index,
            "actual_level": actual_s.values,
            "forecast_level": forecast_s.values,
        }
    )
    out["best_x_var"] = best_x_var
    out["selected_lag_bic"] = selected_lag
    out["selected_lag"] = selected_lag
    out["rmse_level"] = rmse(actual_s, forecast_s)
    return out


# -------------------------
# rixpress encoder helpers (I/O boundary)
# -------------------------
def serialize_df_to_json(df: pd.DataFrame, out_path: str) -> None:
    """Write list-of-records JSON (rixpress encoder style)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    df2 = df.copy()
    if "date" in df2.columns:
        df2["date"] = pd.to_datetime(df2["date"]).dt.strftime("%Y-%m-%d")

    p.write_text(json.dumps(df2.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")


def build_cell_artifacts(df_y: pd.DataFrame, df_x: pd.DataFrame, cfg: Optional[HW2Config] = None) -> Dict[str, bytes]:
    """Build notebook-style artifacts in-memory as {relative_path: bytes}.
    Pure in the sense: no disk I/O; matplotlib styling confined to context managers.
    """
    cfg = cfg or HW2Config()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rc_context
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    artifacts: Dict[str, bytes] = {}

    def put_text(rel: str, text: str) -> None:
        artifacts[rel] = text.encode("utf-8")

    def put_fig(rel: str, fig) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        artifacts[rel] = buf.getvalue()

    # ---------- cell 01: master ----------
    df_master = build_master(
        df_y=df_y,
        df_x=df_x,
        date_col=cfg.date_col,
        target_col=cfg.target_col,
        analysis_end=cfg.analysis_end,
        col_na_thresh_ratio=cfg.col_na_thresh_ratio,
    )

    put_text(
        "cell_01_load_master/summary.txt",
        "\n".join(
            [
                f"df_master shape: {df_master.shape}",
                f"date range: {df_master.index.min().date()} .. {df_master.index.max().date()}",
                f"columns: {len(df_master.columns)}",
            ]
        )
        + "\n",
    )

    # ---------- cell 02: corr screen ----------
    best_x_var, top_candidates, valid_vars = select_best_x_by_corr_and_coverage(
        df_master=df_master,
        target_col=cfg.target_col,
        cutoff_date=cfg.cutoff_date,
        last_origin=cfg.last_origin,
        top_k=cfg.top_k,
    )

    with plt.style.context("bmh"), rc_context({"figure.figsize": (10, 5)}):
        fig = plt.figure()
        top_candidates.head(5).sort_values().plot(kind="barh")
        plt.title(f"Top 5 Variables Correlated with D({cfg.target_col})")
        plt.xlabel("Absolute Correlation")
        put_fig("cell_02_corr_screen/top5_abs_corr.png", fig)

    put_text(
        "cell_02_corr_screen/top10_candidates.txt",
        "Top 10 candidates (abs corr):\n" + top_candidates.head(10).to_string() + "\n",
    )
    put_text(
        "cell_02_corr_screen/coverage_filter.txt",
        "\n".join(
            [
                f"cutoff_date = {cfg.cutoff_date.date()}",
                f"last_origin = {cfg.last_origin.date()}",
                "",
                "valid_vars (first 20):",
                "\n".join(valid_vars[:20]),
                "",
                f"SELECTED best_x_var = {best_x_var}",
                "",
            ]
        ),
    )

    # ---------- build pair & train diff ----------
    df_pair_level = build_pair_level(df_master, target_col=cfg.target_col, x_col=best_x_var)
    train_pair_level = df_pair_level.loc[:cfg.cutoff_date].dropna()
    train_diff = _attach_inferred_freq(train_pair_level.diff().dropna())

    # ---------- cell 03: ACF/PACF + lag selection + whiteness ----------
    with plt.style.context("bmh"), rc_context({"figure.figsize": (14, 8)}):
        fig, axes = plt.subplots(2, 2)

        plot_acf(train_diff[cfg.target_col], lags=12, ax=axes[0, 0])
        axes[0, 0].set_title(f"ACF of D({cfg.target_col}) (train)")

        plot_pacf(train_diff[cfg.target_col], lags=12, ax=axes[0, 1], method="ywm")
        axes[0, 1].set_title(f"PACF of D({cfg.target_col}) (train)")

        plot_acf(train_diff[best_x_var], lags=12, ax=axes[1, 0])
        axes[1, 0].set_title(f"ACF of D({best_x_var}) (train)")

        plot_pacf(train_diff[best_x_var], lags=12, ax=axes[1, 1], method="ywm")
        axes[1, 1].set_title(f"PACF of D({best_x_var}) (train)")

        plt.tight_layout()
        put_fig("cell_03_pair_and_lag/acf_pacf_train.png", fig)

    # Lag selection + fit VAR
    selected_lag, _ = select_var_lag_bic(train_diff, maxlags=cfg.maxlags_select)
    res = fit_var(train_diff, selected_lag)

    # Whiteness test (text)
    try:
        wt = res.test_whiteness(nlags=10)
        put_text("cell_03_pair_and_lag/whiteness_test.txt", str(wt.summary()) + "\n")
    except Exception as e:
        put_text("cell_03_pair_and_lag/whiteness_test.txt", f"Whiteness test failed: {e}\n")

    # ---------- cell 04: recursive forecast ----------
    actual_s, forecast_s = rolling_recursive_1step_level_forecast(
        df_pair_level=df_pair_level,
        target_col=cfg.target_col,
        x_col=best_x_var,
        lag=selected_lag,
        test_start=cfg.test_start,
        test_end=cfg.test_end,
    )
    rmse_level = rmse(actual_s, forecast_s)

    pred_df = pd.DataFrame(
        {
            "date": actual_s.index,
            "actual_level": actual_s.values,
            "forecast_level": forecast_s.values,
        }
    )
    pred_df["best_x_var"] = best_x_var
    pred_df["selected_lag_bic"] = selected_lag
    pred_df["rmse_level"] = rmse_level

    put_text("cell_04_recursive_forecast/rmse.txt", f"RMSE(level) = {rmse_level:.6f}\n")

    with plt.style.context("bmh"), rc_context({"figure.figsize": (12, 5)}):
        fig = plt.figure()
        plt.plot(pred_df["date"], pred_df["actual_level"], label="Actual", marker="o")
        plt.plot(pred_df["date"], pred_df["forecast_level"], label="Forecast", linestyle="--", marker="o")
        plt.title(f"Recursive 1-step forecast: x={best_x_var}, VAR({selected_lag})")
        plt.legend()
        put_fig("cell_04_recursive_forecast/recursive_forecast_level.png", fig)

    artifacts["cell_04_recursive_forecast/predictions.csv"] = pred_df.to_csv(index=False).encode("utf-8")

    # ---------- cell 05: IRF + IRF reversed ----------
    try:
        irf = res.irf(cfg.irf_horizon)
        fig = irf.plot(orth=True, impulse=best_x_var, response=cfg.target_col)
        put_fig("cell_05_irf_and_diagnostics/irf.png", fig)
    except Exception as e:
        put_text("cell_05_irf_and_diagnostics/irf.png.ERROR.txt", f"IRF failed: {e}\n")

    try:
        # reversed ordering robustness
        res_rev = fit_var(train_diff[[best_x_var, cfg.target_col]], selected_lag)
        irf_rev = res_rev.irf(cfg.irf_horizon)
        fig = irf_rev.plot(orth=True, impulse=best_x_var, response=cfg.target_col)
        put_fig("cell_05_irf_and_diagnostics/irf_reversed.png", fig)
    except Exception as e:
        put_text("cell_05_irf_and_diagnostics/irf_reversed.png.ERROR.txt", f"IRF reversed failed: {e}\n")

    # ---------- metrics.json ----------
    metrics = {
        "target_col": cfg.target_col,
        "best_x_var": best_x_var,
        "selected_lag_bic": int(selected_lag),
        "rmse_level": float(rmse_level),
        "n_test": int(len(pred_df)),
    }
    artifacts["metrics.json"] = json.dumps(metrics, indent=2, ensure_ascii=False).encode("utf-8")
    return artifacts


def zip_dir_to_file(obj: Union[Dict[str, bytes], str, Path], out_path: str) -> None:
    """Create zip at out_path from either dict{rel:bytes} or an on-disk directory."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if isinstance(obj, dict):
            for rel in sorted(obj.keys()):
                zf.writestr(rel, obj[rel])
        else:
            root = Path(obj)
            if not root.exists():
                raise FileNotFoundError(f"Directory does not exist: {root}")
            for p in root.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(root)))
