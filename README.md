# Reproducible VAR Forecasting Pipeline (Nix + rixpress + Python)

This repository contains a **fully reproducible** forecasting workflow for **quarterly macroeconomic time series**.  
It builds a **bivariate VAR model** (target + 1 selected predictor), makes **recursive 1-step-ahead forecasts**, and exports a set of **figures / tables / metrics** in a clean folder structure.

The whole pipeline is designed to be:
- easy to run (one command),
- easy to reproduce (Nix),
- easy to verify (pytest + GitHub Actions),
- easy to extend (pure functions in `pipeline/functions.py`).

---

## 1) Data (what is inside `data/raw/`)

We use **two Excel files**:

- `data/raw/y.xlsx` (sheet: `data_y`)  
  Contains the **target series** and some related series, quarterly.
- `data/raw/x.xlsx` (sheet: `data_x`)  
  Contains **many candidate predictors** (a large X panel), quarterly.

**Key column names**
- date column: `date`
- target column used in this repo: `import_clv_qna_sa`

> If you want to use your own dataset: keep the same sheet names (`data_x`, `data_y`) and the same `date` column format, or update the command in `run_pipeline.R`.

---

## 2) What analysis is done (method summary)

This workflow follows a standard VAR forecasting procedure:

### Step A — Build a “master” dataset
- Convert `date` to datetime and set it as index.
- Filter sample to an end date (`2025-04-01`).
- Merge Y and X by date.
- Drop X columns with too many missing values (default: keep columns with ≥ 80% non-missing).

### Step B — Data-driven predictor selection (correlation screening)
To avoid spurious correlation caused by trends, we compute **first differences** and then:
- compute absolute correlation between `Δ target` and each `Δ X`,
- list top candidates,
- apply a **coverage filter**: the chosen X must have **no missing values** in the required training/forecast window.

In our current run, the top-5 (absolute correlation with `Δ import_clv_qna_sa`) are:

1. `mb_r_sa_q` (≈ 0.511)  
2. `import_s_clv_qna_sa` (≈ 0.503) ✅ selected after coverage check  
3. `import_b_clv_qna_sa` (≈ 0.500)  
4. `im_r_de_oe_base_q` (≈ 0.479)  
5. `trp51_clv_qna_sa` (≈ 0.456)

### Step C — VAR lag selection (BIC)
We estimate VAR on **differenced series** and select lag length using **BIC**.
- In our current run: **BIC selects VAR(4)**.

### Step D — Recursive 1-step-ahead forecasting (level RMSE)
We do **expanding window** forecasting:

Estimate on `t=1..T`, forecast `T+1`;  
Estimate on `t=1..T+1`, forecast `T+2`; etc.

Forecast period here is the last part of the sample:
- from `2021-10-01` to `2025-04-01` (15 quarters)

We compute **RMSE in levels** (forecasting the level using the differenced forecast + last observed level).

### Step E — Diagnostics (bonus-style outputs)
We export:
- ACF / PACF plots on training differenced series,
- VAR residual whiteness test summary,
- IRF (impulse response) and IRF with reversed ordering (robustness check).

---

## 3) Outputs (where results are saved)

After you run the pipeline, the main outputs are:

### (A) Human-facing report outputs: `./artifacts/`
- `artifacts/metrics.json` ✅ key numbers (best X, selected lag, RMSE, test length)
- `artifacts/cell_02_corr_screen/top5_abs_corr.png`
- `artifacts/cell_03_pair_and_lag/acf_pacf_train.png`
- `artifacts/cell_03_pair_and_lag/whiteness_test.txt`
- `artifacts/cell_04_recursive_forecast/recursive_forecast_level.png`
- `artifacts/cell_04_recursive_forecast/predictions.csv`
- `artifacts/cell_05_irf_and_diagnostics/irf.png`
- `artifacts/cell_05_irf_and_diagnostics/irf_reversed.png`

Example metrics (your run may match this exactly if data is unchanged):
```json
{
  "target_col": "import_clv_qna_sa",
  "best_x_var": "import_s_clv_qna_sa",
  "selected_lag_bic": 4,
  "rmse_level": 1148.4678209699427,
  "n_test": 15
}
````

### (B) rixpress build outputs: `./pipeline-output/`

This folder is created by `rixpress::rxp_make()` + `rxp_copy()` and contains cached derivations (objects produced by the DAG).

---

## 4) Quick start (ONE command, recommended)

### 4.1 Prerequisite

Install **Nix** (recommended for full reproducibility).
Official install guide (copy into browser):

```text
https://nixos.org/download/
```

### 4.2 Run the pipeline

From the project root:

```bash
nix-shell --run "Rscript run_pipeline.R"
```

This will:

1. run the rixpress DAG (build pipeline objects into Nix store, then copy to `./pipeline-output/`)
2. run Python exporter to write figures/tables to `./artifacts/`

After it finishes, check:

* `artifacts/metrics.json`
* `artifacts/cell_04_recursive_forecast/predictions.csv`
* plots under `artifacts/cell_*/`

---

## 5) Run only the Python report exporter (optional)

If you only want the report artifacts (no rixpress build), you can run:

```bash
nix-shell --run 'python -m pipeline.export_outputs \
  --x_path data/raw/x.xlsx --y_path data/raw/y.xlsx \
  --sheet_x data_x --sheet_y data_y \
  --date_col date --target_col import_clv_qna_sa \
  --out_dir artifacts'
```

---

## 6) Tests (unit tests + regression tests)

Run all tests:

```bash
nix-shell --run "pytest -q"
```

What is tested:

* **smoke test**: pipeline runs and writes key output files
* **artifact contract test**: required artifact files exist
* **regression test**: key numbers match the reference result (selected predictor / lag / RMSE)

---

## 7) How to change / extend the pipeline (for future work)

### 7.1 Change target variable

* Update `--target_col ...` in `run_pipeline.R`, OR run the exporter command with a different `--target_col`.
* If you need different default dates/windows, edit `HW2Config` in:

  * `pipeline/functions.py`

### 7.2 Add more diagnostics / new figures

All “analysis logic” should stay in **pure functions** inside:

* `pipeline/functions.py`

Then add new outputs in:

* `build_cell_artifacts(...)` (still no disk I/O: it returns `{path: bytes}`)

Examples of easy extensions:

* Granger causality tests (statsmodels VARResults has causality testing)
* more IRF plots (different impulses/responses)
* multi-step forecasts (h > 1)

### 7.3 Keep functions pure (important rule)

* Pure functions: input DataFrames → output objects (DataFrames / Series / numbers)
* File I/O should only happen at the boundary:

  * `pipeline/export_outputs.py` (read Excel + write artifacts)
  * rixpress encoder helpers (zip/json) if needed

---

## 8) Docker (optional)

This repo also includes `Dockerfile` and `docker-compose.yml`.
If your environment supports Docker, you can containerize execution.

Typical usage pattern:

```bash
docker build -t var-pipeline .
docker run --rm -it -v "$PWD":/work -w /work var-pipeline \
  bash -lc 'nix-shell --run "Rscript run_pipeline.R"'
```

> Note: Docker support depends on your OS setup. If you are on Windows, the easiest path is usually **WSL2 + Docker Desktop**.

---

## 9) CI (GitHub Actions)

A GitHub Actions workflow is included under:

* `.github/workflows/ci.yml`

It runs:

1. `pytest -q`
2. `Rscript run_pipeline.R` (smoke pipeline execution)

So every push checks that:

* code still works,
* outputs are still generated,
* reproducibility is maintained.

---

## 10) Repo structure (high-level)

```text
data/raw/                  input Excel data
pipeline/functions.py      pure analysis functions (core logic)
pipeline/export_outputs.py IO boundary: reads data, writes artifacts
gen-pipeline.R             rixpress DAG definition
run_pipeline.R             one-command entrypoint (recommended)
tests/                     pytest unit + regression tests
default.nix                Nix environment definition
pipeline.nix               rixpress pipeline environment (if used)
.github/workflows/ci.yml   CI definition
```

---

## 11) Notes (very common questions)

### Q1: Should I commit `pipeline-output/` and `artifacts/` to GitHub?

Normally **NO**. They are generated outputs and can be reproduced by one command.
This repo uses `.gitignore` to avoid committing them.

### Q2: Why do I see “No frequency information was provided” warning?

This comes from statsmodels when pandas DateTimeIndex does not have a fixed `freq`.
We attach inferred frequency when possible to reduce such warnings.
It does not affect the final result here.

---

## 12) Credits

This project is built following the reproducible pipeline ideas from RAP4MADS:

```text
https://rap4mads.eu/
```

Main Python libraries:

* pandas
* numpy
* statsmodels
* matplotlib


