# tests/test_smoke.py
import tempfile
from pathlib import Path

from pipeline.export_outputs import run_all


def test_pipeline_smoke_runs_and_writes_outputs():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "artifacts"
        metrics = run_all(
            x_path="data/raw/x.xlsx",
            y_path="data/raw/y.xlsx",
            sheet_x="data_x",
            sheet_y="data_y",
            date_col="date",
            target_col="import_clv_qna_sa",
            out_dir=str(out),
        )

        assert (out / "metrics.json").exists()
        assert (out / "cell_02_corr_screen" / "top5_abs_corr.png").exists()
        assert (out / "cell_04_recursive_forecast" / "recursive_forecast_level.png").exists()
        assert "rmse_level" in metrics
        assert metrics["selected_lag_bic"] >= 1
