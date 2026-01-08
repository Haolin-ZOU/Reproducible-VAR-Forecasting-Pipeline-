import pandas as pd
from pipeline.functions import HW2Config, build_cell_artifacts

def test_artifacts_contract_contains_key_outputs():
    cfg = HW2Config(date_col="date", target_col="import_clv_qna_sa")
    df_x = pd.read_excel("data/raw/x.xlsx", sheet_name="data_x")
    df_y = pd.read_excel("data/raw/y.xlsx", sheet_name="data_y")

    art = build_cell_artifacts(df_y, df_x, cfg=cfg)

    required = [
        "metrics.json",
        "cell_02_corr_screen/top5_abs_corr.png",
        "cell_03_pair_and_lag/acf_pacf_train.png",
        "cell_03_pair_and_lag/whiteness_test.txt",
        "cell_04_recursive_forecast/recursive_forecast_level.png",
        "cell_05_irf_and_diagnostics/irf.png",
        "cell_05_irf_and_diagnostics/irf_reversed.png",
    ]
    for k in required:
        assert k in art, f"Missing artifact: {k}"

    # Quick sanity: PNG signature
    assert art["cell_02_corr_screen/top5_abs_corr.png"][:8] == b"\x89PNG\r\n\x1a\n"

