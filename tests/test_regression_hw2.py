import pandas as pd
from pipeline.functions import HW2Config, run_hw2_pipeline

def test_hw2_regression_key_numbers_match_rap_hw2():
    cfg = HW2Config(date_col="date", target_col="import_clv_qna_sa")

    df_x = pd.read_excel("data/raw/x.xlsx", sheet_name="data_x")
    df_y = pd.read_excel("data/raw/y.xlsx", sheet_name="data_y")

    out = run_hw2_pipeline(df_y, df_x, cfg=cfg)

    # 1) test window length
    assert len(out) == 15

    # 2) selected predictor & lag (from RAP_HW2.py / notebook)
    assert out["best_x_var"].iloc[0] == "import_s_clv_qna_sa"
    assert int(out["selected_lag_bic"].iloc[0]) == 4

    # 3) RMSE regression (4 dp consistent with notebook)
    rmse = float(out["rmse_level"].iloc[0])
    assert round(rmse, 4) == 1148.4678

