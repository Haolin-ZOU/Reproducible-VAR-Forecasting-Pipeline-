library(rixpress)

list(
  rxp_py_file(
    name = y_raw,
    path = "data/raw/y.xlsx",
    read_function = "lambda p: __import__('pandas').read_excel(p, sheet_name='data_y')"
  ),
  rxp_py_file(
    name = x_raw,
    path = "data/raw/x.xlsx",
    read_function = "lambda p: __import__('pandas').read_excel(p, sheet_name='data_x')"
  ),
  rxp_py(
    name = preds_json,
    expr = "run_hw2_pipeline(y_raw, x_raw)",
    user_functions = "pipeline/functions.py",
    encoder = "serialize_df_to_json"
  ),
    rxp_py(
    name = cells_zip,
    expr = "build_cell_artifacts(y_raw, x_raw)",
    user_functions = "pipeline/functions.py",
    encoder = "zip_dir_to_file"
  )
) |>
  rxp_populate(build = FALSE)

