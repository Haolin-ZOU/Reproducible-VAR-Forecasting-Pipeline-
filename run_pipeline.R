# run_pipeline.R
library(rixpress)

source("gen-pipeline.R")

message("Build process started...")
rxp_make()
message("✓ rixpress build completed")

# Your rixpress version: rxp_copy() takes NO (path/overwrite) args
rxp_copy()

# Normalize folder name to stable: ./pipeline-output
if (dir.exists("pipeline-outputs") && !dir.exists("pipeline-output")) {
  file.rename("pipeline-outputs", "pipeline-output")
}

if (!dir.exists("pipeline-output")) {
  stop("rxp_copy() finished but ./pipeline-output was not found. Check which folder was created.")
}
message("✓ copied to ./pipeline-output")

cmd <- paste(
  "python -m pipeline.export_outputs",
  "--x_path data/raw/x.xlsx",
  "--y_path data/raw/y.xlsx",
  "--sheet_x data_x",
  "--sheet_y data_y",
  "--date_col date",
  "--target_col import_clv_qna_sa",
  "--out_dir artifacts"
)

message("Running: ", cmd)
status <- system(cmd)
if (status != 0) stop("export_outputs failed with status: ", status)

message("Done.")
message("  - rixpress outputs: ./pipeline-output")
message("  - report outputs  : ./artifacts")
