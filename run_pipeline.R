# run_pipeline.R
# One-command entrypoint:
#   nix-shell --run "Rscript run_pipeline.R"
#
# It will produce:
#   - pipeline-output/   (rixpress build outputs)
#   - artifacts/         (notebook-style figures/text + metrics.json)

library(rixpress)

# 1) Define the rixpress DAG
source("gen-pipeline.R")

message("Build process started...")
rxp_make()
message("✓ rixpress build completed")

# 2) Copy pipeline outputs to a stable folder name
# IMPORTANT: keep this folder name stable across OS and scripts
rxp_copy(path = "pipeline-output", overwrite = TRUE)
message("✓ copied to ./pipeline-output")

# 3) Export notebook-style artifacts via Python
# (this is the “human-facing report outputs”)
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
if (status != 0) {
  stop("export_outputs failed with status: ", status)
}

message("Done.")
message("  - Rixpress outputs: ./pipeline-output")
message("  - Report artifacts : ./artifacts (see metrics.json)")
