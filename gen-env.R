library(rix)

rix(
  date = "2025-10-14",
  r_pkgs = c(
    "rixpress",
    "jsonlite",
    "readxl",
    "dplyr",
    "ggplot2",
    "glue",
    "fs",
    "reticulate"
  ),
  py_conf = list(
    py_version = "3.11",
    py_pkgs = c(
      "pandas",
      "numpy",
      "matplotlib",
      "statsmodels",
      "scipy",
      "openpyxl",
      "pytest"
    )
  ),
  system_pkgs = c("git", "uv", "fontconfig"),
  ide = "none",
  project_path = ".",
  overwrite = TRUE,
  print = TRUE
)
