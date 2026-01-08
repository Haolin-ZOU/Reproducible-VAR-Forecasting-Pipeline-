FROM nixos/nix:2.18.1

# Make nix behave nicely in Docker
ENV NIX_CONFIG="sandbox = false"

WORKDIR /app
COPY . /app

# Optional warmup: build deps during image build (faster first run)
# Comment out if you want a lighter build step.
RUN nix-shell --run "python -c \"import pandas, numpy, statsmodels, matplotlib\"" \
 && nix-shell --run "R -q -e \"library(rixpress); library(readxl)\""

CMD nix-shell --run "Rscript run_pipeline.R"
