FROM nixos/nix:2.22.1

WORKDIR /app
COPY . /app

# Speed up builds a bit; also avoids some locale issues in R/Python
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Optional: warm-up build (will cache Nix store layers in the image build)
# If you want faster `docker run`, keep this. If you want faster `docker build`, remove it.
RUN nix-shell --run "python -c \"import pandas, numpy, statsmodels, matplotlib\""

# Default command: run the full pipeline (rixpress build + export artifacts)
CMD ["bash", "-lc", "nix-shell --run \"Rscript run_pipeline.R\""]
