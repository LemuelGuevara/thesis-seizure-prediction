#!/bin/bash

set -e

UV_COMMANDS=(
    "uv run -m src.preprocessing.run_precompute_stfts_pipeline"
    "uv run -m src.preprocessing.run_time_frequency_pipeline"
    "uv run -m src.preprocessing.run_bispectrum_pipeline"
)

echo "Starting preprocessing pipeline..."

for cmd in "${UV_COMMANDS[@]}"; do
    echo "Running: $cmd"
    $cmd
done

echo "Preprocessing pipeline completed."
