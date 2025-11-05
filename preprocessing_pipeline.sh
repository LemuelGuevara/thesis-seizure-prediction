#!/bin/bash

set -e

UV_COMMANDS=(
    "uv run -m src.preprocessing.run_precompute_stfts_pipeline"
    "uv run -m src.preprocessing.run_time_frequency_pipeline"
    "uv run -m src.preprocessing.run_bispectrum_pipeline"
)

echo "Starting preprocessing pipeline..."
echo

TOTAL_START=$(date +%s)

for cmd in "${UV_COMMANDS[@]}"; do
    echo "---------------------------------------------"
    echo "Running: $cmd"
    STEP_START=$(date +%s)

    $cmd

    STEP_END=$(date +%s)
    STEP_DURATION=$((STEP_END - STEP_START))

    printf "Completed in %02d:%02d:%02d\n\n" \
        $((STEP_DURATION/3600)) $(((STEP_DURATION/60)%60)) $((STEP_DURATION%60))
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo "---------------------------------------------"
printf "All preprocessing completed in %02d:%02d:%02d\n" \
    $((TOTAL_DURATION/3600)) $(((TOTAL_DURATION/60)%60)) $((TOTAL_DURATION%60))
