#!/bin/bash

export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --resume_from results/GRADING_BENCH_gemini-3-pro-preview_1764875502 --global_config gradingbench.json --dataset GRADING_BENCH --model openrouter/google/gemini-3-pro-preview
