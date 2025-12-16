#!/bin/bash

export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --resume_from results/GRADING_BENCH_gemini-2.5-pro_1765168085 --global_config gradingbench.json --dataset GRADING_BENCH --model openrouter/google/gemini-2.5-pro
