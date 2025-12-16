#!/bin/bash

export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --resume_from results/GRADING_BENCH_qwen3-32b_1765293252 --global_config gradingbench.json --dataset GRADING_BENCH --model openrouter/qwen/qwen3-32b
