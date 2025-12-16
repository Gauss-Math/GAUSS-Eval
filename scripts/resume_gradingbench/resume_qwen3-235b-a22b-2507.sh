#!/bin/bash

export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --resume_from results/GRADING_BENCH_qwen3-235b-a22b-2507_1765291907 --global_config gradingbench.json --dataset GRADING_BENCH --model openrouter/qwen/qwen3-235b-a22b-2507
