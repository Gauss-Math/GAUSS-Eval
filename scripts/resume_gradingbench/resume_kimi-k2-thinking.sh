#!/bin/bash

export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --resume_from results/GRADING_BENCH_kimi-k2-thinking_1764875577 --global_config gradingbench.json --dataset GRADING_BENCH --model openrouter/moonshot/kimi-k2-thinking
