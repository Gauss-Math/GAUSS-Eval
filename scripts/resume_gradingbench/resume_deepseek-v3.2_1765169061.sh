#!/bin/bash

export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --resume_from results/GRADING_BENCH_deepseek-v3.2_1765169061 --global_config gradingbench.json --dataset GRADING_BENCH --model openrouter/deepseek/deepseek-v3.2
