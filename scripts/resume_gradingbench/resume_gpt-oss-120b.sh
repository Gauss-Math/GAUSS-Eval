#!/bin/bash

export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --resume_from results/GRADING_BENCH_gpt-oss-120b_1765167980 --global_config gradingbench.json --dataset GRADING_BENCH --model openrouter/openai/gpt-oss-120b
