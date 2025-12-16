#!/bin/bash

export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --resume_from results/GRADING_BENCH_deepseek-chat-v3.1_1765168019 --global_config gradingbench.json --dataset GRADING_BENCH --model openrouter/deepseek/deepseek-chat-v3.1
