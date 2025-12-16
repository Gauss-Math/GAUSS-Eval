export OPENROUTER_API_KEY=$(cat api/openrouter.key)

# python src/main.py --global_config gradingbench.json --dataset GRADING_BENCH --model openrouter/deepseek/deepseek-r1-0528
python src/main.py --resume_from results/GRADING_BENCH_deepseek-r1-0528_1764876026 --global_config gradingbench.json --dataset GRADING_BENCH --model openrouter/deepseek/deepseek-r1-0528