export OPENROUTER_API_KEY=$(cat api/openrouter.key)

# python src/main.py --global_config gradingbench.json --dataset GRADING_BENCH --model openrouter/openai/gpt-5.2

python src/main.py --resume_from results/GRADING_BENCH_gpt-5.2_1765508578 --global_config gradingbench.json --dataset GRADING_BENCH --model openrouter/openai/gpt-5.2