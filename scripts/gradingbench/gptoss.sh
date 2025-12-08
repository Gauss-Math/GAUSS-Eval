export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --global_config gradingbench.json --dataset GRADING_BENCH --model openrouter/openai/gpt-oss-120b

