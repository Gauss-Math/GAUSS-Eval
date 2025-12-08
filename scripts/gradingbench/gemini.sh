export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --global_config gradingbench.json --dataset GRADING_BENCH --model openrouter/google/gemini-2.5-pro
