export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --dataset USAMO2025 --model openrouter/openai/gpt-oss-120b --global_config global_config.json