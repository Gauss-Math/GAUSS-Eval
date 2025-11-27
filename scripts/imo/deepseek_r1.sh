export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --dataset IMO2025 --model openrouter/deepseek/deepseek-r1-0528 --global_config global_config.json

