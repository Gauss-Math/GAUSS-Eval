export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --dataset IMO2025 --model openrouter/qwen/qwen3-32b --global_config global_config.json

