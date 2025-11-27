export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --dataset IMO2025 --model openrouter/moonshotai/kimi-k2-thinking --global_config global_config.json

