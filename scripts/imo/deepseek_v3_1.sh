export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --dataset IMO2025 --model openrouter/deepseek/deepseek-chat-v3.1 --global_config global_config.json

