export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --dataset USAMO2025 --model openrouter/google/gemini-2.5-pro --global_config global_config.json
