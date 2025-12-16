export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --global_config config_USAMO2025_44samples_2025-12-11_01-38-55.json --dataset USAMO2025 --model openrouter/openai/gpt-5