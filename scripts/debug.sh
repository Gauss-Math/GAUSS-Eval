export OPENROUTER_API_KEY=$(cat api/openrouter.key)

python src/main.py --dataset DEBUG --model openrouter/openai/gpt-4o-mini --global_config global_config.json