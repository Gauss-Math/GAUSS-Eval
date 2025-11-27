export OPENROUTER_API_KEY=$(cat api/openrouter.key)

# python src/main.py --dataset USAMO2025 --model openrouter/qwen/qwen3-max --global_config global_config.json
python src/main.py --resume_from results/USAMO2025_qwen3-max_1761922242
