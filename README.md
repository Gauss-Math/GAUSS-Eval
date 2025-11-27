<div align="center">

## GAUSS Eval: <br> Human–LLM Judge Consistency Analysis

![3d](assets/3d.png)

<a href="https://gauss.ai/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-gauss.ai-blue" height="25" /></a>

<div style="font-family: charter;">
GAUSS Team
</div>
</div>

## Release
- [Nov 28, 2025] We release the code and blog

## Usage

1. Clone this repo:
```shell
git clone git@github.com:Gauss-Math/GAUSS-Eval.git
```

2. Install the dependencies:
```
cd GAUSS-Eval
uv sync
source .venv/bin/activate
export GAUSS_EVAL_ROOT=$(pwd)
```

3. Prepare your API keys and put it into ```api/{FILE_NAME}.key```. Example usage:
```shell
export OPENROUTER_API_KEY=$(cat api/openrouter.key)
```

4. (Optional) Configure prompt, data, rubrics, etc using UI with specified ```{PORT_NAME}```, i.e. ```10100```:
```
cd ui && python server.py 10100
```

5. Run the inference either using commands in the UI or in ```scripts/```.

    - Example: ```bash scripts/usamo/gpt5.sh```
    - If your run does not generate responses for some samples (usually due to unstable providers), please run: ```python src/main.py --resume_from {YOUR_RESULT_DIR}``` and make sure a temporary summary ```summary.json``` is in it.

6. Generate brief summary using:
```
python src/analyze.py
```

7. (Optional) Generate reports using:
```
python src/analyze.py --generate-reports --figure
```
