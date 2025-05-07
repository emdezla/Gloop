# Gloop-LLM

## Installation
```pip install -r requirements.txt```

Set up hugging face

```
export HF_TOKEN="hf_…your_token…"
huggingface-cli login --token $HF_TOKEN --add-to-git-credential

export WANDB_API_KEY="...your key"
```

## Training
### 1. Generate Image Data
Run `data_formating.ipynb`

### 2. Train
Run `fine_tuning_vlm_trl.ipynb`