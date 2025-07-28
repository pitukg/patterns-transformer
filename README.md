# Minimal Parity Transformer Experiment

This project trains a minimal Transformer (no positional encodings) to recognize a simple parity rule in a bag-of-tokens setting using Hugging Face Transformers and PyTorch.

## Task
- **Tokens:** `<circle>`, `<triangle>`, `<rectangle>`
- **Goal:** Predict if each token appears an even number of times in a random sequence (label=1 if so, else 0).

## Files
- `data.py`: PyTorch `Dataset` for generating or streaming parity data.
- `model.py`: Custom TransformerEncoder (no positional encodings).
- `train.py`: Hand-written PyTorch training loop (no HF Trainer).
- `config.json`: Hyperparameters and experiment settings.
- `run.sh`: Entrypoint script to launch training.

## Usage
```bash
bash run.sh
```

## Config
Edit `config.json` to change hyperparameters, dataset sizes, or streaming/pre-generation mode.

---

*This is a minimal educational experiment. For research or production, use robust data handling and model code!* 