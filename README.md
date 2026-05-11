# LLM — GPT Training from Scratch

A from-scratch implementation of a GPT-style language model in PyTorch, including a custom BPE tokenizer, sliding-window dataloader, and a full training loop with evaluation and text generation.

---

## Project Structure

```
llm/
├── main.py                 # Entry point: training pipeline
├── config.yaml             # All hyperparameters and paths
├── dataset/
│   ├── The_Verdict.txt     # Training text
│   └── tokenizer.json      # Saved tokenizer (auto-generated)
├── models/
│   └── model.pth           # Saved model checkpoint (auto-generated)
├── src/
│   ├── __init__.py
│   ├── tokenizer.py        # Custom BPE tokenizer
│   ├── dataloader.py       # Sliding-window dataset and DataLoader
│   ├── llm_parts.py        # GPT model architecture
│   ├── llm_utils.py        # Training loop, generation, loss utilities
│   └── logger.py           # Logging setup
└── loss-plot.pdf           # Training/validation loss curve (auto-generated)
```

---

## Installation

Requires Python 3.9+.

```bash
pip install uv
pip install -r requirements.txt
```

---

## Usage

Run the full pipeline (tokenize, build model, train, generate):

```bash
python main.py
```

All behavior is controlled through `config.yaml` — no command-line arguments needed.

---

## Configuration (`config.yaml`)

### `tokenizer`

| Key | Description |
|---|---|
| `tok_load_from` | Path to a saved tokenizer JSON. Leave empty `""` to train a new one. |
| `tok_save_path` | Where to save the trained tokenizer. Leave empty `""` to skip saving. |
| `vocab_size` | Target vocabulary size for BPE training. |
| `min_freq` | Stop BPE merges when the best pair appears fewer than this many times. Set to `null` to always train until `vocab_size`. |
| `eos_token` | End-of-sequence special token. |
| `unk_token` | Unknown token used for unseen characters. |

### `dataloader`

| Key | Description |
|---|---|
| `batch_size` | Number of sequences per batch. |
| `context_length` | Length of each input sequence (also the model's context window). |
| `stride` | Step size for the sliding window. Smaller = more overlapping samples = more batches per epoch. Equal to `context_length` gives non-overlapping windows. |
| `drop_last` | Drop the last incomplete batch. |
| `shuffle` | Shuffle training samples each epoch. |
| `num_workers` | DataLoader worker processes (0 = main process only). |

### `optimizer`

| Key | Description |
|---|---|
| `learning_rate` | AdamW learning rate. |
| `weight_decay` | L2 regularization coefficient. |

### `llm_init`

| Key | Description |
|---|---|
| `emb_dim` | Token and positional embedding dimension. |
| `n_heads` | Number of attention heads (must divide `emb_dim` evenly). |
| `n_layers` | Number of transformer blocks. |
| `dropout` | Dropout rate applied throughout the model. |
| `qkv_bias` | Whether to include bias terms in Q/K/V projections. |

### `llm_train`

| Key | Description |
|---|---|
| `from_pretrained` | Path to a `.pth` checkpoint to resume from. Leave empty `""` to train from scratch. |
| `model_save_path` | Where to save the trained model. Leave empty `""` to skip saving. |
| `epochs` | Number of full passes through the training data. |
| `train_val_split` | Fraction of data used for training (e.g. `0.9` = 90% train, 10% val). |

### `llm_gen`

| Key | Description |
|---|---|
| `max_gen_length` | Number of new tokens to generate during sampling. |
| `temperature` | Sampling temperature. Higher = more random. `0.0` = greedy argmax. |
| `top_k` | Restrict sampling to the top-k most likely tokens at each step. |

---

## Model Architecture

The model is a decoder-only GPT transformer:

- **Token + positional embeddings** learned from scratch
- **N transformer blocks**, each containing:
  - Multi-head causal self-attention (with upper-triangular mask)
  - Feed-forward block (linear → GELU → linear, 4× expansion)
  - Pre-norm LayerNorm on both sub-layers
  - Residual (shortcut) connections
- **Final LayerNorm** before the output head
- **Linear output head** projecting to vocabulary logits

### Tokenizer

A flat character-level BPE tokenizer (no word pre-splitting). The full text is treated as a continuous character sequence and merges are learned greedily using a max-heap with lazy deletion for efficiency. Special tokens (`<|EOS|>`, `<|UNK|>`) are handled atomically and never split.

---

## Training Pipeline (`main.py`)

1. Load config and set up logger
2. Load or train the BPE tokenizer
3. Initialize or load a pretrained GPT model and AdamW optimizer
4. Generate a sample with the untrained/loaded model
5. Build train and validation `DataLoader`s using a sliding window
6. Compute baseline train/val loss (5 batches each)
7. Train for `epochs` epochs with periodic evaluation every `eval_freq` steps
8. Save the model checkpoint if a path is configured
9. Generate 5 samples with the trained model

---

## Checkpoint Format

Checkpoints are saved as a dict with two keys:

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
}
```

This allows resuming training with the exact optimizer state (momentum, variance) by setting `from_pretrained` in the config.
