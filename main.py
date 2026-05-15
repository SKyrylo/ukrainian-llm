import torch

from src.dataloader import create_dataloader
from src.tokenizer import BPETokenizer
from src.llm_utils import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
    calc_loss_batch,
    calc_loss_loader,
    train_model,
    plot_losses
)
from src.llm_parts import GPTModel

from time import time
import yaml
from types import SimpleNamespace
import os
from pathlib import Path
import logging
from src.logger import setup_logger
from dotenv import load_dotenv
load_dotenv()


def _resolve_output_dir(config) -> Path | None:
    """Return the resolved output directory (and create it), or None when
    output_dir is not set.

    When using Google Drive on Colab, mount the drive BEFORE running this
    script by adding a cell at the top of your notebook:
        from google.colab import drive
        drive.mount('/content/drive')
    Then set output_dir in config.yaml to e.g. /content/drive/MyDrive/llm
    """
    raw = getattr(config, "output_dir", "") or ""
    if not raw:
        return None

    out = Path(raw)
    if str(out).startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
        raise RuntimeError(
            "output_dir points to Google Drive but Drive is not mounted.\n"
            "Run this in a Colab notebook cell first:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')"
        )

    out.mkdir(parents=True, exist_ok=True)
    return out


def _out(output_dir: Path | None, filename: str) -> Path:
    """Resolve a filename to output_dir, or keep it as a local relative path."""
    if output_dir is not None:
        return output_dir / filename
    return Path(filename)


def main(
    config,
    logger,
    start_context="Hello, I am"
    ):
    output_dir = _resolve_output_dir(config)
    if output_dir:
        logger.info("Output directory: %s", output_dir)
    else:
        logger.info("Output directory: local (output_dir not set)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset_dir = Path(config.dataset_dir)
    txt_files = sorted(dataset_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {dataset_dir.resolve()}")
    logger.info("Found %d dataset file(s) in %s", len(txt_files), dataset_dir.resolve())

    eos = config.tokenizer['eos_token']
    texts = []
    for fpath in txt_files:
        with open(fpath, encoding="utf-8", errors="replace") as f:
            texts.append(f.read())
    text_data = f" {eos} ".join(texts)

    if config.tokenizer['tok_load_from'] == "":
        # BPE training is O(n) in text length — cap the sample to avoid OOM
        # on large corpora. 10 MB of diverse text is enough for 32k merges.
        tok_max_chars = config.tokenizer.get('tok_train_max_chars', 10_000_000)
        tok_train_text = text_data[:tok_max_chars]
        logger.info(
            "Training a new tokenizer on %.1f MB sample (full corpus: %.1f MB)...",
            len(tok_train_text) / 1e6,
            len(text_data) / 1e6,
        )
        tokenizer = BPETokenizer()
        tokenizer.fit(
            tok_train_text,
            vocab_size=config.tokenizer['vocab_size'],
            min_freq=config.tokenizer['min_freq'],
            eos_token=config.tokenizer['eos_token'],
            unk_token=config.tokenizer['unk_token']
        )

        if config.tokenizer['tok_save_path'] != "":
            tok_save = _out(output_dir, config.tokenizer['tok_save_path'])
            tok_save.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(str(tok_save))
            logger.info("Tokenizer trained and saved to %s", tok_save)
    else:
        tok_load = config.tokenizer['tok_load_from']
        logger.info("Loading existing tokenizer from %s", tok_load)
        tokenizer = BPETokenizer(tok_load)
    
    MODEL_CONFIG = {
        "vocab_size": len(tokenizer.str2int),                   # Vocabulary size
        "context_length": config.dataloader['context_length'],  # Context length
        "emb_dim": config.llm_init['emb_dim'],                  # Embedding dimension
        "num_heads": config.llm_init['n_heads'],                # Number of attention heads
        "num_layers": config.llm_init['n_layers'],              # Number of layers
        "drop_rate": config.llm_init['dropout'],                # Dropout rate
        "qkv_bias": config.llm_init['qkv_bias']                 # Query-Key-Value bias
    }

    if config.llm_train['from_pretrained']:
        logger.info("Loading pretrained model from %s", config.llm_train['from_pretrained'])
        
        checkpoint = torch.load(
            config.llm_train['from_pretrained'],
            weights_only=True
        )

        model = GPTModel(MODEL_CONFIG)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer['learning_rate'],
            weight_decay=config.optimizer['weight_decay']
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        logger.info("Initializing a new model with configuration: %s", MODEL_CONFIG)

        model = GPTModel(MODEL_CONFIG)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer['learning_rate'],
            weight_decay=config.optimizer['weight_decay']
        )
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Model configuration: {MODEL_CONFIG}")
    logger.info(f"Model total parameters (all trainable): {total_params}")
    logger.info(f"Total size of the model: {total_params * 4 / (1024 ** 2):.2f} MB (assuming 4 bytes per parameter)")
    
    model.eval() 
    out = generate(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer).to(device),
        max_new_tokens=config.llm_gen['max_gen_length'],
        context_size=MODEL_CONFIG["context_length"],
        top_k=config.llm_gen['top_k'],
        temperature=config.llm_gen['temperature']
        )

    print(token_ids_to_text(out, tokenizer))

    # ── Tokenise corpus (with disk cache) ───────────────────────────────────
    # tokenizer.encode() is O(n × merges) in pure Python — encoding the full
    # corpus at once would take days.  We therefore:
    #   1. Cap the text at encode_max_chars (default 30 MB).
    #   2. Cache the resulting token-ID tensor to disk so subsequent runs
    #      skip encoding entirely and load in seconds.
    encode_max = config.dataloader.get('encode_max_chars', 30_000_000)
    encode_text = text_data[:encode_max]
    logger.info(
        "Corpus for training: %.1f MB (encode_max_chars=%.0f M)",
        len(encode_text) / 1e6, encode_max / 1e6,
    )

    cache_path = _out(output_dir, "tokens_cache.pt") if output_dir \
        else Path("tokens_cache.pt")

    if cache_path.exists():
        logger.info("Loading cached token IDs from %s ...", cache_path)
        all_token_ids = torch.load(cache_path, weights_only=True)
        logger.info("Cache loaded: %d tokens", len(all_token_ids))
    else:
        logger.info(
            "Encoding %.1f MB of text — this runs on CPU and may take "
            "30–90 minutes depending on vocab size. It will be cached "
            "to disk and skipped on future runs.", len(encode_text) / 1e6
        )
        all_token_ids = tokenizer.encode(encode_text)
        torch.save(all_token_ids, cache_path)
        logger.info(
            "Encoded %d tokens → cached to %s", len(all_token_ids), cache_path
        )

    # ── Train / validation split ─────────────────────────────────────────────
    train_ratio = config.llm_train['train_val_split']
    split_idx = int(train_ratio * len(all_token_ids))
    train_ids = all_token_ids[:split_idx]
    val_ids   = all_token_ids[split_idx:]

    train_loader = create_dataloader(
        train_ids,
        batch_size=config.dataloader['batch_size'],
        context_length=MODEL_CONFIG["context_length"],
        stride=config.dataloader['stride'],
        drop_last=config.dataloader['drop_last'],
        shuffle=config.dataloader['shuffle'],
        num_workers=config.dataloader['num_workers'],
    )

    val_loader = create_dataloader(
        val_ids,
        batch_size=config.dataloader['batch_size'],
        context_length=MODEL_CONFIG["context_length"],
        stride=MODEL_CONFIG["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=config.dataloader['num_workers'],
    )

    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)
        break

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)
        break

    print(len(train_loader))
    print(len(val_loader))

    context_length = MODEL_CONFIG["context_length"]
    train_tokens = len(train_loader.dataset) * context_length
    val_tokens = len(val_loader.dataset) * context_length

    print("Training tokens:", train_tokens)
    print("Validation tokens:", val_tokens)
    print("All tokens:", train_tokens + val_tokens)

    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    model.train()

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    start_time = time()

    num_epochs = config.llm_train['epochs']
    train_losses, val_losses, tokens_seen = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=config.llm_train.get('eval_freq', 500),
        eval_iter=config.llm_train.get('eval_iter', 20),
        start_context="Українська мова",
        tokenizer=tokenizer,
        temperature=config.llm_gen['temperature'],
        top_k=config.llm_gen['top_k'],
        patience=config.llm_train.get('patience', 3),
        save_best_path=str(_out(output_dir, config.llm_train['model_save_path']))
            if config.llm_train.get('model_save_path') else None,
    )
    model.eval()

    end_time = time()

    if config.llm_train['model_save_path']:
        final_path = _out(output_dir, config.llm_train['model_save_path'])
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            final_path,
        )
        logger.info("Final model saved to %s", final_path)

    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_path = str(_out(output_dir, "loss-plot.pdf"))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, plot_path)

    for i in range(5):
        print(f"\nSample {i+1}:")
        out = generate(
            model=model,
            idx=text_to_token_ids(start_context, tokenizer).to(device),
            max_new_tokens=config.llm_gen['max_gen_length'],
            context_size=MODEL_CONFIG["context_length"],
            top_k=config.llm_gen['top_k'],
            temperature=config.llm_gen['temperature']
            )

        print(token_ids_to_text(out, tokenizer))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logger = setup_logger(__name__)

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        config = SimpleNamespace(**config)
    
    start_context = "Українська мова"
    main(config, logger, start_context)
