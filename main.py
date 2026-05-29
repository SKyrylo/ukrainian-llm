# main.py — top-level entry point for the GPT language model training pipeline.
#
# Pipeline overview (in order of execution)
# ------------------------------------------
# 1. Parse the config.yaml file into a SimpleNamespace for dot-notation access.
# 2. Resolve (and optionally create) the output directory.
# 3. Detect the compute device (CUDA GPU or CPU).
# 4. Load all .txt files from the dataset directory and concatenate them with
#    EOS separators into a single flat string.
# 5. Train a new BPE tokenizer on the concatenated text, OR load a previously
#    trained tokenizer from disk.
# 6. Build the MODEL_CONFIG dictionary from the tokenizer and config values.
# 7. Instantiate a GPTModel from scratch OR load weights from a checkpoint file.
# 8. Print a baseline text sample from the untrained / loaded model.
# 9. Encode the full corpus once (loading from disk cache when available),
#    split at the token level, and build DataLoader objects for both splits.
# 10. Run the training loop (train_model) — includes periodic evaluation,
#     qualitative sample generation, early stopping, and best-model saving.
# 11. Save the final model checkpoint to disk.
# 12. Plot and save the train/val loss curves as a PDF.
# 13. Generate 5 text samples from the trained model for final inspection.
import hashlib
import torch
from typing import List

# Import the sliding-window DataLoader factory for the training and val splits
from src.dataloader import create_dataloader
# Import the custom BPE tokenizer (fit, save, load, encode, decode)
from src.tokenizer import BPETokenizer
# Import all generation, loss, training, and plotting utilities
from src.llm_utils import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
    train_model,
    plot_losses
)
# Import the GPT model architecture (all building blocks + top-level model)
from src.llm_parts import GPTModel

# Standard library imports for timing, config parsing, path management, logging
from time import time
import yaml
from types import SimpleNamespace
import os
from pathlib import Path
import logging
from src.logger import setup_logger
# python-dotenv loads environment variables from a .env file if present
from dotenv import load_dotenv
load_dotenv()


# -----------------------------------------------------------------------
# _resolve_output_dir — helper that turns the output_dir config value
# into a concrete Path object (and creates it), or returns None when
# no output_dir is configured.
#
# Google Colab note: when using Google Drive as the output location,
# the Drive must be mounted before running this script by adding a cell:
#     from google.colab import drive
#     drive.mount('/content/drive')
# Then set output_dir in config.yaml to e.g. /content/drive/MyDrive/llm
# -----------------------------------------------------------------------
def _resolve_output_dir(config) -> Path | None:
    """Return the resolved output directory (and create it), or None when
    output_dir is not set.

    When using Google Drive on Colab, mount the drive BEFORE running this
    script by adding a cell at the top of your notebook:
        from google.colab import drive
        drive.mount('/content/drive')
    Then set output_dir in config.yaml to e.g. /content/drive/MyDrive/llm
    """
    # getattr with a default handles configs that omit the key entirely
    raw = getattr(config, "output_dir", "") or ""
    # Empty string means "write everything to the current working directory"
    if not raw:
        return None

    out = Path(raw)
    # Guard: if the path starts with the Colab Drive mount point, verify it exists
    if str(out).startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
        raise RuntimeError(
            "output_dir points to Google Drive but Drive is not mounted.\n"
            "Run this in a Colab notebook cell first:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')"
        )

    # Create the directory (and any missing parent directories) if it doesn't exist
    out.mkdir(parents=True, exist_ok=True)
    return out


# -----------------------------------------------------------------------
# _out — resolve a bare filename to the output_dir, or keep it as a
# local relative path if no output_dir was configured.
# -----------------------------------------------------------------------
def _out(output_dir: Path | None, filename: str) -> Path:
    """Resolve a filename to output_dir, or keep it as a local relative path."""
    if output_dir is not None:
        # Prepend the configured output directory to the filename
        return output_dir / filename
    # No output_dir configured → write to the current working directory
    return Path(filename)


# -----------------------------------------------------------------------
# _encode_incrementally — encode a corpus file-by-file with periodic
# checkpoint saving and automatic resume across Colab sessions.
#
# Why file-by-file instead of one joined string
# ----------------------------------------------
# The previous approach joined all files into one giant string and called
# tokenizer.encode() once.  That meant the entire encoding job had to
# finish in a single Python process run — if Colab killed the session
# after 11 hours, all progress was lost.
#
# This function encodes each file individually, accumulates the token IDs,
# and writes a partial checkpoint to disk every SAVE_EVERY files.  If the
# session is killed, restarting main.py will load the checkpoint and
# continue from the last saved position.  The final token_cache.pt is
# identical to what the old approach would have produced.
#
# Cache invalidation
# ------------------
# The cache key is an MD5 of (filename + file-size for every file, in
# sorted order) concatenated with the tokenizer's vocab size.  Hashing
# file metadata is instant compared to hashing 700 MB of text, and the
# key changes correctly when files are added/removed/replaced or the
# tokenizer is retrained.
#
# Spacing note
# ------------
# The original pipeline joined files with f" {eos} " which produces chunks
# like ["text1 ", "<EOS>", " text2 ", "<EOS>", " text3"].  This function
# recreates the same spacing around each file so that the resulting token
# IDs are bit-for-bit identical to the old single-pass approach.
# -----------------------------------------------------------------------
def _encode_incrementally(
    txt_files,
    eos_token: str,
    tokenizer,
    cache_path: Path,
    logger,
    save_every: int = 200,   # save a checkpoint every this many files
) -> torch.Tensor:
    """Encode *txt_files* one by one, saving checkpoints every *save_every* files.

    Returns the full concatenated token-ID tensor (same result as encoding
    the joined corpus string in one pass).
    """
    # ---- Build the cache key from file metadata (fast — no text hashing) ----
    meta_parts = "|".join(
        f"{f.name}:{f.stat().st_size}" for f in txt_files
    )
    cache_meta = "v2-" + hashlib.md5(
        f"{meta_parts}|vocab={len(tokenizer.str2int)}".encode()
    ).hexdigest()

    # ---- Full cache hit: skip encoding entirely -----------------------------
    if cache_path.exists():
        try:
            cached = torch.load(cache_path, weights_only=True)
            if cached.get("meta") == cache_meta:
                token_ids = cached["token_ids"]
                logger.info(
                    "Loaded %d cached tokens from %s", len(token_ids), cache_path
                )
                return token_ids
            else:
                logger.info("Token cache is stale — re-encoding corpus...")
        except Exception as exc:
            logger.warning("Cache unreadable (%s) — re-encoding...", exc)

    # ---- Check for a partial checkpoint (resume across sessions) ------------
    # The checkpoint lives alongside the final cache so it ends up on Drive
    ckpt_path = cache_path.with_name("encode_checkpoint.pt")
    n = len(txt_files)
    eos_id = tokenizer.str2int.get(eos_token, 0)
    start_idx = 0
    accumulated_ids: List[torch.Tensor] = []

    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, weights_only=True)
            if ckpt.get("meta") == cache_meta and ckpt.get("files_done", 0) > 0:
                accumulated_ids = [ckpt["token_ids"]]
                start_idx = ckpt["files_done"]
                logger.info(
                    "Resuming from checkpoint: %d / %d files already encoded "
                    "(%d tokens so far)",
                    start_idx, n, len(ckpt["token_ids"]),
                )
            else:
                logger.info(
                    "Checkpoint found but does not match current corpus — "
                    "starting fresh"
                )
        except Exception as exc:
            logger.warning("Checkpoint unreadable (%s) — starting fresh", exc)

    if start_idx == 0:
        logger.info(
            "Encoding %d files (checkpoint saved every %d files → %s)...",
            n, save_every, ckpt_path,
        )

    # ---- Encode file by file with a tqdm bar --------------------------------
    from tqdm import tqdm
    with tqdm(
        total=n,
        initial=start_idx,
        desc="Encoding",
        unit="file",
    ) as pbar:
        for i in range(start_idx, n):
            fpath = txt_files[i]

            # Read each file on demand so we never hold the full 700 MB corpus
            # in RAM as a single Python string
            with open(fpath, encoding="utf-8", errors="replace") as fh:
                text = fh.read()

            # Reproduce the spacing that f" {eos_token} ".join(texts) would create
            # so the resulting token IDs match the original single-pass approach:
            #   first  file: "text "        (trailing space before EOS)
            #   middle files: " text "      (both spaces)
            #   last   file: " text"        (leading space after EOS, no trailing)
            #   only   file: "text"         (no spaces, no EOS)
            if n == 1:
                chunk = text
            elif i == 0:
                chunk = text + " "
            elif i == n - 1:
                chunk = " " + text
            else:
                chunk = " " + text + " "

            # Encode this one file's chunk (fast with the heap-based encoder)
            file_ids = tokenizer.encode(chunk)
            accumulated_ids.append(file_ids)

            # Insert the EOS token between every pair of documents
            if i < n - 1:
                accumulated_ids.append(torch.tensor([eos_id]))

            # Update the progress bar with the current filename
            name = fpath.name
            if len(name) > 40:
                name = "…" + name[-39:]
            pbar.set_description(f"Encoding  {name}")
            pbar.update(1)

            # Save a checkpoint every save_every files.  We also concatenate
            # the accumulated tensors into one before saving to keep the
            # checkpoint file small and to release individual tensor memory.
            files_done = i + 1
            if (files_done - start_idx) % save_every == 0:
                partial = torch.cat(accumulated_ids)
                # Replace list with a single tensor so earlier tensors can be GC'd
                accumulated_ids = [partial]
                torch.save(
                    {"meta": cache_meta, "token_ids": partial, "files_done": files_done},
                    ckpt_path,
                )
                logger.info(
                    "  Checkpoint: %d / %d files, %d tokens → %s",
                    files_done, n, len(partial), ckpt_path,
                )

    # ---- All files done: save the final cache and remove the checkpoint -----
    all_token_ids = torch.cat(accumulated_ids)
    torch.save({"meta": cache_meta, "token_ids": all_token_ids}, cache_path)
    logger.info(
        "Encoding complete: %d tokens → %s", len(all_token_ids), cache_path
    )
    if ckpt_path.exists():
        ckpt_path.unlink()

    return all_token_ids


# -----------------------------------------------------------------------
# main — orchestrates the complete training pipeline
# -----------------------------------------------------------------------
# Parameters
# ----------
# config        : SimpleNamespace loaded from config.yaml
# logger        : configured logging.Logger instance for structured output
# start_context : seed text used for qualitative generation samples
# -----------------------------------------------------------------------
def main(
    config,
    logger,
    start_context="Hello, I am"
    ):
    # Step 1 — resolve and create the output directory (or default to cwd)
    output_dir = _resolve_output_dir(config)
    if output_dir:
        logger.info("Output directory: %s", output_dir)
    else:
        logger.info("Output directory: local (output_dir not set)")
    # Step 2 — detect available compute device; prefer CUDA for speed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Step 3 — find all .txt dataset files and sort them for reproducible order
    dataset_dir = Path(config.dataset_dir)
    txt_files = sorted(dataset_dir.glob("*.txt"))
    # Abort early with a clear error if no training data is found
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {dataset_dir.resolve()}")
    logger.info("Found %d dataset file(s) in %s", len(txt_files), dataset_dir.resolve())

    # Step 4 + 5 — tokenizer: train a new one or load an existing one from disk
    #
    # When training a new tokenizer we must read all corpus text up-front.
    # When loading a pre-trained tokenizer we skip that expensive read
    # entirely; the encoding step (_encode_incrementally) will read files
    # on demand one by one, so we never hold the full 700 MB corpus in RAM.
    eos = config.tokenizer['eos_token']

    if config.tokenizer['tok_load_from'] == "":
        # ---- Train a brand-new tokenizer ------------------------------------
        # We need the full corpus as one string for BPE frequency counting
        texts = []
        for fpath in txt_files:
            # errors="replace" handles any stray non-UTF-8 bytes gracefully
            with open(fpath, encoding="utf-8", errors="replace") as f:
                texts.append(f.read())
        # Single flat string: doc1 <EOS> doc2 <EOS> doc3 ...
        text_data = f" {eos} ".join(texts)

        logger.info("Training a new tokenizer on %.1f MB...", len(text_data) / 1e6)
        tokenizer = BPETokenizer()
        # fit() runs the BPE merge loop until vocab_size tokens are learned
        tokenizer.fit(
            text_data,
            vocab_size=config.tokenizer['vocab_size'],
            min_freq=config.tokenizer['min_freq'],
            eos_token=config.tokenizer['eos_token'],
            unk_token=config.tokenizer['unk_token']
        )

        # Optionally persist the tokenizer so future runs can skip re-training
        if config.tokenizer['tok_save_path'] != "":
            tok_save = _out(output_dir, config.tokenizer['tok_save_path'])
            # Ensure the parent directory of the save path exists
            tok_save.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(str(tok_save))
            logger.info("Tokenizer trained and saved to %s", tok_save)
    else:
        # ---- Load a pre-trained tokenizer (no corpus read needed here) ------
        tok_load = config.tokenizer['tok_load_from']
        logger.info("Loading existing tokenizer from %s", tok_load)
        tokenizer = BPETokenizer(tok_load)
    
    # Step 6 — build the model configuration dictionary
    # All hyperparameters come from config.yaml so they can be changed without
    # touching the source code.  vocab_size is derived from the tokenizer so it
    # always stays in sync with the actual vocabulary.
    MODEL_CONFIG = {
        "vocab_size": len(tokenizer.str2int),                   # Vocabulary size
        "context_length": config.dataloader['context_length'],  # Context length
        "emb_dim": config.llm_init['emb_dim'],                  # Embedding dimension
        "num_heads": config.llm_init['n_heads'],                # Number of attention heads
        "num_layers": config.llm_init['n_layers'],              # Number of layers
        "drop_rate": config.llm_init['dropout'],                # Dropout rate
        "qkv_bias": config.llm_init['qkv_bias']                 # Query-Key-Value bias
    }

    # Step 7 — initialise the model, either from scratch or from a checkpoint
    if config.llm_train['from_pretrained']:
        # Resume training from a previously saved checkpoint
        logger.info("Loading pretrained model from %s", config.llm_train['from_pretrained'])
        
        # weights_only=True prevents arbitrary code execution during unpickling
        checkpoint = torch.load(
            config.llm_train['from_pretrained'],
            weights_only=True
        )

        # Instantiate model architecture and load the saved parameter tensors
        model = GPTModel(MODEL_CONFIG)
        model.load_state_dict(checkpoint["model_state_dict"])
        # Restore the AdamW optimizer and its momentum/variance state so that
        # the optimiser continues smoothly from where training was interrupted
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer['learning_rate'],
            weight_decay=config.optimizer['weight_decay']
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        # Fresh start: create a model with randomly initialised weights
        logger.info("Initializing a new model with configuration: %s", MODEL_CONFIG)

        model = GPTModel(MODEL_CONFIG)
        # AdamW is the de-facto standard optimiser for transformer training;
        # it combines Adam momentum with decoupled weight decay regularisation
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer['learning_rate'],
            weight_decay=config.optimizer['weight_decay']
        )
    
    # Move all model parameters and buffers to the target device (CPU or CUDA)
    model.to(device)
    
    # Count and log the total number of trainable parameters for reference
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Model configuration: {MODEL_CONFIG}")
    logger.info(f"Model total parameters (all trainable): {total_params}")
    # Assumes 32-bit (4-byte) float parameters; gives a rough memory estimate
    logger.info(f"Total size of the model: {total_params * 4 / (1024 ** 2):.2f} MB (assuming 4 bytes per parameter)")
    
    # Step 8 — generate a baseline sample BEFORE training begins
    # This shows what the model outputs with random (or loaded) weights,
    # giving a qualitative reference point for how much training improves output
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

    # Step 9 — encode the full corpus once, cache the result to disk, and
    # split at the token level into training and validation sets.
    #
    # WHY encode once: encoding 700 MB of Ukrainian text against 31 K BPE
    # merge rules is O(corpus × log corpus) with the new heap-based encoder,
    # typically finishing in under an hour.  The result is cached as
    # token_cache.pt so every subsequent run is instant (milliseconds).
    #
    # WHY split at the token level: splitting the raw text string at a
    # character offset and encoding each half independently would double
    # the encoding work.  Splitting the token tensor is O(1) and gives
    # an exact train/val ratio at the token level.
    #
    # Checkpoint safety: if the Colab session is killed mid-encoding,
    # encode_checkpoint.pt in the same output directory records how many
    # files were done.  Re-running main.py picks up automatically from
    # the last checkpoint — no work is lost.
    # If token_cache_from is set in config, load that file directly and skip
    # encoding entirely.  Use this on Kaggle (or any run) when you have a
    # pre-built cache from a different machine (e.g. Colab) whose file list
    # differs from the current dataset_dir, so the automatic cache-key check
    # would incorrectly report the cache as stale.
    prebuilt_cache = getattr(config, 'token_cache_from', '') or ''
    if prebuilt_cache:
        logger.info("Loading pre-built token cache from %s", prebuilt_cache)
        _loaded = torch.load(prebuilt_cache, weights_only=True)
        # Support both the {meta, token_ids} dict format and a bare tensor
        all_token_ids = (
            _loaded["token_ids"] if isinstance(_loaded, dict) else _loaded
        )
        logger.info("Loaded %d tokens from pre-built cache", len(all_token_ids))
    else:
        cache_path = _out(output_dir, "token_cache.pt")
        all_token_ids = _encode_incrementally(
            txt_files, eos, tokenizer, cache_path, logger,
        )

    # Compute the split index in token space
    train_ratio = config.llm_train['train_val_split']
    split_token_idx = int(train_ratio * len(all_token_ids))
    # Training slice: first train_ratio fraction of the encoded corpus
    train_token_ids = all_token_ids[:split_token_idx]
    # Validation slice: remaining tokens
    val_token_ids   = all_token_ids[split_token_idx:]
    logger.info(
        "Token split: %d train / %d val  (%.0f%% / %.0f%%)",
        len(train_token_ids), len(val_token_ids),
        train_ratio * 100, (1 - train_ratio) * 100,
    )

    # encode_only mode: exit here once the cache has been written to disk.
    # Use this on a CPU session to pay the one-time encoding cost and persist
    # token_cache.pt to output_dir without running the (GPU-intensive) training.
    # Set encode_only: false (or omit the key) in config.yaml for training runs.
    if getattr(config, 'encode_only', False):
        logger.info(
            "encode_only=true — token cache saved to %s. Exiting.", cache_path
        )
        return

    # Build DataLoaders from the pre-encoded token tensors — no encoding here
    train_loader = create_dataloader(
        train_token_ids,
        batch_size=config.dataloader['batch_size'],
        context_length=MODEL_CONFIG["context_length"],
        stride=config.dataloader['stride'],
        drop_last=config.dataloader['drop_last'],
        shuffle=config.dataloader['shuffle'],
        num_workers=config.dataloader['num_workers'],
    )

    # Validation loader uses stride == context_length (no overlap between windows)
    # to avoid evaluating the same tokens twice, and no shuffling to keep order
    val_loader = create_dataloader(
        val_token_ids,
        batch_size=config.dataloader['batch_size'],
        context_length=MODEL_CONFIG["context_length"],
        # Validation stride equals context_length so windows tile without overlap
        stride=MODEL_CONFIG["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=config.dataloader['num_workers'],
    )

    logger.info(
        "Dataloaders ready: %d train batches, %d val batches",
        len(train_loader), len(val_loader),
    )

    # Step 10 — run the training loop and record elapsed wall-clock time
    start_time = time()

    num_epochs = config.llm_train['epochs']
    # train_model returns loss histories and a token-count list for plotting
    train_losses, val_losses, tokens_seen = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=config.llm_train.get('eval_freq', 500),
        eval_iter=config.llm_train.get('eval_iter', 20),
        # start_context for sample generation printed at end of each epoch
        start_context="Українська мова",
        tokenizer=tokenizer,
        temperature=config.llm_gen['temperature'],
        top_k=config.llm_gen['top_k'],
        # patience=0 disables early stopping; any positive int enables it
        patience=config.llm_train.get('patience', 3),
        # save_best_path=None disables best-model checkpointing
        save_best_path=str(_out(output_dir, config.llm_train['model_save_path']))
            if config.llm_train.get('model_save_path') else None,
    )
    # Switch back to eval mode after the training loop finishes
    model.eval()

    end_time = time()

    # Step 11 — save the final model checkpoint
    # This saves the model after all epochs (which may not be the best checkpoint
    # if early stopping fired; the best checkpoint was already saved during training)
    if config.llm_train['model_save_path']:
        final_path = _out(output_dir, config.llm_train['model_save_path'])
        # Ensure the checkpoint directory exists before writing
        final_path.parent.mkdir(parents=True, exist_ok=True)
        # Save both model weights and optimizer state for full resumability
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            final_path,
        )
        logger.info("Final model saved to %s", final_path)

    # Report total training time in minutes for convenience
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Step 12 — plot and save the loss curves
    # linspace maps checkpoint indices to fractional epoch positions so the
    # x-axis scale is accurate even when eval_freq doesn't divide num_epochs evenly
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_path = str(_out(output_dir, "loss-plot.pdf"))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, plot_path)

    # Step 13 — generate 5 diverse text samples from the trained model
    # These provide a qualitative assessment of the model's language quality
    for i in range(5):
        print(f"\nSample {i+1}:")
        # Each call to generate is independent; temperature and top_k introduce
        # diversity so the five samples should not be identical
        out = generate(
            model=model,
            idx=text_to_token_ids(start_context, tokenizer).to(device),
            max_new_tokens=config.llm_gen['max_gen_length'],
            context_size=MODEL_CONFIG["context_length"],
            top_k=config.llm_gen['top_k'],
            temperature=config.llm_gen['temperature']
            )

        print(token_ids_to_text(out, tokenizer))


# -----------------------------------------------------------------------
# Script entry point — runs when this file is executed directly.
# The logger is set to DEBUG at the root level so that all child loggers
# in src/* can propagate messages without being filtered.
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Set the root logger level to DEBUG so module-level loggers are not suppressed
    logging.getLogger().setLevel(logging.DEBUG)
    # Get a named logger for this module using the shared setup from src/logger.py
    logger = setup_logger(__name__)

    # Load the YAML config file and convert the resulting dict to a SimpleNamespace
    # so that config values can be accessed with dot notation (e.g. config.llm_init)
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        config = SimpleNamespace(**config)
    
    # Seed text used for all qualitative generation samples during and after training
    start_context = "Українська мова"
    main(config, logger, start_context)
