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
# 9. Split the text into train and validation sets and build DataLoader objects.
# 10. Evaluate pre-training losses on both splits.
# 11. Run the training loop (train_model) — includes periodic evaluation,
#     qualitative sample generation, early stopping, and best-model saving.
# 12. Save the final model checkpoint to disk.
# 13. Plot and save the train/val loss curves as a PDF.
# 14. Generate 5 text samples from the trained model for final inspection.
import torch

# Import the sliding-window DataLoader factory for the training and val splits
from src.dataloader import create_dataloader
# Import the custom BPE tokenizer (fit, save, load, encode, decode)
from src.tokenizer import BPETokenizer
# Import all generation, loss, training, and plotting utilities
from src.llm_utils import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
    calc_loss_batch,
    calc_loss_loader,
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

    # Step 4 — load all text files and join them with EOS separators
    # The EOS token between documents prevents the model from learning to
    # predict across document boundaries as if they were continuous text
    eos = config.tokenizer['eos_token']
    texts = []
    for fpath in txt_files:
        # errors="replace" handles any stray non-UTF-8 bytes gracefully
        with open(fpath, encoding="utf-8", errors="replace") as f:
            texts.append(f.read())
    # Single flat string: doc1 <EOS> doc2 <EOS> doc3 ...
    text_data = f" {eos} ".join(texts)

    # Step 5 — tokenizer: train a new one or load an existing one from disk
    if config.tokenizer['tok_load_from'] == "":
        # No pre-existing tokenizer — fit a new BPE tokenizer on the full corpus
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
        # Load an existing tokenizer from the path specified in config
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

    # Train/validation split
    # train_ratio controls what fraction of the corpus is used for training;
    # the remainder becomes the validation set
    train_ratio = config.llm_train['train_val_split']
    split_idx = int(train_ratio * len(text_data))
    # Training data: the first (train_ratio * 100)% of the corpus
    train_data = text_data[:split_idx]
    # Validation data: the remaining portion of the corpus
    val_data   = text_data[split_idx:]

    # Step 9 — create DataLoader objects for training and validation
    # Training loader uses random shuffling and drops the last incomplete batch
    # to guarantee uniform batch sizes during gradient updates
    train_loader = create_dataloader(
        train_data,
        tokenizer,
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
        val_data,
        tokenizer,
        batch_size=config.dataloader['batch_size'],
        context_length=MODEL_CONFIG["context_length"],
        # Validation stride equals context_length so windows tile exactly
        stride=MODEL_CONFIG["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=config.dataloader['num_workers'],
    )

    # Print one batch shape to verify that the loader is producing the expected sizes
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)
        break

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)
        break

    # Print the number of batches in each loader for reference
    print(len(train_loader))
    print(len(val_loader))

    # Compute and report the approximate token counts in each split
    context_length = MODEL_CONFIG["context_length"]
    # dataset length × context_length gives the total tokens represented in the loader
    train_tokens = len(train_loader.dataset) * context_length
    val_tokens = len(val_loader.dataset) * context_length

    print("Training tokens:", train_tokens)
    print("Validation tokens:", val_tokens)
    print("All tokens:", train_tokens + val_tokens)

    # Step 10 — evaluate pre-training loss on 5 batches from each split
    # These values provide a baseline to compare against post-training losses
    model.eval()
    with torch.no_grad():
        # num_batches=5 gives a quick estimate; increase for a more exact measurement
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    model.train()

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    # Step 11 — run the training loop and record elapsed wall-clock time
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

    # Step 12 — save the final model checkpoint
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

    # Step 13 — plot and save the loss curves
    # linspace maps checkpoint indices to fractional epoch positions so the
    # x-axis scale is accurate even when eval_freq doesn't divide num_epochs evenly
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_path = str(_out(output_dir, "loss-plot.pdf"))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, plot_path)

    # Step 14 — generate 5 diverse text samples from the trained model
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
