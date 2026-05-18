# llm_utils.py — utility functions for text generation, loss computation,
# model training, and loss-curve plotting.
#
# Public API summary:
#   generate_text            — greedy next-token generation (no temperature/top-k)
#   generate                 — full generation with temperature scaling and top-k filtering
#   text_to_token_ids        — convert a raw string to a batched token-ID tensor
#   token_ids_to_text        — convert a token-ID tensor back to a string
#   calc_loss_batch          — cross-entropy loss for a single mini-batch
#   calc_loss_loader         — average loss over N batches of a DataLoader
#   evaluate_model           — compute train and validation loss without gradient updates
#   generate_and_print_sample— generate a short text sample and print it during training
#   train_model              — main training loop with evaluation, early stopping,
#                              and best-checkpoint saving
#   plot_losses              — render and save the train/val loss curves as a PDF
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# -----------------------------------------------------------------------
# generate_text — simple greedy (argmax) decoding
# -----------------------------------------------------------------------
# This is the baseline generation function that always picks the single
# most probable next token at each step.  It is deterministic: given the
# same input and model weights it always produces the same output.
#
# Parameters
# ----------
# model        : trained GPTModel (or any model returning (B, T, vocab) logits)
# idx          : (batch, n_tokens) tensor of current context token IDs
# max_new_tokens: number of additional tokens to generate
# context_size : maximum context the model was trained on; older tokens are
#               dropped when the running sequence exceeds this length
def generate_text(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


# -----------------------------------------------------------------------
# generate — full generation with temperature scaling and top-k filtering
# -----------------------------------------------------------------------
# Extends generate_text with two sampling controls that make the output
# more diverse and coherent:
#
# top_k : Before sampling, keep only the top-k highest logit values and
#         set the rest to -inf.  This prevents the model from sampling
#         very low-probability "garbage" tokens while still allowing
#         diversity among the most likely candidates.
#
# temperature : Divide logits by this scalar before softmax.
#   - temperature < 1.0 → distribution is sharper (more greedy / confident)
#   - temperature > 1.0 → distribution is flatter (more diverse / creative)
#   - temperature = 0.0 → falls back to pure argmax (same as generate_text)
#
# eos_id : If provided, generation stops early when this token ID is sampled.
#          Useful when the model has been trained with an EOS token.
def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None
    ):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        # Truncate to the model's context window (drop old tokens if needed)
        idx_cond = idx[:, -context_size:]
        # Forward pass — gradients not needed during generation
        with torch.no_grad():
            logits = model(idx_cond)
        # Select the logit vector for the last (most recent) token position
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            # Identify the threshold: the smallest logit among the top-k
            min_val = top_logits[:, -1]
            # Zero out all logits below the threshold by setting them to -inf
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            # Dividing by temperature before softmax sharpens or flattens the distribution
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            # multinomial draws one token index according to the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            # Pure greedy decoding when temperature == 0
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if eos_id is not None and idx_next.item() == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


# -----------------------------------------------------------------------
# text_to_token_ids / token_ids_to_text — conversion helpers
# -----------------------------------------------------------------------
# These two functions act as the interface between human-readable strings
# and the integer tensor representation used by the model.

# Encode a plain-text string into a (1, n_tokens) tensor so it can be
# fed directly to the model as a batch of size 1.
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    # Guard: some tokenizer implementations may return a plain list
    if not isinstance(encoded, torch.Tensor):
        encoded = torch.tensor(encoded)
    return encoded.unsqueeze(0)  # add batch dimension -> (1, n_tokens)


# Decode a (1, n_tokens) or (n_tokens,) token-ID tensor back to a string.
# The batch dimension is removed with squeeze before calling the tokenizer.
def token_ids_to_text(token_ids, tokenizer):
    # Remove the batch dimension: (1, n_tokens) → (n_tokens,)
    flat = token_ids.squeeze(0)
    # Convert tensor to a plain Python list before passing to the tokenizer
    return tokenizer.decode(flat.tolist())


# -----------------------------------------------------------------------
# calc_loss_batch — loss for a single (input, target) mini-batch
# -----------------------------------------------------------------------
# Moves both batches to the target device, runs a forward pass, and
# computes the mean cross-entropy loss over all token positions.
#
# The logits tensor of shape (B, T, V) is flattened to (B*T, V) and the
# targets of shape (B, T) are flattened to (B*T,) before cross_entropy,
# because PyTorch's cross_entropy expects 2-D logits and 1-D targets.
def calc_loss_batch(input_batch, target_batch, model, device):
    # Move data to the device where the model lives (CPU or CUDA)
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # Forward pass through the model to obtain logits
    logits = model(input_batch)
    # Flatten (B, T, V) → (B*T, V) and (B, T) → (B*T,) for cross_entropy
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


# -----------------------------------------------------------------------
# calc_loss_loader — average loss over multiple batches
# -----------------------------------------------------------------------
# Iterates over a DataLoader for up to num_batches batches and returns
# the arithmetic mean of the per-batch cross-entropy losses.
#
# num_batches controls how many batches are evaluated:
#   None  → evaluate every batch in the loader (exact loss)
#   int N → evaluate only the first N batches (fast approximation)
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    # Guard: return NaN if the data loader contains no batches at all
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        # Default: use all available batches in the loader
        num_batches = len(data_loader)
    else:
        # Reduce the numer of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    
    # Accumulate loss over the first num_batches batches
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    
    # Return the mean loss (sum divided by number of batches evaluated)
    return total_loss / num_batches


# -----------------------------------------------------------------------
# evaluate_model — compute train/val loss without updating weights
# -----------------------------------------------------------------------
# Switches the model to eval mode (disables dropout), computes losses on
# both loaders using eval_iter batches each, then restores training mode.
#
# Using a limited eval_iter rather than the full dataset gives a fast
# estimate of generalisation performance during the training loop.
def evaluate_model(
    model,
    train_loader,
    val_loader,
    device,
    eval_iter
    ):
    # Disable dropout layers so evaluation is deterministic
    model.eval()

    # torch.no_grad() avoids storing intermediate activations, saving memory
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    
    # Re-enable dropout for continued training after evaluation
    model.train()
    
    return train_loss, val_loss


# -----------------------------------------------------------------------
# generate_and_print_sample — qualitative check during training
# -----------------------------------------------------------------------
# Called at the end of each epoch to give a human-readable indication of
# how the model is progressing.  The model is switched to eval mode for
# generation and restored to train mode afterwards.
def generate_and_print_sample(
    model,
    tokenizer,
    device,
    start_context,
    temperature=0.0,
    top_k=None
    ):
    # Switch to eval mode to disable dropout during generation
    model.eval()

    # Derive the context window size from the positional embedding table shape
    context_size = model.pos_emb.weight.shape[0]
    # Encode the starting text and move to the target device
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        # Generate 50 new tokens following the start_context prompt
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            temperature=temperature,
            top_k=top_k
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " ")) # Compact print format
    # Restore training mode (re-enables dropout)
    model.train()


# -----------------------------------------------------------------------
# train_model — main training loop
# -----------------------------------------------------------------------
# Runs the standard supervised training loop for num_epochs epochs.
# Every eval_freq gradient steps it evaluates the model and records losses.
# Supports:
#   - Early stopping via `patience` (stops when val_loss stops improving)
#   - Best-checkpoint saving via `save_best_path`
#   - A qualitative text sample printed at the end of each epoch
#
# Returns three lists:
#   train_losses      — training loss recorded at each eval checkpoint
#   val_losses        — validation loss recorded at each eval checkpoint
#   track_tokens_seen — total tokens processed at each eval checkpoint
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    temperature=0.0,
    top_k=None,
    patience=3,
    save_best_path=None,
    ):
    """Train the model with optional early stopping and best-model checkpointing.

    Args:
        patience:        Stop if val_loss has not improved for this many consecutive
                         evaluation steps. Set to 0 to disable early stopping.
        save_best_path:  If given, save the model state whenever val_loss improves.
                         Pass the same path as model_save_path to always keep the best
                         checkpoint on disk (safe for overnight runs).
    """
    # Lists to collect metrics for plotting and inspection after training
    train_losses, val_losses, track_tokens_seen = [], [], []
    # tokens_seen tracks the total number of tokens processed across all steps
    # global_step counts every gradient update regardless of epoch boundaries
    tokens_seen, global_step = 0, -1
    # best_val_loss tracks the lowest validation loss seen so far for early stopping
    best_val_loss = float("inf")
    # patience_counter counts consecutive eval steps without improvement
    patience_counter = 0
    # stop_training flag allows breaking out of the epoch loop from within the batch loop
    stop_training = False

    for epoch in range(num_epochs):
        # Ensure dropout is active at the start of each epoch
        model.train()

        for input_batch, target_batch in train_loader:
            # Zero gradients before computing the new gradient for this batch
            optimizer.zero_grad()
            # Compute the cross-entropy loss for the current mini-batch
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # Backpropagate: compute gradients of the loss w.r.t. all parameters
            loss.backward()
            # Apply the AdamW parameter update using the computed gradients
            optimizer.step()
            # Track total tokens seen: input_batch.numel() = batch_size * context_length
            tokens_seen += input_batch.numel()
            global_step += 1

            # Periodic evaluation: check train/val loss every eval_freq steps
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                # Record losses and token count for plotting
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Step {global_step}, "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                # Early stopping and best-model checkpointing logic
                if val_loss < best_val_loss:
                    # New best: reset the patience counter and save a checkpoint
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_best_path:
                        import torch, os
                        from pathlib import Path
                        # Create the directory for the checkpoint if it doesn't exist
                        os.makedirs(Path(save_best_path).parent, exist_ok=True)
                        # Save both the model weights and optimizer state so training
                        # can be resumed from this exact checkpoint if needed
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                            },
                            save_best_path,
                        )
                        print(f"  ✓ New best val_loss={best_val_loss:.4f} — checkpoint saved.")
                else:
                    # No improvement: increment the patience counter
                    patience_counter += 1
                    # If the counter has reached the patience threshold, stop training
                    if patience > 0 and patience_counter >= patience:
                        print(
                            f"Early stopping: val_loss has not improved for "
                            f"{patience} eval steps (best={best_val_loss:.4f})."
                        )
                        stop_training = True
                        break

        # At the end of each epoch, generate and print a short text sample
        # to give a qualitative sense of how training is progressing
        generate_and_print_sample(model, tokenizer, device, start_context, temperature, top_k)

        # Break the outer epoch loop if early stopping was triggered inside the batch loop
        if stop_training:
            break

    return train_losses, val_losses, track_tokens_seen


# -----------------------------------------------------------------------
# plot_losses — visualise training and validation loss curves
# -----------------------------------------------------------------------
# Creates a figure with two x-axes:
#   - Primary (bottom): number of training epochs
#   - Secondary (top):  total tokens seen during training
# Both axes share the same y-axis (loss value).
# The resulting plot is saved to `save_path` (default: loss-plot.pdf).
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, save_path="loss-plot.pdf"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training loss as a solid line
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    # Plot validation loss as a dash-dot line to distinguish it visually
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    # Force epoch tick marks to be integers (no fractional epoch labels)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Create a twin x-axis at the top of the plot scaled to tokens seen
    ax2 = ax1.twiny()
    # Plot with alpha=0 so the line is invisible; only the axis labels are shown
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    # Tighten margins to avoid clipping axis labels
    fig.tight_layout()
    # Persist the figure to disk as a PDF for high-quality vector output
    plt.savefig(save_path)
    plt.show()


# -----------------------------------------------------------------------
# Smoke-test / debugging script — run this file directly to test that
# forward passes, loss calculations, and generation work end-to-end.
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.tokenizer import BPETokenizer
    from src.llm_parts import GPTModel

    # start_context = "Every effort moves you"
    # Load a pre-trained tokenizer to set the correct vocabulary size
    tokenizer = BPETokenizer(vocab_file="dataset/tokenizer.json")

    # Configuration matching a ~124 M parameter GPT-2-scale model
    GPT_CONFIG_124M = {
        'vocab_size': len(tokenizer.str2int),  # must match tokenizer vocab
        'context_length': 1024,
        "emb_dim": 768,
        'num_heads': 12,
        "num_layers": 12,
        'drop_rate': 0.1,
        'qkv_bias': False
    }

    # Instantiate an untrained model with random weights
    model = GPTModel(GPT_CONFIG_124M)

    # token_ids = generate_text(
    #     model=model,
    #     idx=text_to_token_ids(start_context, tokenizer),
    #     max_new_tokens=10,
    #     context_size=GPT_CONFIG_124M['context_length']
    # )

    # print(f"Output text: \n{token_ids_to_text(token_ids, tokenizer)}")

    # Hardcoded token-ID batches used to test loss computation manually
    inputs = torch.tensor([
        [88, 3, 116, 236],   # 'an acting '
        [299, 74, 3, 226]    # 'Buy me '
    ])
    targets = torch.tensor([
        [3, 116, 236, 300],  # ' acting Jack'
        [74, 3, 226, 351]    # 'y me Mrs. Gisburn'
    ])

    with torch.no_grad():
        logits = model(inputs)
    
    probs = torch.softmax(logits, dim=-1) # Probability of each token in the vocabulary
    print(probs.shape)

    # Argmax over the vocab dimension gives the most probable token at each position
    token_ids = torch.argmax(probs, dim=-1, keepdim=True)
    print(f"Token IDs: {token_ids}")

    # Decode both targets and model predictions to compare textual output
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

    text_idx = 0
    # Extract the model's assigned probability for each correct target token (batch 1)
    target_probs_1 = probs[text_idx, [0, 1, 2, 3], targets[text_idx]]
    print(f"Text 1: {target_probs_1}")

    text_idx = 1
    # Extract the model's assigned probability for each correct target token (batch 2)
    target_probs_2 = probs[text_idx, [0, 1, 2, 3], targets[text_idx]]
    print(f"Text 2: {target_probs_2}")

    # Compute logarithm of all token probabilities
    # Log probabilities are summed (rather than multiplied) for numerical stability
    log_probs = torch.log(torch.cat([target_probs_1, target_probs_2]))
    print(f"Log probabilities: {log_probs}")

    # Calculate the average probability for each token
    # The negative mean log-probability is equivalent to cross-entropy loss
    avg_log_probs = torch.mean(log_probs)
    print(f"Average log probability: {avg_log_probs}")

    # Flatten tensors for PyTorch's cross_entropy (expects 2-D logits, 1-D targets)
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()

    # This should match -avg_log_probs (up to floating-point rounding)
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(f"Loss: {loss}")
