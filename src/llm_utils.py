import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


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
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if eos_id is not None and idx_next.item() == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    if not isinstance(encoded, torch.Tensor):
        encoded = torch.tensor(encoded)
    return encoded.unsqueeze(0)  # add batch dimension -> (1, n_tokens)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the numer of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    
    return total_loss / num_batches


def evaluate_model(
    model,
    train_loader,
    val_loader,
    device,
    eval_iter
    ):
    model.eval()

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    
    model.train()
    
    return train_loss, val_loss


def generate_and_print_sample(
    model,
    tokenizer,
    device,
    start_context,
    temperature=0.0,
    top_k=None
    ):
    model.eval()

    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
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
    model.train()


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
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    best_val_loss = float("inf")
    patience_counter = 0
    stop_training = False

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Step {global_step}, "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_best_path:
                        import torch, os
                        from pathlib import Path
                        os.makedirs(Path(save_best_path).parent, exist_ok=True)
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                            },
                            save_best_path,
                        )
                        print(f"  ✓ New best val_loss={best_val_loss:.4f} — checkpoint saved.")
                else:
                    patience_counter += 1
                    if patience > 0 and patience_counter >= patience:
                        print(
                            f"Early stopping: val_loss has not improved for "
                            f"{patience} eval steps (best={best_val_loss:.4f})."
                        )
                        stop_training = True
                        break

        generate_and_print_sample(model, tokenizer, device, start_context, temperature, top_k)

        if stop_training:
            break

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, save_path="loss-plot.pdf"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.tokenizer import BPETokenizer
    from src.llm_parts import GPTModel

    # start_context = "Every effort moves you"
    tokenizer = BPETokenizer(vocab_file="dataset/tokenizer.json")

    GPT_CONFIG_124M = {
        'vocab_size': len(tokenizer.str2int),  # must match tokenizer vocab
        'context_length': 1024,
        "emb_dim": 768,
        'num_heads': 12,
        "num_layers": 12,
        'drop_rate': 0.1,
        'qkv_bias': False
    }

    model = GPTModel(GPT_CONFIG_124M)

    # token_ids = generate_text(
    #     model=model,
    #     idx=text_to_token_ids(start_context, tokenizer),
    #     max_new_tokens=10,
    #     context_size=GPT_CONFIG_124M['context_length']
    # )

    # print(f"Output text: \n{token_ids_to_text(token_ids, tokenizer)}")

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

    token_ids = torch.argmax(probs, dim=-1, keepdim=True)
    print(f"Token IDs: {token_ids}")

    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

    text_idx = 0
    target_probs_1 = probs[text_idx, [0, 1, 2, 3], targets[text_idx]]
    print(f"Text 1: {target_probs_1}")

    text_idx = 1
    target_probs_2 = probs[text_idx, [0, 1, 2, 3], targets[text_idx]]
    print(f"Text 2: {target_probs_2}")

    # Compute logarithm of all token probabilities
    log_probs = torch.log(torch.cat([target_probs_1, target_probs_2]))
    print(f"Log probabilities: {log_probs}")

    # Calculate the average probability for each token
    avg_log_probs = torch.mean(log_probs)
    print(f"Average log probability: {avg_log_probs}")

    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(f"Loss: {loss}")
