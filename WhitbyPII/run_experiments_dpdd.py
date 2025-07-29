# run_experiments_dpdd.py
import torch
import os
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import get_scheduler
from dpdd.monitor_dpdd import DPDDMonitor
from tqdm import tqdm

#Simple Dataset Wrapper
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.texts = dataframe["text"].astype(str).tolist()
        self.examples = tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    def __len__(self):
        return self.examples["input_ids"].size(0)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.examples.items()}

#Save HuggingFace-Compatible Format
def save_huggingface_format(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[HF Save] Model and tokenizer saved to {output_dir}")
    except AttributeError:
        print(f"[HF Save] PEFT model detected — saving with peft.save_pretrained")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

#Training Loop
def train_loop(
    model,
    train_dataloader,
    val_dataloader,
    tokenizer,
    num_epochs=2,
    save_path="checkpoints",
    dpdd_monitor=None,
    learning_rate=1e-5,
    device=torch.device("cuda:0")
):
    model.train()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(train_dataloader)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Initialize DPDD Monitor
    dpdd_monitor = dpdd_monitor if dpdd_monitor else DPDDMonitor()
    dropped_indices_log = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        loop = tqdm(train_dataloader, desc="Training")

        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            loss.backward()

            # DPDD gradient monitoring
            if dpdd_monitor:
                dpdd_monitor.record_gradients(model)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())

        # DPDD score + sample filtering
        if dpdd_monitor:
            dpdd_score = dpdd_monitor.compute_dpdd_score()
            print(f"[DPDD] Score after epoch {epoch+1}: {dpdd_score:.4f}")

            if hasattr(train_dataloader.dataset, 'texts'):
                keep_indices = dpdd_monitor.get_non_dropped_indices(threshold=0.1, total=len(train_dataloader.dataset))
                dropped = sorted(set(range(len(train_dataloader.dataset))) - set(keep_indices))
                dropped_indices_log.append(dropped)
                print(f"[DPDD] Dropped {len(dropped)} samples for next epoch.")

                new_dataset = Subset(train_dataloader.dataset, keep_indices)
                train_dataloader = DataLoader(new_dataset, batch_size=train_dataloader.batch_size, shuffle=True)

            dpdd_monitor.reset()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                val_outputs = model(**val_batch, labels=val_batch["input_ids"])
                val_loss += val_outputs.loss.item()
        val_loss /= len(val_dataloader)
        print(f"[Validation] Loss after epoch {epoch+1}: {val_loss:.4f}")
        model.train()

        # Save checkpoint (LoRA or standard)
        os.makedirs(save_path, exist_ok=True)
        checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch+1}")
        try:
            model.save_pretrained(checkpoint_path)
            print(f"[Checkpoint] LoRA model saved to {checkpoint_path}")
        except AttributeError:
            torch.save(model.state_dict(), f"{checkpoint_path}.pt")
            print(f"[Checkpoint] Model state_dict saved to {checkpoint_path}.pt")

    # Final HuggingFace-format save
    hf_output_dir = os.path.join(save_path, "hf_format")
    save_huggingface_format(model, tokenizer, hf_output_dir)

    print("Training complete.")
    return model
