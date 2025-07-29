import os
import json
import random
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils.trainer import train_loop, TextDataset
from dpdd.monitor_dpdd import DPDDMonitor
from peft import get_peft_model, LoraConfig, TaskType  # Requires peft installed

from transformers.utils import logging
logging.set_verbosity_error()

# ---- Paths and Constants ---- #
EXPERIMENT_CSV = "data/factorial_experiments.csv"
DATA_PATH = "data/train_with_canaries_cleaned.csv"
MAX_LENGTH = 512
BATCH_SIZE = 2
REPLICATIONS = 2
NUM_EPOCHS = 2
MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Tokenizer ---- #
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=True, trust_remote_code=True)

# Fix pad token issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# ---- Load Full Dataset ---- #
df_full = pd.read_csv(DATA_PATH, sep="\t", on_bad_lines='skip')

def get_filtered_data(year_group):
    df = df_full.copy()
    if year_group == "pre-2018":
        df = df[df["year"] < 2018]
    elif year_group == "2018-2022":
        df = df[(df["year"] >= 2018) & (df["year"] <= 2022)]
    else:
        df = df[df["year"] > 2022]

    return df

def apply_lora(model, r):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

# ---- Load Experiment Grid ---- #
configs = pd.read_csv(EXPERIMENT_CSV)
random.seed(42)

# ---- Run Experiments ---- #
for idx, row in configs.iterrows():
    print(f"\n🔁 Running config {idx+1}/{len(configs)}")

    lora_rank = row['lora_rank']
    learning_rate = row['learning_rate']
    retrieval_docs = row['retrieval_docs']
    elastic_threshold = row['es_threshold']
    dpdd_on = row['dpdd']
    year_group = row['year_group']

    for rep in range(1, REPLICATIONS + 1):
        print(f"→ Replication {rep}/{REPLICATIONS}")

        # Subset + split
        df = get_filtered_data(year_group)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=rep)

        # Tokenize
        train_dataset = TextDataset(train_df, tokenizer, MAX_LENGTH)
        val_dataset = TextDataset(val_df, tokenizer, MAX_LENGTH)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=True, trust_remote_code=True)

        # Apply LoRA
        if lora_rank > 0:
            model = apply_lora(model, r=lora_rank)

        # Apply DPDD
        dpdd_monitor = DPDDMonitor() if dpdd_on else None

        # Save folder
        exp_id = f"lr{learning_rate}_r{lora_rank}_doc{retrieval_docs}_th{elastic_threshold}_yr{year_group}_rep{rep}"
        save_dir = os.path.join("checkpoints", exp_id)
        os.makedirs(save_dir, exist_ok=True)

        # Train
        model = train_loop(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            tokenizer=tokenizer,
            num_epochs=NUM_EPOCHS,
            save_path=save_dir,
            dpdd_monitor=dpdd_monitor,
            learning_rate=learning_rate,
            device=DEVICE
        )

        # ---- Generate Summaries for Validation Set ---- #
        model.eval()
        generated = []

        for _, row in val_df.iterrows():
            if not isinstance(row["text"], str):
                continue  # skip if text is not a valid string
            prompt = row["text"][:MAX_LENGTH]
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).input_ids.to(DEVICE)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=100,
                    do_sample=False,
                    num_beams=4,
                    eos_token_id=tokenizer.eos_token_id
                )
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

            generated.append({
                "text": row["text"],
                "reference": row["category"],
                "generated_summary": decoded,
                "config": exp_id
            })

        # ---- Save Generated Summaries ---- #
        out_path = os.path.join(save_dir, "generated_summaries.jsonl")
        with open(out_path, "w") as f:
            for entry in generated:
                f.write(json.dumps(entry) + "\n")

        print(f"[✓] Saved generated summaries to {out_path}")
