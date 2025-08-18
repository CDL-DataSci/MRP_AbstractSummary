import os
import sys
import json
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType
from dpdd.monitor_dpdd import DPDDMonitor
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

class _TeeIO:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass
    def close(self): 
        for s in self.streams:
            try:
                s.close()
            except Exception:
                pass

class tee_logs:
    def __init__(self, logfile_path):
        self.logfile_path = logfile_path
        self._log_f = None
        self._old_out = None
        self._old_err = None
    def __enter__(self):
        os.makedirs(os.path.dirname(self.logfile_path), exist_ok=True)
        self._log_f = open(self.logfile_path, "a", buffering=1, encoding="utf-8")
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout = _TeeIO(self._old_out, self._log_f)
        sys.stderr = _TeeIO(self._old_err, self._log_f)
        print(f"[LOG] Writing training logs to {self.logfile_path}")
        return self
    def __exit__(self, exc_type, exc, tb):
        if self._log_f:
            self._log_f.flush()
            self._log_f.close()
        sys.stdout = self._old_out
        sys.stderr = self._old_err
        return False
#Rope scaling workaround for CSV
def patch_rope_scaling(model_name):
    from pathlib import Path
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        print("not available for rope patch")
        return
    try:
        local_dir = snapshot_download(model_name, allow_patterns=["config.json"], token=True)
        config_path = Path(local_dir) / "config.json"
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            rs = cfg.get("rope_scaling", None)
            if isinstance(rs, dict) and "type" not in rs and "factor" in rs:
                cfg["rope_scaling"] = {"type": "linear", "factor": float(rs["factor"])}
                config_path.write_text(json.dumps(cfg, indent=2))
    except Exception as e:
        print(f"Could not patch rope_scaling (continuing): {e}")

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        texts = dataframe["text"].astype(str).tolist()
        self.examples = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.refs = dataframe.get("category", pd.Series([None]*len(texts))).tolist()

    def __len__(self):
        return self.examples["input_ids"].size(0)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.examples.items()}
        item["ref"] = self.refs[idx]
        return item

def save_hf(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[HF Save] Model and tokenizer saved to {output_dir}")

def train_with_dpdd(
    model,
    train_loader,
    val_loader,
    tokenizer,
    num_epochs,
    save_dir,
    learning_rate,
    dpdd_monitor: DPDDMonitor,
    dpdd_keep_threshold: float = 0.1,
):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    metrics_path = os.path.join(save_dir, "metrics.jsonl")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        running_loss = 0.0
        steps = 0

        for batch in train_loader:
            device = next(model.parameters()).device
            batch_on = {k: (v.to(device) if hasattr(v, "to") else v)
                        for k, v in batch.items() if k in ("input_ids", "attention_mask")}
            outputs = model(**batch_on, labels=batch_on["input_ids"])
            loss = outputs.loss
            loss.backward()

            if dpdd_monitor:
                dpdd_monitor.record_gradients(model)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            steps += 1

        train_loss = running_loss / max(steps, 1)

        dropped_count = 0
        dpdd_score = None
        if dpdd_monitor:
            dpdd_score = float(dpdd_monitor.compute_dpdd_score())
            print(f"[DPDD] Score after epoch {epoch+1}: {dpdd_score:.4f}")

            if hasattr(train_loader.dataset, "__len__"):
                total = len(train_loader.dataset)
                keep_indices = dpdd_monitor.get_non_dropped_indices(
                    threshold=dpdd_keep_threshold, total=total
                )
                keep_indices = sorted(set(keep_indices))
                dropped = sorted(set(range(total)) - set(keep_indices))
                dropped_count = len(dropped)
                print(f"[DPDD] Dropped {dropped_count} / {total} samples for next epoch.")

                new_dataset = Subset(train_loader.dataset, keep_indices)
                train_loader = DataLoader(new_dataset, batch_size=train_loader.batch_size, shuffle=True)

            dpdd_monitor.reset()

#Validation
        model.eval()
        val_loss = 0.0
        vsteps = 0
        with torch.no_grad():
            for vbatch in val_loader:
                device = next(model.parameters()).device
                v_on = {k: (v.to(device) if hasattr(v, "to") else v)
                        for k, v in vbatch.items() if k in ("input_ids", "attention_mask")}
                vout = model(**v_on, labels=v_on["input_ids"])
                val_loss += vout.loss.item()
                vsteps += 1
        val_loss = val_loss / max(vsteps, 1)
        model.train()

        ckpt_dir = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        print(f"[Checkpoint] Model saved to {ckpt_dir}")

        with open(metrics_path, "a", encoding="utf-8") as mf:
            mf.write(json.dumps({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "dpdd_score": dpdd_score,
                "dropped_count": dropped_count,
                "learning_rate": float(learning_rate),
            }) + "\n")

        print(f"[Validation] Loss after epoch {epoch+1}: {val_loss:.4f}")

    save_hf(model, tokenizer, os.path.join(save_dir, "hf_format"))
    print("Training complete.")
    return model

def main():
    EXPERIMENT_CSV = "data/factorial_experiments.csv"
    DATA_PATH = "data/train_with_canaries_cleaned.csv"
    MAX_LENGTH = 256
    TRAIN_BATCH = 1
    VAL_BATCH = 1
    NUM_EPOCHS = 2
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    BATCH_GEN = 8           
    MAX_NEW_TOKENS = 100
    DPDD_KEEP_THRESHOLD = 0.1 

    patch_rope_scaling(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    configs = pd.read_csv(EXPERIMENT_CSV, sep=None, engine="python")
    random.seed(42)

    df_full = pd.read_csv(DATA_PATH, sep="\t", on_bad_lines="skip")

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
        lcfg = LoraConfig(
            r=int(r),
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
        return model

    for idx, row in configs.iterrows():
        print(f"\n[DPDD RUN] Config {idx+1}/{len(configs)}")
        lora_rank = row["lora_rank"]
        learning_rate = row["learning_rate"]
        retrieval_docs = row["retrieval_docs"]  
        elastic_threshold = row["es_threshold"]  
        dpdd_on = row["dpdd"] 
        year_group = row["year_group"]

        use_dpdd_filtering = True

        for rep in (1, 2):  
            print(f"Replication {rep}/2")

            df = get_filtered_data(year_group)
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=rep)
            train_dataset = TextDataset(train_df, tokenizer, MAX_LENGTH)
            val_dataset = TextDataset(val_df, tokenizer, MAX_LENGTH)
            train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                token=True,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map="auto",
            )
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False

            if hasattr(model, "enable_input_require_grads"):
                try:
                    model.enable_input_require_grads()
                except Exception:
                    pass

            #apply LoRA
            if int(lora_rank) > 0:
                model = apply_lora(model, r=int(lora_rank))

            #DPDD monitor
            dpdd_monitor = DPDDMonitor() if use_dpdd_filtering else None

            exp_id = f"lr{learning_rate}_r{lora_rank}_doc{retrieval_docs}_th{elastic_threshold}_yr{year_group}_rep{rep}"
            save_dir = os.path.join("checkpoints_dpdd", exp_id)
            os.makedirs(save_dir, exist_ok=True)

            meta_path = os.path.join(save_dir, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump({
                    "lora_rank": int(lora_rank),
                    "learning_rate": float(learning_rate),
                    "retrieval_docs": int(retrieval_docs),
                    "es_threshold": float(elastic_threshold),
                    "dpdd": True,
                    "year_group": str(year_group),
                    "replication": rep,
                    "n_train": int(len(train_df)),
                    "n_val": int(len(val_df)),
                    "model_name": MODEL_NAME,
                    "epochs": int(2),
                    "train_batch": int(TRAIN_BATCH),
                    "val_batch": int(VAL_BATCH),
                    "max_length": int(MAX_LENGTH),
                    "dpdd_keep_threshold": float(DPDD_KEEP_THRESHOLD),
                }, mf, indent=2)

            log_path = os.path.join(save_dir, "train.log")
            with tee_logs(log_path):
                model = train_with_dpdd(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    tokenizer=tokenizer,
                    num_epochs=2,
                    save_dir=save_dir,
                    learning_rate=learning_rate,
                    dpdd_monitor=dpdd_monitor,
                    dpdd_keep_threshold=DPDD_KEEP_THRESHOLD,
                )

                print("Generating summaries...")
                model.eval()
                try:
                    model.config.use_cache = True
                except Exception:
                    pass

                try:
                    gen_device = next(model.parameters()).device
                except Exception:
                    gen_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                generated = []
                buf_texts, buf_refs = [], []

                def flush_batch(texts, refs):
                    if not texts:
                        return
                    enc = tokenizer(
                        texts,
                        return_tensors="pt",
                        truncation=True,
                        max_length=MAX_LENGTH,
                        padding=True,
                    )
                    input_ids = enc["input_ids"].to(gen_device)
                    attention_mask = enc["attention_mask"].to(gen_device)
                    with torch.no_grad():
                        outs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False,
                            num_beams=1,
                            early_stopping=True,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    dec = tokenizer.batch_decode(outs, skip_special_tokens=True)
                    for t, r, d in zip(texts, refs, dec):
                        generated.append({
                            "text": t,
                            "reference": r,
                            "generated_summary": d,
                            "config": exp_id,
                        })

                for _, vrow in val_df.iterrows():
                    t = vrow.get("text", None)
                    if not isinstance(t, str):
                        continue
                    buf_texts.append(t[:MAX_LENGTH])
                    buf_refs.append(vrow.get("category", None))
                    if len(buf_texts) == BATCH_GEN:
                        flush_batch(buf_texts, buf_refs)
                        buf_texts, buf_refs = [], []

                flush_batch(buf_texts, buf_refs)

                out_path = os.path.join(save_dir, "generated_summaries.jsonl")
                with open(out_path, "w", encoding="utf-8") as f:
                    for e in generated:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")
                print(f"Saved generated summaries to {out_path}")

if __name__ == "__main__":
    main()
