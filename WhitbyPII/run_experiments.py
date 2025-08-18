import os
import sys
import json
import random
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils.trainer import train_loop, TextDataset
from dpdd.monitor_dpdd import DPDDMonitor
from peft import get_peft_model, LoraConfig, TaskType 
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
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(model_name, allow_patterns=["config.json"])
    config_path = Path(local_dir) / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            if "rope_scaling" in cfg and isinstance(cfg["rope_scaling"], dict) and "type" not in cfg["rope_scaling"]:
                cfg["rope_scaling"] = {
                    "type": "linear",
                    "factor": cfg["rope_scaling"]["factor"]
                }
                config_path.write_text(json.dumps(cfg, indent=2))
        except Exception as e:
            print(f"Could not patch rope_scaling: {e}")

EXPERIMENT_CSV = "data/factorial_experiments.csv"
DATA_PATH = "data/train_with_canaries_cleaned.csv"
MAX_LENGTH = 256      
BATCH_SIZE = 1     
REPLICATIONS = 2
NUM_EPOCHS = 2
MODEL_NAME = "meta-llama/Llama-3.1-8B"

#Tokenizer
patch_rope_scaling(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

configs = pd.read_csv(EXPERIMENT_CSV, sep=None, engine="python")
random.seed(42)

#Run Experiments
for idx, row in configs.iterrows():
    print(f"\nRunning config {idx+1}/{len(configs)}")

    lora_rank = row["lora_rank"]
    learning_rate = row["learning_rate"]
    retrieval_docs = row["retrieval_docs"]
    elastic_threshold = row["es_threshold"]
    dpdd_on = row["dpdd"]
    year_group = row["year_group"]

    for rep in range(1, REPLICATIONS + 1):
        print(f"Replication {rep}/{REPLICATIONS}")

        df = get_filtered_data(year_group)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=rep)
        train_dataset = TextDataset(train_df, tokenizer, MAX_LENGTH)
        val_dataset = TextDataset(val_df, tokenizer, MAX_LENGTH)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
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

        if int(lora_rank) > 0:
            model = apply_lora(model, r=int(lora_rank))

        dpdd_monitor = DPDDMonitor() if bool(dpdd_on) else None

        exp_id = f"lr{learning_rate}_r{lora_rank}_doc{retrieval_docs}_th{elastic_threshold}_yr{year_group}_rep{rep}"
        save_dir = os.path.join("checkpoints", exp_id)
        os.makedirs(save_dir, exist_ok=True)

        meta_path = os.path.join(save_dir, "metadata.json")
        with open(meta_path, "w") as mf:
            json.dump({
                "lora_rank": int(lora_rank),
                "learning_rate": float(learning_rate),
                "retrieval_docs": int(retrieval_docs),
                "es_threshold": float(elastic_threshold),
                "dpdd": bool(dpdd_on),
                "year_group": str(year_group),
                "replication": rep,
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "model_name": MODEL_NAME,
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "max_length": MAX_LENGTH,
            }, mf, indent=2)

        log_path = os.path.join(save_dir, "train.log")
        with tee_logs(log_path):
            model = train_loop(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                tokenizer=tokenizer,
                num_epochs=NUM_EPOCHS,
                save_path=save_dir,
                dpdd_monitor=dpdd_monitor,
                learning_rate=learning_rate,
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
            BATCH_GEN = 8 
            buffer_texts, buffer_refs = [], []

            def _flush_batch(texts, refs):
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
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=100,     
                        do_sample=False,        
                        num_beams=1,            
                        early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for t, r, d in zip(texts, refs, decoded):
                    generated.append({
                        "text": t,
                        "reference": r,
                        "generated_summary": d,
                        "config": exp_id,
                    })

            for _, vrow in val_df.iterrows():
                if not isinstance(vrow.get("text", None), str):
                    continue
                buffer_texts.append(vrow["text"][:MAX_LENGTH])
                buffer_refs.append(vrow.get("category", None))
                if len(buffer_texts) == BATCH_GEN:
                    _flush_batch(buffer_texts, buffer_refs)
                    buffer_texts, buffer_refs = [], []

            _flush_batch(buffer_texts, buffer_refs)

            out_path = os.path.join(save_dir, "generated_summaries.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for entry in generated:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"Saved generated summaries to {out_path}")
        # ---------------------------------------------------------------- #
