# train.py

import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from utils.trainer import TextDataset, train_loop
from dpdd.monitor_dpdd import DPDDMonitor

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/train_with_canaries_cleaned.csv')
parser.add_argument('--lora_rank', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=3e-4)
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--dpdd', action='store_true')
parser.add_argument('--output_dir', type=str, default='checkpoints')
args = parser.parse_args()

# Load data
df = pd.read_csv(args.data_path)
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

# Load tokenizer and model
base_model_name = "meta-llama/Llama-3-8b-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

# Fix pad token issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)

#Add LoRA
lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

#Tokenized datasets
train_dataset = TextDataset(train_df, tokenizer, max_length=512)
val_dataset = TextDataset(val_df, tokenizer, max_length=512)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

#DPDD monitor
dpdd_monitor = DPDDMonitor() if args.dpdd else None

#Train
trained_model = train_loop(
    model,
    train_loader,
    val_loader,
    tokenizer,
    num_epochs=args.num_epochs,
    save_path=args.output_dir,
    dpdd_monitor=dpdd_monitor,
    learning_rate=args.learning_rate
)
