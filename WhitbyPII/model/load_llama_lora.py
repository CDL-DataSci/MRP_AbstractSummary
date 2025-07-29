# WhitbyPII/model/load_llama_lora.py

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

def load_llama3_lora(
    base_model="meta-llama/Llama-3.1-8B",
    load_in_8bit=True,
    r=16,
    alpha=32,
    dropout=0.05
):
    #Quantization config
    quant_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None
    ) if load_in_8bit else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        quantization_config=quant_config,
        trust_remote_code=True
    )

    # LoRA config
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Inject LoRA adapters
    model = get_peft_model(model, lora_config)
    return model, tokenizer