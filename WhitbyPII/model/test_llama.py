# WhitbyPII/model/llama_test.py

from load_llama_lora import load_llama3_lora
import torch

model, tokenizer = load_llama3_lora()

inputs = tokenizer("Hello Whitby!", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))