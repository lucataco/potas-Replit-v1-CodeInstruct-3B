# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    AutoTokenizer.from_pretrained("teknium/Replit-v1-CodeInstruct-3B", trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained("teknium/Replit-v1-CodeInstruct-3B", torch_dtype=torch.bfloat16, trust_remote_code=True)

if __name__ == "__main__":
    download_model()