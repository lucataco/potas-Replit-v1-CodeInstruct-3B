from potassium import Potassium, Request, Response
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    REPO = "teknium/Replit-v1-CodeInstruct-3B"
    tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.to(device)

    context = {
        "model": model,
        "tokenizer": tokenizer
    }
    return context


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    max_tokens = request.json.get("max_tokens", 128)
    temperature = request.json.get("temperature", 0.7)
    model = context.get("model")
    tokenizer = context.get("tokenizer")

    top_p=0.9 
    eos_token_id=tokenizer.eos_token_id

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    generated_ids = model.generate(input_ids, max_new_tokens=max_tokens, do_sample=True, use_cache=True, temperature=temperature, top_p=top_p, eos_token_id=eos_token_id)
    output = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return Response(
        json = {"response": output}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()