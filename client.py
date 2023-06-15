import banana_dev as banana

p = {
    "prompt": "Write a python function that determines if a given number is prime or not"
}

api_key = ""
model_key = ""

out = banana.run(api_key, model_key, p)
print(out["modelOutputs"][0])