from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = FastAPI()

model_name = "distilbert/distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=request.max_length)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
