from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = FastAPI()

model_name = "distilbert/distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()  

class TextGenerationRequest(BaseModel):
    """
    Request model for text generation API.
    """
    prompt: str
    max_length: int = 50

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    """
    Generate text based on the provided prompt and max length.

    Args:
        request (TextGenerationRequest): The input prompt and max length.

    Returns:
        dict: Generated text in JSON format.
    """
    inputs = tokenizer(request.prompt, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=request.max_length, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
