from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import logging

app = FastAPI()

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

model_name = "distilbert/distilgpt2"
tokenizer = None
model = None

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    logger.info("Model and tokenizer loaded successfully.")

class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50

@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    logger.info(f"Received prompt: {request.prompt}")
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, 
                                     max_length=request.max_length, 
                                     pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")
        return {"generated_text": generated_text}
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
