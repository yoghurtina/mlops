from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import logging
from omegaconf import OmegaConf
import os

app = FastAPI()

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs"))
config = OmegaConf.load(os.path.join(config_dir, "config.yaml"))

model_name = config.model.name
tokenizer = None
model = None

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    tokenizer = GPT2Tokenizer.from_pretrained(config.model.path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(config.model.path)
    model.eval()
    logger.info("Model and tokenizer loaded successfully.")

class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50
    
@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    """
    Generate text based on the provided prompt and max length.

    Args:
        request (TextGenerationRequest): The input prompt and max length.

    Returns:
        dict: Generated text in JSON format or error message.
    """
    logger.info(f"Received prompt: {request.prompt}")
    try:
        # Validate model and tokenizer
        if model is None or tokenizer is None:
            logger.error("Model and tokenizer are not loaded.")
            return {"error": "Model and tokenizer are not loaded."}, 500

        # Validate prompt
        if not request.prompt.strip():
            logger.error("Prompt cannot be empty.")
            return {"error": "Prompt cannot be empty."}, 400

        # Validate max_length
        if request.max_length <= 0:
            logger.error("max_length must be greater than 0.")
            return {"error": "max_length must be greater than 0."}, 400

        # Generate text
        inputs = tokenizer(request.prompt, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids,
                                     max_length=request.max_length,
                                     pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")
        return {"generated_text": generated_text}, 200
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        return {"error": str(e)}, 500


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
