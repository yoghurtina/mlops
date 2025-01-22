from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import logging
from omegaconf import OmegaConf
import os
import tempfile
from mlops.util import download_from_gcs

app = FastAPI()

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

try:
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs"))
    config = OmegaConf.load(os.path.join(config_dir, "config.yaml"))
except:
    config = OmegaConf.load("/configs/config.yaml")

model_name = config.model.name
tokenizer = None
model = None
local_model_path = None


@app.on_event("startup")
async def load_model():
    global tokenizer, model, local_model_path

    try:
        if config.model.path.startswith("gs://"):
            # Extract bucket name and model path
            gcs_url = config.model.path[5:]  # Remove 'gs://'
            bucket_name, gcs_path = gcs_url.split("/", 1)

            # Create a temporary directory to store the downloaded model
            local_model_path = tempfile.mkdtemp()
            logger.info(f"Downloading model from GCS: {config.model.path} to {local_model_path}")
            download_from_gcs(bucket_name, gcs_path, local_model_path)
        else:
            local_model_path = config.model.path

        # Load model and tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(local_model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(local_model_path)
        model.eval()
        logger.info("Model and tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50

@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    logger.info(f"Received prompt: {request.prompt}")
    try:
        if model is None or tokenizer is None:
            logger.error("Model and tokenizer are not loaded.")
            return {"error": "Model and tokenizer are not loaded."}, 500

        if not request.prompt.strip():
            logger.error("Prompt cannot be empty.")
            return {"error": "Prompt cannot be empty."}, 400

        if request.max_length <= 0:
            logger.error("max_length must be greater than 0.")
            return {"error": "max_length must be greater than 0."}, 400

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

def main():
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
