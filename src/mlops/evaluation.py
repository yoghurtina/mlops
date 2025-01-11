from transformers import GPT2Tokenizer, GPT2LMHeadModel
import evaluate
import torch
from datasets import load_dataset

def evaluate_model(model_name, dataset_name="wikitext", config_name="wikitext-2-raw-v1", split="validation", max_length=512):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    dataset = load_dataset(dataset_name, config_name, split=split).select(range(100))  # Use a small subset for testing
    metric = evaluate.load("perplexity")  # Use the evaluate library

    total_loss = 0
    for sample in dataset:
        text = sample["text"]
        if not text.strip(): # Skip empty strings
            continue
        inputs = tokenizer(
            text, return_tensors="pt", max_length=max_length, truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            total_loss += loss
    perplexity = torch.exp(torch.tensor(total_loss / len(dataset))).item()
    return perplexity

if __name__ == "__main__":
    model_name = "distilbert/distilgpt2"
    print("Perplexity:", evaluate_model(model_name))
