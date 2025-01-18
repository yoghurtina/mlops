from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import evaluate
import torch
from typing import Optional


def evaluate_model(
    model_name: str,
    dataset_name: str = "wikitext",
    config_name: str = "wikitext-2-raw-v1",
    split: str = "validation",
    max_length: int = 512,
    subset_size: Optional[int] = 100,
) -> float:
    """
    Evaluate a language model's perplexity on a dataset.

    Args:
        model_name (str): Name of the pre-trained model to evaluate.
        dataset_name (str): Dataset name to load from Hugging Face's datasets library.
        config_name (str): Specific configuration of the dataset.
        split (str): Dataset split to use ("train", "validation", etc.).
        max_length (int): Maximum token length for input sequences.
        subset_size (Optional[int]): Number of samples to use for evaluation (default: 100).

    Returns:
        float: The perplexity of the model on the dataset.
    """
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # Load and preprocess dataset
    dataset = load_dataset(dataset_name, config_name, split=split)
    if subset_size:
        dataset = dataset.select(range(subset_size))
    
    # Load evaluation metric
    evaluate.load("perplexity")

    # Evaluate perplexity
    total_loss = 0.0
    num_samples = len(dataset)

    for sample in dataset:
        text = sample["text"].strip()
        if not text:  # Skip empty or invalid samples
            continue
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()

    average_loss = total_loss / num_samples
    perplexity = torch.exp(torch.tensor(average_loss)).item()

    return perplexity


if __name__ == "__main__":
    model_name = "distilbert/distilgpt2"
    perplexity = evaluate_model(model_name)
    print(f"Perplexity: {perplexity:.2f}")
