from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import torch
from omegaconf import DictConfig
import hydra
from typing import Optional
import os

def evaluate_model(
    model_name: str,
    dataset_name: str,
    config_name: str,
    split: str,
    max_length: int,
    subset_size: Optional[int],
) -> float:
    """
    Evaluate a language model's perplexity on a dataset.

    Args:
        model_name (str): Name of the pre-trained model to evaluate.
        dataset_name (str): Dataset name to load from Hugging Face's datasets library.
        config_name (str): Specific configuration of the dataset.
        split (str): Dataset split to use ("train", "validation", etc.).
        max_length (int): Maximum token length for input sequences.
        subset_size (Optional[int]): Number of samples to use for evaluation (default: None).

    Returns:
        float: The perplexity of the model on the dataset.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    dataset = load_dataset(dataset_name, config_name, split=split)
    if subset_size:
        dataset = dataset.select(range(subset_size))

    # Evaluate perplexity
    total_loss = 0.0
    num_samples = len(dataset)

    for sample in dataset:
        text = sample["text"].strip()
        if not text:  
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



config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs"))
@hydra.main(version_base=None, config_path=config_dir, config_name="config")

def main(cfg: DictConfig) -> None:
    """
    Main function to evaluate the model using Hydra configuration.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    # perplexity = evaluate_model(
    #     model_name=cfg.model.name,  # Use model name from existing config
    #     dataset_name="wikitext",  # Specify directly or move to config
    #     config_name="wikitext-2-raw-v1",  # Specify directly or move to config
    #     split="validation",  # Specify directly or move to config
    #     max_length=512,  # Specify directly or move to config
    #     subset_size=100,  # Specify directly or move to config
    # )
    perplexity = evaluate_model(
        model_name=cfg.model.name,
        dataset_name=cfg.data.dataset_name,
        config_name=cfg.data.config_name,
        split=cfg.data.split,
        max_length=cfg.data.max_length,
        subset_size=cfg.data.subset_size,
    )

    print(f"Perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    main()
