import os
from typing import Dict
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from datasets import load_dataset, Dataset as HFDataset
import torch

class TextDataset(Dataset):
    """
    A custom Dataset for handling text data using Hugging Face's `datasets` library
    and tokenizing it for PyTorch models.

    Attributes:
        tokenizer (PreTrainedTokenizer): Tokenizer for text processing.
        dataset (HFDataset): Loaded and preprocessed Hugging Face dataset.
        max_length (int): Maximum token length for input sequences.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str = "wikitext",
        config_name: str = "wikitext-2-raw-v1",
        split: str = "train",
        max_length: int = 512,
    ):
        """
        Initializes the TextDataset.

        Args:
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer.
            dataset_name (str): Name of the dataset to load.
            config_name (str): Specific configuration name of the dataset.
            split (str): Split of the dataset ("train", "validation", etc.).
            max_length (int): Maximum token length for input sequences.
        """
        self.dataset: HFDataset = load_dataset(dataset_name, config_name, split=split).filter(
            lambda x: len(x["text"].strip()) > 0
        )
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_length: int = max_length

    def __len__(self) -> int:
        """
        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        """
        Retrieves a single tokenized data sample.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing `input_ids` and `attention_mask`.
        """
        text = self.dataset[idx]["text"].strip()
        if not text:
            text = "[PAD]"
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def get_dataloader(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    split: str,
    batch_size: int,
    max_length: int,
    config_name: str = "wikitext-2-raw-v1",
) -> DataLoader:
    """
    Creates a DataLoader for the TextDataset.

    Args:
        dataset_name (str): Name of the dataset to load.
        tokenizer (PreTrainedTokenizer): Hugging Face tokenizer.
        split (str): Dataset split ("train", "validation", etc.).
        batch_size (int): Batch size for the DataLoader.
        max_length (int): Maximum token length for input sequences.
        config_name (str): Specific configuration name of the dataset.

    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = TextDataset(tokenizer, dataset_name, config_name, split, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=os.cpu_count())
