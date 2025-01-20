import os
from typing import Dict, Optional
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
        limit: Optional[int] = None,
    ):
        """
        Initializes the TextDataset.

        Args:
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer for text processing.
            dataset_name (str): Name of the dataset to load.
            config_name (str): Specific configuration name of the dataset.
            split (str): Dataset split ("train", "validation", etc.).
            max_length (int): Maximum token length for input sequences.

        Raises:
            RuntimeError: If the dataset fails to load.
        """
        try:
            self.dataset: HFDataset = load_dataset(
                dataset_name, config_name, split=split, cache_dir="./cache"
            ).filter(lambda x: len(x["text"].strip()) > 0)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_length: int = max_length

        if limit is not None:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a tokenized data sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - `input_ids`: Tensor of tokenized input IDs.
                - `attention_mask`: Tensor of attention masks.
        """
        text = self.dataset[idx]["text"].strip()
        if not text:
            text = "[PAD]"
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",  # Ensures consistent padding to max_length
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
    num_workers: Optional[int] = None,
    limit: Optional[int] = None,
) -> DataLoader:
    """
    Creates a DataLoader for the TextDataset.

    Args:
        dataset_name (str): Name of the dataset to load.
        tokenizer (PreTrainedTokenizer): Hugging Face tokenizer for text processing.
        split (str): Dataset split ("train", "validation", etc.).
        batch_size (int): Batch size for the DataLoader.
        max_length (int): Maximum token length for input sequences.
        config_name (str): Specific configuration name of the dataset (default: "wikitext-2-raw-v1").
        num_workers (Optional[int]): Number of worker processes for data loading.

    Returns:
        DataLoader: A PyTorch DataLoader for the dataset, ready for training or evaluation.
    """
    dataset = TextDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        config_name=config_name,
        split=split,
        max_length=max_length,
        limit=limit,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),  # Shuffle data only for training split
        num_workers=num_workers or os.cpu_count(),  # Use specified num_workers or default to CPU count
    )
