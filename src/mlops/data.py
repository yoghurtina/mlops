import os
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset

class TextDataset(Dataset):
    def __init__(self, tokenizer, dataset_name="wikitext", config_name="wikitext-2-raw-v1", split="train", max_length=512):
        self.dataset = load_dataset(dataset_name, config_name, split=split).filter(lambda x: len(x["text"].strip()) > 0)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
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

def get_dataloader(dataset_name, tokenizer, split, batch_size, max_length, config_name="wikitext-2-raw-v1"):
    dataset = TextDataset(tokenizer, dataset_name, config_name, split, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=os.cpu_count())
