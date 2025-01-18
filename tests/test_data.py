import pytest
from transformers import GPT2Tokenizer
from mlops.data import TextDataset, get_dataloader

def test_text_dataset_loading():
    """Test that the dataset loads and filters properly."""
    tokenizer = GPT2Tokenizer.from_pretrained("distilbert/distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
    dataset = TextDataset(
        tokenizer=tokenizer, dataset_name="wikitext", config_name="wikitext-2-raw-v1", split="train"
    )
    assert len(dataset) > 0, "Dataset should not be empty"

def test_text_dataset_tokenization():
    """Test that the dataset tokenizes the text correctly."""
    tokenizer = GPT2Tokenizer.from_pretrained("distilbert/distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
    max_length = 512
    dataset = TextDataset(
        tokenizer=tokenizer, 
        dataset_name="wikitext", 
        config_name="wikitext-2-raw-v1", 
        split="train", max_length=max_length
    )
    sample = dataset[0]
    assert "input_ids" in sample and "attention_mask" in sample, \
        "Tokenization should produce input_ids and attention_mask"
    assert len(sample["input_ids"]) == max_length, f"Input IDs should match the max length {max_length}"
    assert len(sample["attention_mask"]) == max_length, f"Attention mask should match the max length {max_length}"

def test_dataloader_creation():
    """Test that the DataLoader is created without errors."""
    tokenizer = GPT2Tokenizer.from_pretrained("distilbert/distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
    dataloader = get_dataloader(
        dataset_name="wikitext",
        tokenizer=tokenizer,
        split="train",
        batch_size=4,
        max_length=512,
    )
    assert len(dataloader) > 0, "Dataloader should have batches"
    for batch in dataloader:
        assert "input_ids" in batch and "attention_mask" in batch, "Dataloader should produce input batches"
        assert batch["input_ids"].shape[0] <= 4, "Batch size should not exceed the specified size"

def test_dataloader_with_different_batch_sizes():
    """Test that the DataLoader handles different batch sizes correctly."""
    tokenizer = GPT2Tokenizer.from_pretrained("distilbert/distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token  
    for batch_size in [1, 8, 16]:
        dataloader = get_dataloader(
            dataset_name="wikitext",
            tokenizer=tokenizer,
            split="train",
            batch_size=batch_size,
            max_length=512,
        )
        for batch in dataloader:
            assert batch["input_ids"].shape[0] <= batch_size, \
                f"Batch size should not exceed {batch_size}"

def test_text_dataset_empty_text_handling():
    """Test that the dataset handles empty or invalid text samples gracefully."""
    tokenizer = GPT2Tokenizer.from_pretrained("distilbert/distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = TextDataset(
        tokenizer=tokenizer, 
        dataset_name="wikitext", 
        config_name="wikitext-2-raw-v1", 
        split="train", max_length=512
    )
    for idx in range(len(dataset)):
        sample = dataset[idx]
        assert sample["input_ids"] is not None, "Input IDs should not be None"
        assert sample["attention_mask"] is not None, "Attention mask should not be None"

def test_dataloader_with_invalid_dataset():
    """Test that the DataLoader raises an error for an invalid dataset."""
    tokenizer = GPT2Tokenizer.from_pretrained("distilbert/distilgpt2")
    with pytest.raises(RuntimeError):
        get_dataloader(
            dataset_name="invalid_dataset",
            tokenizer=tokenizer,
            split="train",
            batch_size=4,
            max_length=512,
        )

def test_dataset_padding_consistency():
    """Test that all sequences in a batch have consistent padding."""
    tokenizer = GPT2Tokenizer.from_pretrained("distilbert/distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataloader = get_dataloader(
        dataset_name="wikitext",
        tokenizer=tokenizer,
        split="train",
        batch_size=4,
        max_length=512,
    )
    for batch in dataloader:
        for sample in batch["input_ids"]:
            assert len(sample) == 512, "Each sample should be padded to max length"

def test_dataset_truncation():
    """Test that long sequences are truncated to max length."""
    tokenizer = GPT2Tokenizer.from_pretrained("distilbert/distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = TextDataset(
        tokenizer=tokenizer, dataset_name="wikitext", config_name="wikitext-2-raw-v1", split="train", max_length=128
    )
    sample = dataset[0]
    assert len(sample["input_ids"]) == 128, "Input IDs should be truncated to max length"
    assert len(sample["attention_mask"]) == 128, "Attention mask should be truncated to max length"
