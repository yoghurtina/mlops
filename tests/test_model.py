import pytest
import torch
from mlops.model import GPT2FineTuner
from transformers import GPT2Tokenizer
from unittest.mock import MagicMock


@pytest.fixture
def tokenizer():
    """Fixture for loading the tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained("distilbert/distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def test_configure_optimizers(model):
    """Test the optimizer and scheduler configuration."""
    optimizers, schedulers = model.configure_optimizers()
    assert len(optimizers) == 1, "There should be one optimizer"
    assert len(schedulers) == 1, "There should be one scheduler"
    assert isinstance(optimizers[0], torch.optim.AdamW), "The optimizer should be AdamW"
    assert "scheduler" in schedulers[0], "Scheduler configuration should include 'scheduler'"
    assert schedulers[0]["interval"] == "step", "Scheduler interval should be 'step'"


def test_model_construction():
    """Test that the GPT2FineTuner is constructed properly."""
    model = GPT2FineTuner(model_name="distilbert/distilgpt2")
    assert model is not None, "Model should be created successfully"
    assert model.model.config.architectures == ["GPT2LMHeadModel"], "Model architecture should be GPT2LMHeadModel"

def test_model_forward_pass():
    """Test that the model processes a batch correctly."""
    model = GPT2FineTuner(model_name="distilbert/distilgpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("distilbert/distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(["Hello world", "Test input"],
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=512)
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    assert "loss" in outputs, "Output should contain loss"

def test_training_step():
    """Test that the training step computes the loss correctly."""
    model = GPT2FineTuner()
    trainer = MagicMock()  # Mock the Trainer
    model.trainer = trainer  # Attach the mocked trainer to the model
    batch = {
        "input_ids": torch.randint(0, 100, (4, 512)),
        "attention_mask": torch.ones((4, 512)),
    }
    loss = model.training_step(batch, 0)
    assert loss is not None, "Loss should be computed during the training step"

@pytest.fixture
def model():
    """Fixture for the GPT-2 fine-tuner model."""
    return GPT2FineTuner(
        model_name="distilbert/distilgpt2",
        learning_rate=5e-5,
        warmup_steps=500,
    )

def test_model_initialization(model):
    """Test that the model initializes correctly."""
    assert model.model is not None, "Model should be initialized"
    assert model.learning_rate == 5e-5, "Learning rate should match the initialized value"
    assert model.warmup_steps == 500, "Warmup steps should match the initialized value"

def test_forward_pass(model, tokenizer):
    """Test the forward pass of the model."""
    input_ids = tokenizer("Hello, world!", return_tensors="pt", truncation=True, padding=True).input_ids
    attention_mask = tokenizer("Hello, world!", return_tensors="pt", truncation=True, padding=True).attention_mask

    outputs = model(input_ids, attention_mask)
    assert "loss" in outputs.keys(), "Output should contain loss"

def test_validation_step(model, tokenizer):
    """Test the validation step of the model."""
    input_ids = tokenizer("Validation step test", 
                          return_tensors="pt", 
                          truncation=True, 
                          padding=True).input_ids
    attention_mask = tokenizer("Validation step test", 
                               return_tensors="pt", 
                               truncation=True, padding=True).attention_mask
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}

    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None, "Validation step should compute a loss"

