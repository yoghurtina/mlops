import pytest
from unittest.mock import patch
from mlops.evaluate import evaluate_model, main
from omegaconf import OmegaConf
import torch


@pytest.fixture
def mock_config():
    """Fixture for a mock Hydra configuration."""
    config = {
        "model": {
            "name": "distilbert/distilgpt2",
        },
        "data": {
            "dataset_name": "wikitext",
            "config_name": "wikitext-2-raw-v1",
            "split": "validation",
            "max_length": 512,
            "subset_size": 10,
        },
    }
    return OmegaConf.create(config)


@patch("mlops.evaluate.load_dataset")
@patch("mlops.evaluate.GPT2LMHeadModel.from_pretrained")
@patch("mlops.evaluate.GPT2Tokenizer.from_pretrained")
def test_evaluate_model(mock_tokenizer, mock_model, mock_dataset):
    """Test the evaluate_model function with mocked dependencies."""
    # Mock tokenizer
    tokenizer_instance = mock_tokenizer.return_value
    tokenizer_instance.pad_token = "[PAD]"
    tokenizer_instance.return_tensors = "pt"

    # Mock model
    model_instance = mock_model.return_value
    model_instance.eval.return_value = None
    model_instance.__call__.return_value = torch.nn.functional.softmax(
        torch.rand(1), dim=0
    )

    # Mock dataset
    mock_dataset.return_value = [{"text": "Once upon a time."} for _ in range(10)]

    # Call the function
    perplexity = evaluate_model(
        model_name="mock_model",
        dataset_name="mock_dataset",
        config_name="mock_config",
        split="validation",
        max_length=512,
        subset_size=10,
    )

    # Assertions
    assert perplexity > 0, "Perplexity should be a positive value"
    mock_tokenizer.assert_called_once()
    mock_model.assert_called_once()
    mock_dataset.assert_called_once()


@patch("mlops.evaluate.evaluate_model")
def test_main(mock_evaluate_model, mock_config):
    """Test the main function with a mocked evaluate_model."""
    mock_evaluate_model.return_value = 42.0  # Mock perplexity result

    # Patch Hydra's decorator to call the function directly
    with patch("mlops.evaluate.hydra.main", lambda *args, **kwargs: lambda fn: fn):
        main(mock_config)

    # Assertions
    mock_evaluate_model.assert_called_once_with(
        model_name=mock_config.model.name,
        dataset_name=mock_config.data.dataset_name,
        config_name=mock_config.data.config_name,
        split=mock_config.data.split,
        max_length=mock_config.data.max_length,
        subset_size=mock_config.data.subset_size,
    )


def test_evaluate_model_invalid_model_name():
    """Test evaluate_model with an invalid model name."""
    with pytest.raises(Exception):
        evaluate_model(
            model_name="invalid_model_name",
            dataset_name="wikitext",
            config_name="wikitext-2-raw-v1",
            split="validation",
            max_length=512,
            subset_size=10,
        )


def test_evaluate_model_empty_dataset():
    """Test evaluate_model with an empty dataset."""
    with patch("mlops.evaluate.load_dataset", return_value=[]):
        with pytest.raises(ZeroDivisionError):
            evaluate_model(
                model_name="distilbert/distilgpt2",
                dataset_name="wikitext",
                config_name="wikitext-2-raw-v1",
                split="validation",
                max_length=512,
                subset_size=10,
            )
