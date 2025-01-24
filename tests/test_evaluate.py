import pytest
from mlops.evaluate import evaluate_model
from omegaconf import OmegaConf


@pytest.fixture
def mock_config():
    """Fixture for a mock Hydra configuration."""
    config = {
        "model": {"name": "distilbert/distilgpt2"},
        "data": {
            "dataset_name": "wikitext",
            "config_name": "wikitext-2-raw-v1",
            "split": "validation",
            "max_length": 512,
            "subset_size": 10,
        },
    }
    return OmegaConf.create(config)


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

