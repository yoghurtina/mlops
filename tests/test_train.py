import pytest
from mlops.train import main
from omegaconf import OmegaConf
from unittest.mock import patch


@pytest.fixture
def mock_config():
    """Fixture for a mock configuration."""
    config = {
        "model": {
            "name": "distilbert/distilgpt2",
            "learning_rate": 5e-5,
            "warmup_steps": 500,
        },
        "data": {
            "dataset_name": "wikitext",
            "batch_size": 4,
            "max_length": 512,
        },
        "training": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "precision": 32,
            "accumulate_grad_batches": 2,
        },
        "logging": {
            "log_every_n_steps": 10,
        },
    }
    return OmegaConf.create(config)


@patch("mlops.train.get_dataloader")
@patch("mlops.train.GPT2FineTuner")
@patch("mlops.train.pl.Trainer")
def test_main(mock_trainer, mock_model, mock_dataloader, mock_config):
    """
    Test the main function in train.py.
    """
    # Mock the dataloader to return dummy data
    mock_dataloader.return_value = iter([{"input_ids": [0], "attention_mask": [1]}])

    # Mock the trainer
    mock_trainer_instance = mock_trainer.return_value
    mock_trainer_instance.fit.return_value = None

    # Run the main function
    with patch("mlops.train.hydra.main", lambda *args, **kwargs: lambda fn: fn):
        main(mock_config)

    # Assertions
    mock_dataloader.assert_called()
    mock_model.assert_called_with(
        model_name=mock_config.model.name,
        learning_rate=mock_config.model.learning_rate,
        warmup_steps=mock_config.model.warmup_steps,
    )
    mock_trainer.assert_called()
    mock_trainer_instance.fit.assert_called()


def test_hydra_config_loading(mock_config):
    """
    Test that the Hydra configuration is loaded correctly.
    """
    assert mock_config.model.name == "distilbert/distilgpt2"
    assert mock_config.data.batch_size == 4
    assert mock_config.training.max_epochs == 1
