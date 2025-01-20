import pytorch_lightning as pl
from transformers import GPT2Tokenizer
from mlops.data import get_dataloader
from mlops.model import GPT2FineTuner
import hydra
from omegaconf import DictConfig
import os

config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs"))

@hydra.main(version_base=None, config_path=config_dir, config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to fine-tune a GPT-2 model using PyTorch Lightning and Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    # Load tokenizer
    print(f"Training with model: {cfg.model.name}")
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model.name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure compatibility with padding

    # Load data loaders
    train_loader = get_dataloader(
        dataset_name=cfg.data.dataset_name,
        tokenizer=tokenizer,
        split="train",
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.max_length,
    )
    val_loader = get_dataloader(
        dataset_name=cfg.data.dataset_name,
        tokenizer=tokenizer,
        split="validation",
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.max_length,
        limit=cfg.data.limit,
    )
    

    # Initialize model
    model = GPT2FineTuner(
        model_name=cfg.model.name,
        learning_rate=cfg.model.learning_rate,
        warmup_steps=cfg.model.warmup_steps,
    )

    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
