import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mlops.data import get_dataloader
from mlops.model import GPT2FineTuner
import hydra
from omegaconf import DictConfig
import os
import logging

logger = logging.getLogger("train")
logging.basicConfig(level=logging.INFO)

config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs"))

@hydra.main(version_base=None, config_path=config_dir, config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to fine-tune a GPT-2 model using PyTorch Lightning and Hydra for configuration.

    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    # Load tokenizer
    logger.info(f"Training with model: {cfg.model.name}")
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
    
    # Configure checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.output_path,  # Directory to save the model
        filename="model-{epoch:02d}-{val_loss:.2f}",  # Unique filename with metrics
        save_top_k=1,  # Save only the best model
        monitor="val_loss",  # Metric to monitor for saving the best model
        mode="min",  # Save the model with the minimum validation loss
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
        callbacks=[checkpoint_callback],
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)

    # Save the best model in Hugging Face's format
    model_path = checkpoint_callback.model_path
    logger.info(f"Best model checkpoint saved at: {model_path}")
    
    # Save the model and tokenizer directly in cfg.training.output_path
    logger.info(f"Saving the model and tokenizer to {cfg.training.output_path}")
    model.model.save_pretrained(cfg.training.output_path)
    tokenizer.save_pretrained(cfg.training.output_path)

    logger.info(f"Model and tokenizer saved successfully in: {cfg.training.output_path}")
    logger.info("Files saved:")
    logger.info(f"- {os.path.join(cfg.training.output_path, 'model.safetensors')}: Model weights")
    logger.info(f"- {os.path.join(cfg.training.output_path, 'config.json')}: Model configuration")
    logger.info(f"- {os.path.join(cfg.training.output_path, 'vocab.json')}: Tokenizer vocabulary")
    logger.info(f"- {os.path.join(cfg.training.output_path, 'merges.txt')}: Tokenizer merges")


if __name__ == "__main__":
    main()
