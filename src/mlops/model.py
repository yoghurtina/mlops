import pytorch_lightning as pl
from transformers import GPT2LMHeadModel, AdamW, get_scheduler
import torch
from typing import Any, Dict, Tuple, Optional


class GPT2FineTuner(pl.LightningModule):
    """
    A PyTorch Lightning Module for fine-tuning GPT-2 models.

    Attributes:
        model (GPT2LMHeadModel): Pre-trained GPT-2 model from Hugging Face.
        learning_rate (float): Learning rate for the optimizer.
        warmup_steps (int): Number of warmup steps for the learning rate scheduler.
    """

    def __init__(
        self,
        model_name: str = "distilbert/distilgpt2",
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
    ):
        """
        Initializes the fine-tuner.

        Args:
            model_name (str): Hugging Face model name or path to a pre-trained GPT-2 model.
            learning_rate (float): Learning rate for the optimizer.
            warmup_steps (int): Number of warmup steps for the scheduler.
        """
        super().__init__()
        self.save_hyperparameters()  # Logs hyperparameters automatically
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        """
        Forward pass for the model.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs.
            attention_mask (torch.Tensor): Attention mask for input tokens.

        Returns:
            Any: The output from the GPT-2 model.
        """
        return self.model(input_ids, attention_mask=attention_mask, labels=input_ids)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for a single batch.

        Args:
            batch (Dict[str, torch.Tensor]): The batch of data containing `input_ids` and `attention_mask`.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed training loss.
        """
        outputs = self(batch["input_ids"], batch["attention_mask"])
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Optional[torch.Tensor]:
        """
        Validation step for a single batch.

        Args:
            batch (Dict[str, torch.Tensor]): The batch of data containing `input_ids` and `attention_mask`.
            batch_idx (int): The index of the batch.

        Returns:
            Optional[torch.Tensor]: The computed validation loss.
        """
        outputs = self(batch["input_ids"], batch["attention_mask"])
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> Tuple[Any, Any]:
        """
        Configures the optimizer and scheduler.

        Returns:
            Tuple[Any, Any]: A tuple containing the optimizer and the scheduler.
        """
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = get_scheduler(
            "linear",
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler_config]
