import pytorch_lightning as pl
from transformers import GPT2LMHeadModel, AdamW, get_scheduler
import torch

class GPT2FineTuner(pl.LightningModule):
    def __init__(self, model_name="distilbert/distilgpt2", learning_rate=5e-5, warmup_steps=500):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask, labels=input_ids)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"])
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"])
        loss = outputs.loss
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = get_scheduler(
            "linear",
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]
