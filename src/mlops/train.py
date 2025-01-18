import pytorch_lightning as pl
from transformers import GPT2Tokenizer
from data import get_dataloader
from model import GPT2FineTuner


def main() -> None:
    """
    Main function to fine-tune a GPT-2 model using PyTorch Lightning.
    """
    model_name = "distilbert/distilgpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  

    train_loader = get_dataloader(
        dataset_name="wikitext",
        tokenizer=tokenizer,
        split="train",
        batch_size=4,
        max_length=512,
    )
    val_loader = get_dataloader(
        dataset_name="wikitext",
        tokenizer=tokenizer,
        split="validation",
        batch_size=2,
        max_length=128,
    )

    model = GPT2FineTuner(model_name=model_name)

    trainer = pl.Trainer(
        max_epochs=1,  
        accelerator="cpu",  # or gpu, if available
        devices=1,  
        precision=32, 
        log_every_n_steps=1,
        accumulate_grad_batches=4,  # Simulates a larger batch size
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
