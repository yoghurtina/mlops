import pytorch_lightning as pl
from transformers import GPT2Tokenizer
from data import get_dataloader
from model import GPT2FineTuner

def main():
    model_name = "distilbert/distilgpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = get_dataloader("wikitext", tokenizer, "train", batch_size=4, max_length=512)
    val_loader = get_dataloader("wikitext", tokenizer, "validation", batch_size=2, max_length=128)

    model = GPT2FineTuner(model_name)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        precision=32,
        log_every_n_steps=1,
        accumulate_grad_batches=4,  # Simulates batch size of 4
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
