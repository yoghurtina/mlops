from torch.quantization import quantize_dynamic
from torch.nn.utils import prune
import os
import torch
import logging
import hydra
from omegaconf import DictConfig
from transformers import GPT2LMHeadModel
from safetensors.torch import save_file

torch.backends.quantized.engine = 'qnnpack'  # For ARM devices

logger = logging.getLogger("quantize")
logging.basicConfig(level=logging.INFO)

config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs"))

def quantize_model(model: torch.nn.Module, dtype=torch.qint8) -> torch.nn.Module:
    """
    Applies dynamic quantization to the model.
    
    Args:
        model (torch.nn.Module): The PyTorch model to be quantized.
        dtype (torch.dtype): The target data type for quantization (default: torch.qint8).
    
    Returns:
        torch.nn.Module: Quantized model.
    """
    logger.info("Starting dynamic quantization...")
    quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=dtype)
    logger.info("Dynamic quantization completed.")
    return quantized_model


def prune_model(model: torch.nn.Module, amount: float = 0.2) -> torch.nn.Module:
    """
    Prunes the model by removing a percentage of weights from specified layers.

    Args:
        model (torch.nn.Module): The PyTorch model to be pruned.
        amount (float): The proportion of weights to prune (default: 0.2).

    Returns:
        torch.nn.Module: Pruned model.
    """
    logger.info(f"Pruning {amount * 100}% of weights from the model...")

    parameters_to_prune = []

    # Identify prunable layers
    for name, module in model.named_modules():
        if any(layer in name for layer in ["c_attn", "c_proj", "c_fc"]):
            logger.info(f"Adding layer {name} for pruning.")
            parameters_to_prune.append((module, 'weight'))

    if not parameters_to_prune:
        logger.warning("No prunable layers found. Skipping pruning step.")
        return model

    # Apply global pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Remove pruning reparametrizations to make the model ready for inference
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

    logger.info("Pruning completed.")
    return model

def clean_state_dict(state_dict: dict) -> dict:
    """
    Cleans the state dictionary by removing non-tensor entries.

    Args:
        state_dict (dict): The state dictionary of the model.

    Returns:
        dict: Cleaned state dictionary with only tensor entries.
    """
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cleaned_state_dict[key] = value
        else:
            logger.warning(f"Skipping non-tensor key: {key} (type: {type(value)})")
    return cleaned_state_dict

def save_quantized_pruned_model(model: torch.nn.Module, path: str):
    """
    Saves the quantized and pruned model to the specified path.
    
    Args:
        model (torch.nn.Module): The quantized and pruned model.
        path (str): Directory path to save the model.
    """
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, "model_quantized_pruned.pt")
    # save the model as .safetensors file
    model_state_dict = clean_state_dict(model.state_dict())
    save_file(model_state_dict, model_path)
    logger.info(f"Quantized and pruned model saved to {model_path}")


@hydra.main(version_base=None, config_path=config_dir, config_name="config")
def main(cfg: DictConfig):
    """
    Main function for quantizing and pruning the fine-tuned model.
    
    Args:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    # Load the fine-tuned model
    logger.info(f"Loading model from {cfg.model.path}...")
    model = GPT2LMHeadModel.from_pretrained(cfg.model.path)

    # Quantize and prune the model
    model = quantize_model(model)
    model = prune_model(model, amount=cfg.pruning.amount)

    # Save the quantized and pruned model
    save_path = os.path.join(cfg.model.path, "quantized_pruned")
    save_quantized_pruned_model(model, save_path)


if __name__ == "__main__":
    main()
