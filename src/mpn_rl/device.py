"""PyTorch device selection for MPN-RL."""

import torch


def get_device(device_str: str = "cpu") -> torch.device:
    """Get PyTorch device."""
    if device_str == "gpu":
        device_str = "cuda"

    device = torch.device(device_str)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{device_str}' requested but no GPU is available."
            )
        print(f"Using GPU: {torch.cuda.get_device_name(device.index or 0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(device.index or 0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("Using CPU")

    return device
