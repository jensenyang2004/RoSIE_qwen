"""
Qwen-Image-2.1 vendored from diffusers v0.41.0 (see the header of each file), and the loader every script uses.
"""
import torch

from .pipeline_qwenimage21 import QwenImage21Pipeline
from .transformer_qwenimage21 import QwenImage21Transformer2DModel

MODEL_ID = "Qwen/Qwen-Image-2.1-Turbo"


def load_pipeline(model_id: str = MODEL_ID, device: str = "cuda", dtype=torch.bfloat16,
                  cpu_offload: bool = False) -> QwenImage21Pipeline:
    """The vendored pipeline with the vendored transformer.

    from_pretrained alone would build diffusers' own transformer class from model_index.json, so processors and
    block wrappers installed on the vendored class would silently not run. Load the transformer here and pass it in.
    """
    transformer = QwenImage21Transformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=dtype)
    pipe = QwenImage21Pipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=dtype)
    assert type(pipe.transformer).__module__ == QwenImage21Transformer2DModel.__module__, type(pipe.transformer)
    if cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    return pipe
