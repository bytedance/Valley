# coding=utf-8

"""
ValleyOmni model: a minimal multimodal model that extends Qwen3-VL's text+vision pipeline with audio understanding.

This package provides:
- ValleyOmniConfig
- ValleyOmniForConditionalGeneration
- ValleyOmniProcessor

It registers AutoConfig/AutoModel/AutoProcessor mappings at import time using the dynamic register APIs,
so `from_pretrained` will work with local checkpoints that have `model_type: "valley_omni"` in their config.
"""

from .configuration_valley_omni import ValleyOmniConfig
from .modeling_valley_omni import ValleyOmniForConditionalGeneration, ValleyOmniModel
from .processing_valley_omni import ValleyOmniProcessor

# Try to register into Auto* dynamically (keeps changes local to this module)
try:
    from ..auto.configuration_auto import AutoConfig
    from ..auto.modeling_auto import AutoModel, AutoModelForCausalLM
    from ..auto.processing_auto import AutoProcessor

    # Register config string -> class
    AutoConfig.register("valley_omni", ValleyOmniConfig, exist_ok=True)
    # Register config class -> model classes
    AutoModel.register(ValleyOmniConfig, ValleyOmniModel, exist_ok=True)
    AutoModelForCausalLM.register(ValleyOmniConfig, ValleyOmniForConditionalGeneration, exist_ok=True)
    # Register config class -> processor
    AutoProcessor.register(ValleyOmniConfig, ValleyOmniProcessor, exist_ok=True)
except Exception:
    # Soft-fail if Auto* is not available during documentation builds or partial installations
    pass

__all__ = [
    "ValleyOmniConfig",
    "ValleyOmniModel",
    "ValleyOmniForConditionalGeneration",
    "ValleyOmniProcessor",
]
