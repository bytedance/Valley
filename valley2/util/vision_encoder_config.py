from transformers import PretrainedConfig


siglip_config = PretrainedConfig.from_dict(
    {
        "attention_dropout": 0.0,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "image_size": 384,
        "intermediate_size": 4304,
        "layer_norm_eps": 1e-06,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": 27,
        "patch_size": 14,
    }
)

qwen2vl_vit_config = PretrainedConfig.from_dict(
    {
        "depth": 32,
        "embed_dim": 1280,
        "hidden_act": "quick_gelu",
        "hidden_size": 3584,
        "in_channels": 3,
        "in_chans": 3,
        "mlp_ratio": 4,
        "model_type": "qwen2_vl",
        "num_heads": 16,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "spatial_patch_size": 14,
        "temporal_patch_size": 2,
        "_attn_implementation": "flash_attention_2",
        "_attn_implementation_internal": "flash_attention_2"
    }
)


siglip_processor_config = {
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [
        0.5,
        0.5,
        0.5
    ],
    "image_processor_type": "SiglipImageProcessor",
    "image_std": [
        0.5,
        0.5,
        0.5
    ],
    "processor_class": "SiglipProcessor",
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "size": {
        "height": 384,
        "width": 384
    }
}

qwen2vl_processor_config = {
    "min_pixels": 3136,
    "max_pixels": 12845056,
    "patch_size": 14,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [
        0.48145466,
        0.4578275,
        0.40821073
    ],
    "image_std": [
        0.26862954,
        0.26130258,
        0.27577711
    ],
    "image_processor_type": "Qwen2VLImageProcessor",
    "processor_class": "Qwen2VLProcessor"
}