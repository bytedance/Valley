# coding=utf-8

"""
Configuration for the ValleyOmni model.

Design goals:
- Reuse Qwen3-VL text+vision configs for the main LLM trunk and the visual encoder
- Add audio-related config fields and metadata needed to integrate an external audio encoder
- Keep RoPE parameters compatible with Qwen3-VL's MRoPE/TM-RoPE usage
"""

from typing import Optional

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters, rope_config_validation, standardize_rope_params
from ..qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig


class ValleyOmniConfig(PreTrainedConfig):
    r"""
    Configuration for the ValleyOmni multimodal model (text + vision + audio).

    This config composes sub-configs and adds audio-specific fields.

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen3VLTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Qwen3VLVisionConfig`):
            The config object or dictionary of the vision backbone.
        audio_encoder_type (`str`, *optional*, defaults to "qwen3_omni_moe"):
            Which upstream audio encoder to use. Options: "qwen2_5_omni", "qwen3_omni_moe".
        audio_config (`dict`, *optional*):
            Optional nested dict to initialize the chosen upstream audio encoder. When `None`, defaults from upstream
            modules are used.
        sample_rate (`int`, *optional*, defaults to 16000):
            Audio sampling rate used by the processor.
        feature_dim (`int`, *optional*, defaults to 128):
            Number of mel bins or audio features per frame used by the upstream feature extractor.
        connector_hidden_size (`int`, *optional*, defaults to 4096):
            Target hidden size of the LLM trunk; audio connector will project to this dimension.
        apply_audio_downsample (`bool`, *optional*, defaults to False):
            Whether the audio connector should downsample the time axis (e.g., stride 2).
        tm_rope_sections (`list[int]`, *optional*, defaults to [24, 20, 20]):
            Multimodal RoPE section splits for [T, H, W]. Used to adapt MRoPE/TM-RoPE.
        position_id_per_seconds (`int`, *optional*, defaults to 25):
            Temporal granularity (tokens per second) used when computing 3D position ids for audio/video.
        audio_token_id (`int`, *optional*, defaults to 151646):
            The audio token index to encode the audio prompt.
        audio_start_token_id (`int`, *optional*, defaults to 151645):
            The audio start token index.
        audio_end_token_id (`int`, *optional*, defaults to 151644):
            The audio end token index.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The vision start token index.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The vision end token index.
        tie_word_embeddings (`bool`, *optional*, defaults to False):
            Whether to tie the word embeddings.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling.
    """

    model_type = "valley_omni"
    sub_configs = {"vision_config": Qwen3VLVisionConfig, "text_config": Qwen3VLTextConfig} # "audio_config": Qwen3OmniMoeAudioEncoderConfig, 
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_encoder_type: str = "qwen3_omni_moe",
        audio_config: Optional[dict] = None,
        sample_rate: int = 16000,
        feature_dim: int = 128,
        connector_hidden_size: int = 8192, # 4096,
        apply_audio_downsample: bool = False,
        tm_rope_sections: Optional[list[int]] = None,
        position_id_per_seconds: int = 25,
        audio_token_id: int = 151675, # 151646 in transformers, reset to A3B default 151675 when combining weights
        audio_start_token_id: int = 151669, # 151645 in transformers, reset to A3B default 151669 when combining weights
        audio_end_token_id: int = 151670, # 151644 in transformers, reset to A3B default 151670 when combining weights
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        tie_word_embeddings: bool = False,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        **kwargs,
    ):
        # Sub-configs
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        # Audio & routing fields
        self.audio_encoder_type = audio_encoder_type
        self.audio_config = audio_config or {}
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim
        self.connector_hidden_size = connector_hidden_size
        self.apply_audio_downsample = apply_audio_downsample
        self.tm_rope_sections = tm_rope_sections or [24, 20, 20]
        self.position_id_per_seconds = position_id_per_seconds
        self.audio_token_id = audio_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        # RoPE parameters for the text trunk
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or rope_parameters
        rope_theta = kwargs.get("rope_theta", 5000000.0)
        standardize_rope_params(self, rope_theta=rope_theta)
        # Qwen3-VL uses MRoPE with interleaving; ignore validation keys accordingly
        rope_config_validation(self, ignore_keys={"mrope_section", "mrope_interleaved"})

        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)

    def get_text_config(self, decoder: bool = False) -> Qwen3VLTextConfig:
        return self.text_config

    def get_vision_config(self) -> Qwen3VLVisionConfig:
        return self.vision_config
