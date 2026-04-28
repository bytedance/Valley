# coding=utf-8

"""
ValleyOmni modeling: extend Qwen3-VL text+vision with an audio branch.

Key points:
- Reuses Qwen3-VL Vision and Text modules without modification
- Integrates an upstream audio encoder (Qwen2.5-Omni or Qwen3-Omni-MoE) via a compatibility wrapper
- Audio features are projected to the text hidden size with an AudioConnector
- Unified MRoPE/TM-RoPE position ids across modalities, following Qwen3-VL conventions
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, is_torchdynamo_compiling
from ..qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)
from ..qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder, Qwen2_5OmniAudioEncoderConfig
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder, Qwen3OmniMoeAudioEncoderConfig
from .configuration_valley_omni import ValleyOmniConfig
from .audio_connector import AudioConnector


@dataclass
class ValleyOmniModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    rope_deltas: Optional[torch.LongTensor] = None

@dataclass
class ValleyOmniCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


class _AudioEncoderCompat(nn.Module):
    """
    Compatibility wrapper for audio encoders to unify the interface:
    - forward(input_features: Tensor[B, F, T], feature_attention_mask: Tensor[B, T]) -> (features: Tensor[B, T', C], lengths: Tensor[B])
    All outputs are padded to the max T' in the batch, with lengths indicating valid tokens.
    """

    def __init__(self, config: ValleyOmniConfig):
        super().__init__()
        self.encoder_type = config.audio_encoder_type
        if self.encoder_type == "qwen2_5_omni":
            ae_cfg = Qwen2_5OmniAudioEncoderConfig(**config.audio_config)
            self.encoder = Qwen2_5OmniAudioEncoder(ae_cfg)
            self.output_dim = ae_cfg.output_dim
        elif self.encoder_type == "qwen3_omni_moe":
            ae_cfg = Qwen3OmniMoeAudioEncoderConfig(**config.audio_config)
            ae_cfg.n_window = 50
            # ae_cfg.output_dim = 2048 # Qwen3Omni default 3584, however in 2048 in model A3B weight of audio encoder
            self.encoder = Qwen3OmniMoeAudioEncoder(ae_cfg)
            self.output_dim = ae_cfg.output_dim
        else:
            raise ValueError(f"Unsupported audio_encoder_type: {self.encoder_type}")

    def forward(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_features: [B, F, T_orig]
            feature_attention_mask: [B, T_orig] with 1 for valid frames
        Returns:
            features_padded: [B, T_max, C]
            lengths: [B] effective valid lengths after CNN (excluding padded tail)
        """

        if isinstance(self.encoder, Qwen3OmniMoeAudioEncoder):
            # B, F, T_total = input_features.shape
            if feature_attention_mask is not None:
                audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
            else:
                audio_feature_lengths = None

            feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
            out = self.encoder(
                input_features=input_features,
                feature_lens=feature_lens,
            )
            seq_flat = out.last_hidden_state  # [N=sum(T'_i), C]
            return seq_flat
        
        else:
            raise RuntimeError("Unknown encoder instance")


class ValleyOmniPreTrainedModel(PreTrainedModel):
    config: ValleyOmniConfig
    base_model_prefix = "model"
    input_modalities = ["image", "video", "audio", "text"]
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True


class ValleyOmniModel(ValleyOmniPreTrainedModel):
    base_model_prefix = "model"
    config: ValleyOmniConfig

    def __init__(self, config: ValleyOmniConfig):
        super().__init__(config)
        # Reuse Qwen3-VL submodules
        self.visual = Qwen3VLVisionModel._from_config(config.get_vision_config())
        self.language_model = Qwen3VLTextModel._from_config(config.get_text_config())
        # Audio branch: upstream encoder + connector
        self.audio_encoder = _AudioEncoderCompat(config)
        self.audio_connector = AudioConnector(
            input_dim=self.audio_encoder.output_dim,
            output_dim=config.get_text_config().hidden_size,
            hidden_dim=config.connector_hidden_size,
            downsample=config.apply_audio_downsample,
        )
        self.rope_deltas = None
        self.spatial_merge_size = config.get_vision_config().spatial_merge_size
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Extend Qwen3-VL placeholder mask logic to include audio tokens.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
            special_audio_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            ).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id
            special_audio_mask = input_ids == self.config.audio_token_id

        # special_image_mask = input_ids == self.config.image_token_id
        # special_video_mask = input_ids == self.config.video_token_id
        # special_audio_mask = input_ids == self.config.audio_token_id
        
        # expand to match embedding shape
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

        # Validate lengths if features were provided
        if image_features is not None:
            n_image_tokens = (special_image_mask[...,0]).sum()
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
                )
        if video_features is not None:
            n_video_tokens = (special_video_mask[...,0]).sum()
            if inputs_embeds[special_video_mask].numel() != video_features.numel():
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
                )
        if audio_features is not None:
            n_audio_tokens = (special_audio_mask[...,0]).sum()
            if inputs_embeds[special_audio_mask].numel() != audio_features.numel():
                raise ValueError(
                    f"Audio features and audio tokens do not match: tokens: {n_audio_tokens}, features {audio_features.shape[0]}"
                )

        return special_image_mask, special_video_mask, special_audio_mask

    def get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: list[torch.Tensor],
        grid_hs: list[torch.Tensor],
        grid_ws: list[torch.Tensor],
    ):
        llm_pos_ids_list = []
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
        t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
        _llm_pos_ids = torch.stack([t_index, h_index, w_index])
        llm_pos_ids_list.append(_llm_pos_ids + start_idx)
        llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
        return llm_pos_ids


    def get_rope_index_qwen3omni(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
        audio_seqlens: Optional[torch.LongTensor] = None,
        second_per_grids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            use_audio_in_video (`bool`, *optional*):
                 If set to `True`, use the audio in video.
            audio_seqlens (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
            second_per_grids (`torch.LongTensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        vision_start_token_id = self.config.vision_start_token_id
        audio_start_token_id = self.config.audio_start_token_id
        position_id_per_seconds = self.config.position_id_per_seconds

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is not None:
                attention_mask = attention_mask == 1
            position_ids = torch.zeros(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=torch.float,
                device=input_ids.device,
            )
            image_idx, video_idx, audio_idx = 0, 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i]]
                image_nums, video_nums, audio_nums = 0, 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                audio_nums = torch.sum(input_ids == audio_start_token_id)
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (
                    (vision_tokens == audio_start_token_id).sum()
                    if use_audio_in_video
                    else (vision_tokens == video_token_id).sum()
                )
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
                multimodal_nums = (
                    image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums
                )
                for _ in range(multimodal_nums):
                    # st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    # bug fix for RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
                    try:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    except RuntimeError as e:
                        import warnings
                        print(input_ids)
                        st_idx, llm_pos_ids_list = 0, []
                        warnings.warn("RuntimeError", category=UserWarning, stacklevel=2)

                    if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                        remain_videos > 0 or remain_images > 0
                    ):
                        ed_vision_start = input_tokens.index(vision_start_token_id, st)
                    else:
                        ed_vision_start = len(input_tokens) + 1
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio_start = input_tokens.index(audio_start_token_id, st)
                    else:
                        ed_audio_start = len(input_tokens) + 1
                    min_ed = min(ed_vision_start, ed_audio_start)

                    text_len = min_ed - st
                    if text_len != 0:
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                        st_idx += text_len
                    # Audio in Video
                    if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                        bos_len, eos_len = 2, 2
                    else:
                        bos_len, eos_len = 1, 1

                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                    st_idx += bos_len
                    # Audio Only
                    if min_ed == ed_audio_start:
                        audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx]).to(input_ids.device)
                        llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + audio_len + eos_len)
                        audio_idx += 1
                        remain_audios -= 1

                    # Image Only
                    elif min_ed == ed_vision_start and input_ids[ed_vision_start + 1] == image_token_id:
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).float()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                        image_len = image_len.to(input_ids.device)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + image_len + eos_len)
                        image_idx += 1
                        remain_images -= 1

                    # Video Only
                    elif min_ed == ed_vision_start and input_ids[ed_vision_start + 1] == video_token_id:
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).float()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                        video_len = video_len.to(input_ids.device)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + video_len + eos_len)
                        video_idx += 1
                        remain_videos -= 1

                    # Audio in Video
                    elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                        audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx]).to(input_ids.device)
                        audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).float()
                        video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_data_index, audio_data_index = 0, 0
                        while (
                            video_data_index < video_llm_pos_ids.shape[-1]
                            and audio_data_index < audio_llm_pos_ids.shape[-1]
                        ):
                            if video_llm_pos_ids[0][video_data_index] <= audio_llm_pos_ids[0][audio_data_index]:
                                llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index : video_data_index + 1])
                                video_data_index += 1
                            else:
                                llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1])
                                audio_data_index += 1
                        if video_data_index < video_llm_pos_ids.shape[-1]:
                            llm_pos_ids_list.append(
                                video_llm_pos_ids[:, video_data_index : video_llm_pos_ids.shape[-1]]
                            )
                        if audio_data_index < audio_llm_pos_ids.shape[-1]:
                            llm_pos_ids_list.append(
                                audio_llm_pos_ids[:, audio_data_index : audio_llm_pos_ids.shape[-1]]
                            )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                        video_len = video_len.to(input_ids.device)
                        st += int(text_len + bos_len + audio_len + video_len + eos_len)

                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1
                    # st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    try:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    except RuntimeError as e:
                        import warnings
                        print(input_ids)
                        st_idx, llm_pos_ids_list = 0, []
                        warnings.warn("RuntimeError", category=UserWarning, stacklevel=2)
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                if st < len(input_tokens):
                    # st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    try:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    except RuntimeError as e:
                        import warnings
                        print(input_ids)
                        st_idx, llm_pos_ids_list = 0, []
                        warnings.warn("RuntimeError", category=UserWarning, stacklevel=2)
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat([item.float() for item in llm_pos_ids_list], dim=1).reshape(3, -1)

                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

            return position_ids, mrope_position_deltas
        else:
            position_ids = attention_mask.float().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

            return position_ids, mrope_position_deltas


    def get_rope_index_qwen3vl(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_seqlens: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 3D position ids [T,H,W] for text+vision+audio, following Qwen3-VL conventions for vision,
        and simple 1D repeated across 3 dims for audio.
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        audio_token_id = self.config.audio_token_id
        audio_start_token_id = self.config.audio_start_token_id

        # Modality presence guards
        has_image = image_grid_thw is not None
        has_video = video_grid_thw is not None
        # Guard: whether audio lengths are available
        has_audio = audio_seqlens is not None

        # Align with Qwen3-VL: split video_grid_thw by timestamps so each frame has t=1
        if has_video:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None or audio_seqlens is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index, audio_index = 0, 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, ids in enumerate(total_input_ids):
                ids = ids[attention_mask[i] == 1]
                input_tokens = ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                # Count segments using vision_start for vision and audio_start for audio
                vision_start_indices = torch.argwhere(ids == vision_start_token_id).squeeze(1)
                vision_tokens = ids[vision_start_indices + 1] if vision_start_indices.numel() > 0 else ids.new_empty(0)
                audio_start_indices = torch.argwhere(ids == audio_start_token_id).squeeze(1)
                audio_tokens = ids[audio_start_indices + 1] if audio_start_indices.numel() > 0 else ids.new_empty(0)

                image_nums = (vision_tokens == image_token_id).sum().item() if has_image else 0
                video_nums = (vision_tokens == video_token_id).sum().item() if has_video else 0
                # audio_nums = (ids == audio_start_token_id).sum().item() if has_audio else 0
                audio_nums = (audio_tokens == audio_token_id).sum().item() if has_audio else 0

                remain_images = int(image_nums)
                remain_videos = int(video_nums)
                remain_audios = int(audio_nums)

                while st < len(input_tokens):
                    # find next segment indices
                    next_image = input_tokens.index(image_token_id, st) if (has_image and image_token_id in input_tokens and remain_images > 0) else len(input_tokens) + 1
                    next_video = input_tokens.index(video_token_id, st) if (has_video and video_token_id in input_tokens and remain_videos > 0) else len(input_tokens) + 1
                    # next_audio = input_tokens.index(audio_start_token_id, st) if (has_audio and audio_start_token_id in input_tokens and remain_audios > 0) else len(input_tokens) + 1
                    next_audio = input_tokens.index(audio_token_id, st) if (has_audio and audio_token_id in input_tokens and remain_audios > 0) else len(input_tokens) + 1

                    min_ed = min(next_image, next_video, next_audio)
                    if min_ed == len(input_tokens) + 1:
                        # remaining text till end
                        text_len = len(input_tokens) - st
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                        st = len(input_tokens)
                        break

                    # pre-text before the segment
                    text_len = max(min_ed - st, 0)
                    if text_len > 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    # segment dispatch
                    if has_image and min_ed == next_image:
                        # image segment
                        t, h, w = (
                            int(image_grid_thw[image_index][0].item()),
                            int(image_grid_thw[image_index][1].item()),
                            int(image_grid_thw[image_index][2].item()),
                        )
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t,
                            h // spatial_merge_size,
                            w // spatial_merge_size,
                        )
                        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)
                        st = next_image + llm_grid_t * llm_grid_h * llm_grid_w
                        image_index += 1
                        remain_images -= 1
                    elif has_video and min_ed == next_video:
                        # video segment (video_grid_thw has been frame-split above)
                        t, h, w = (
                            int(video_grid_thw[video_index][0].item()),
                            int(video_grid_thw[video_index][1].item()),
                            int(video_grid_thw[video_index][2].item()),
                        )
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t,
                            h // spatial_merge_size,
                            w // spatial_merge_size,
                        )
                        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)
                        st = next_video + llm_grid_t * llm_grid_h * llm_grid_w
                        video_index += 1
                        remain_videos -= 1
                    elif has_audio and min_ed == next_audio:
                        # audio segment: [audio_start] + audio_len tokens + [audio_end]
                        bos_len, eos_len = 1, 1
                        # BOS
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                        st_idx = llm_pos_ids_list[-1].max() + 1
                        # audio tokens
                        audio_len = int(audio_seqlens[audio_index].item())
                        audio_llm_pos = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        llm_pos_ids_list.append(audio_llm_pos)
                        st_idx = llm_pos_ids_list[-1].max() + 1
                        # EOS
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)
                        st = next_audio + bos_len + audio_len + eos_len
                        audio_index += 1
                        remain_audios -= 1

                # append remaining text positions (if any)
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype
                )
            return position_ids, mrope_position_deltas


    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        # Same implementation as for images
        return self.get_image_features(pixel_values_videos, video_grid_thw)


    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds
    

    def get_audio_features(self, input_features: torch.Tensor, feature_attention_mask: torch.Tensor):
        """
        Encodes audio into continuous embeddings that can be forwarded to the language model.
        Returns flattened features [N_audio_tokens, hidden_size] and per-sample lengths.
        """
        seqs = self.audio_encoder(input_features=input_features, feature_attention_mask=feature_attention_mask)
        audio_embeds = self.audio_connector(seqs)
        return audio_embeds

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_audio_in_video: Optional[bool] = False,
        video_second_per_grid: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, ValleyOmniModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_dim, frames)`, *optional*):
            Audio log-mel features.
        feature_attention_mask (`torch.LongTensor` of shape `(batch_size, frames)`, *optional*):
            1 for valid frames, 0 for padding.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        audio_mask = None

        # Audio branch
        if input_features is not None:
            audio_embeds = self.get_audio_features(input_features, feature_attention_mask)

            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, audio_features=audio_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

        if pixel_values is not None:
            # image_embeds, deepstack_image_embeds = Qwen3VLModel.get_image_features(self, pixel_values, image_grid_thw)
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = Qwen3VLModel.get_placeholder_mask(
                self, input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            # video_embeds, deepstack_video_embeds = Qwen3VLModel.get_video_features(self, pixel_values_videos, video_grid_thw)
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = Qwen3VLModel.get_placeholder_mask(
                self, input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        deepstack_visual_embeds = None
        visual_pos_masks = None
        if image_mask is not None and video_mask is not None:
            image_mask_b = image_mask[..., 0]
            video_mask_b = video_mask[..., 0]
            visual_pos_masks = image_mask_b | video_mask_b
            deepstack_visual_embeds = []
            image_mask_joint = image_mask_b[visual_pos_masks]
            video_mask_joint = video_mask_b[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            deepstack_visual_embeds = deepstack_video_embeds

        # ---------
        # Begin: Position ids for Qwen3-VL
        # if position_ids is None:
        #     attention_mask_tensor = (
        #         attention_mask if not isinstance(attention_mask, dict) else attention_mask.get("full_attention", None)
        #     )
        #     if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
        #         attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
        #         if attention_mask_tensor.dtype.is_floating_point:
        #             attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
        #             attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        #     prefill_compiled_stage = is_torchdynamo_compiling() and (
        #         (input_ids is not None and input_ids.shape[1] != 1)
        #         or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        #     )
        #     prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
        #         (cache_position is not None and cache_position[0] == 0)
        #         or (past_key_values is None or past_key_values.get_seq_length() == 0)
        #     )
        #     if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
        #         position_ids, rope_deltas = self.get_rope_index_qwen3vl(
        #             input_ids,
        #             image_grid_thw,
        #             video_grid_thw,
        #             attention_mask=attention_mask_tensor,
        #             audio_seqlens=audio_lengths,
        #         )
        #         self.rope_deltas = rope_deltas
        #     else:
        #         batch_size, seq_length, _ = inputs_embeds.shape
        #         delta = (
        #             (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
        #             if cache_position is not None
        #             else 0
        #         )
        #         position_ids = torch.arange(seq_length, device=inputs_embeds.device)
        #         position_ids = position_ids.view(1, -1).expand(batch_size, -1)
        #         if cache_position is not None:
        #             delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
        #         position_ids = position_ids.add(delta)
        #         position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        # End: Position ids for Qwen3-VL
        # ---------
        
        if feature_attention_mask is not None:
            audio_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_lengths = None

        # ---------
        # Begin: Position ids for Qwen3-Omni
        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index_qwen3omni(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video=use_audio_in_video,
                    audio_seqlens=audio_lengths,
                    second_per_grids=video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        # End: Position ids for Qwen3-Omni
        # ---------

        # print("INPUTS_EMBEDS:\n")
        # print(inputs_embeds)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return ValleyOmniModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


class ValleyOmniForConditionalGeneration(ValleyOmniPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config: ValleyOmniConfig

    def __init__(self, config: ValleyOmniConfig):
        super().__init__(config)
        self.model = ValleyOmniModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation.

        - In the prefill step (when cache_position[0] == 0), pass visual and audio features.
        - In subsequent decoding steps, only pass the last token and drop visual/audio features to keep the cache consistent.
        - Force position_ids to None; they are computed inside forward based on rope_deltas.
        """
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )

        # Position ids are computed inside forward for ValleyOmni
        model_inputs["position_ids"] = None

        # Only pass visual/audio features at the first step; drop them afterwards to avoid cache inconsistency
        cp = model_inputs.get("cache_position", None)
        if cp is not None and cp[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["image_grid_thw"] = None
            model_inputs["video_grid_thw"] = None
            model_inputs["input_features"] = None
            model_inputs["feature_attention_mask"] = None

        return model_inputs

    def can_generate(self) -> bool:
        return True

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_audio_in_video: Optional[bool] = False,
        video_second_per_grid: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, ValleyOmniCausalLMOutputWithPast]: # BaseModelOutputWithPast

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            cache_position=cache_position,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state # [B, T, H]
        logits = self.lm_head(hidden_states) # [B, T, V]

        # return BaseModelOutputWithPast(
        #     last_hidden_state=logits,
        #     past_key_values=outputs.past_key_values,
        # )
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return ValleyOmniCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )
