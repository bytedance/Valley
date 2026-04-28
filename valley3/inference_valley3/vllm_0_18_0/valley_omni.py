# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM integration for the custom ValleyOmni multimodal model.

This implementation follows the existing Qwen3-VL and Qwen3-Omni-Thinker patterns:

- Text + vision path is based on the Qwen3-VL implementation
  (``Qwen3_VisionTransformer`` and ``Qwen3LLMForCausalLM``).
- Audio path mirrors Qwen3-Omni-Thinker: audio features
  are produced by an upstream audio encoder and then projected to the
  language model hidden size by an ``AudioConnector``.
- Multimodal processing is wired via ``MULTIMODAL_REGISTRY`` and a
  dedicated ``ValleyOmniMultiModalProcessor`` which relies on the
  HuggingFace-side ``ValleyOmniProcessor`` to produce model inputs.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, Optional, Union
from functools import partial
from itertools import islice
import numpy as np

import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers import PretrainedConfig
from transformers import __version__ as TRANSFORMERS_VERSION
from packaging.version import Version
from transformers.models.valley_omni.configuration_valley_omni import ValleyOmniConfig
from transformers.models.valley_omni.processing_valley_omni import ValleyOmniProcessor
from transformers.models.whisper import WhisperFeatureExtractor
# from transformers.models.valley_omni.modeling_valley_omni import _AudioEncoderCompat as ValleyOmniAudioEncoder
# from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig as ValleyOmniAudioEncoderConfig
# from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder as ValleyOmniAudioEncoder
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder, Qwen2_5OmniAudioEncoderConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder, Qwen3OmniMoeAudioEncoderConfig
from transformers.models.valley_omni.audio_connector import AudioConnector as ValleyOmniAudioConnector

from .qwen2_audio import Qwen2AudioProcessingInfo
from .qwen2_5_vl import Qwen2_5_VLProcessingInfo
from .qwen2_5_omni_thinker import Qwen2_5OmniThinkerMultiModalProcessor
from .qwen3_vl import Qwen3_VisionTransformer, Qwen3LLMForCausalLM

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    MultiModalFeatureSpec
)
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
    BaseDummyInputsBuilder
)
# from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.distributed import get_pp_group

from .interfaces import MultiModalEmbeddings, SupportsLoRA, SupportsMultiModal, SupportsPP, SupportsMRoPE
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix, _merge_multimodal_embeddings
from .vision import (
    get_llm_pos_ids_for_vision,
    get_vit_attn_backend,
    run_dp_sharded_mrope_vision_model
)
logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Audio feature schema and helpers
# ---------------------------------------------------------------------------

class ValleyOmniAudioEncoder(nn.Module):
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
        feature_lens: torch.LongTensor,
    ):
        """
        Args:
            input_features: [B, F, T_orig]
            feature_attention_mask: [B, T_orig] with 1 for valid frames
        Returns:
            features_padded: [B, T_max, C]
            lengths: [B] effective valid lengths after CNN (excluding padded tail)
        """

        if isinstance(self.encoder, Qwen3OmniMoeAudioEncoder):
            out = self.encoder(
                input_features=input_features,
                feature_lens=feature_lens,
            )
            seq_flat = out.last_hidden_state
            return seq_flat
        
        else:
            raise RuntimeError("Unknown encoder instance")


class ValleyOmniAudioFeatureInputs(TensorSchema):
    """
    Dimensions:
        - na: Number of audios
        - nmb: Number of mel bins
        - msl: Maximum sequence length (frames)
        - tsl: Total sequence length (frames)
    """

    type: Literal["audio_features"]
    input_features: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nmb", "tsl", dynamic_dims={"tsl"}),
    ]

    audio_feature_lengths: Annotated[torch.Tensor, TensorShape("na")]

    feature_attention_mask: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("na", "msl", dynamic_dims={"msl"}),
    ]


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Matches ValleyOmni's internal feature length computation.

    The HF implementation chunks very long audios and applies several
    convolutional subsampling stages. We reuse the same formula so that
    the number of audio tokens matches the tokenizer-side prompt
    expansion.
    """

    # Same as valley_omni.processing_valley_omni._get_feat_extract_output_lengths
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (
        input_lengths // 100
    ) * 13
    return feat_lengths, output_lengths



# ---------------------------------------------------------------------------
# Processing info & dummy inputs
# ---------------------------------------------------------------------------

# Adapted from qwen3_omni_moe_thinker
class ValleyOmniProcessingInfo(Qwen2AudioProcessingInfo, Qwen2_5_VLProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(ValleyOmniConfig)

    def get_hf_processor(self, **kwargs: object) -> ValleyOmniProcessor:
        processor = self.ctx.get_hf_processor(
            ValleyOmniProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )
        if not hasattr(processor, "audio_token"):
            processor.audio_token = "<|audio_pad|>"
        if not hasattr(processor, "image_token"):
            processor.image_token = "<|image_pad|>"
        if not hasattr(processor, "video_token"):
            processor.video_token = "<|video_pad|>"
        return processor

    def get_feature_extractor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None, "image": None, "video": None}


# Adapted from qwen3_omni_thinker
class ValleyOmniDummyInputsBuilder(
        BaseDummyInputsBuilder[ValleyOmniProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        hf_processor = self.info.get_hf_processor()

        audio_token: str = hf_processor.audio_token
        image_token: str = hf_processor.image_token
        video_token: str = hf_processor.video_token

        return (
            audio_token * num_audios
            + image_token * num_images
            + video_token * num_videos
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        feature_extractor = self.info.get_feature_extractor()

        target_audio_length = (
            min(
                feature_extractor.chunk_length,
                30,
            )
            * feature_extractor.sampling_rate
        )
        target_width, target_height = self.info.get_image_size_with_most_features()
        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts
        )

        image_overrides = mm_options.get("image") if mm_options else None
        video_overrides = mm_options.get("video") if mm_options else None
        audio_overrides = mm_options.get("audio") if mm_options else None

        mm_data = {
            "audio": self._get_dummy_audios(
                length=target_audio_length,
                num_audios=num_audios,
                overrides=audio_overrides,
            ),
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
            "video": self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=target_num_frames,
                num_videos=num_videos,
                overrides=video_overrides,
            ),
        }

        return mm_data

# ---------------------------------------------------------------------------
# Multi-modal processor
# ---------------------------------------------------------------------------


class ValleyOmniMultiModalProcessor(
    Qwen2_5OmniThinkerMultiModalProcessor
):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np.ndarray:
            length = x.shape[-1]
            if length % hop_length != 0:
                pad_length = hop_length - (length % hop_length)
                x = np.pad(x, (0, pad_length), mode="constant", constant_values=0)
            return x

        # NOTE: WhisperFeatureExtractor cannot handle empty list of audios
        feature_extractor = self.info.get_feature_extractor()
        hop_length = feature_extractor.hop_length
        if audios:
            # NOTE: Qwen3-Omni processor accept "audio"
            # To make sure the cache works with padding=True, we pre-padded
            # the audio to multiple of hop_length.
            mm_data["audio"] = [
                pad_to_hop_length(audio, hop_length)
                if isinstance(audio, np.ndarray)
                else (pad_to_hop_length(audio[0], hop_length), audio[1])
                for audio in audios
            ]

            # TODO(Isotr0py): Remove this patch after upstream fix PR
            # released and Transformers version update:
            # https://github.com/huggingface/transformers/pull/41473
            mm_kwargs = dict(mm_kwargs)
            tok_kwargs = dict(tok_kwargs)
            if Version(TRANSFORMERS_VERSION) < Version("4.58.0"):
                # move truncation to audio_kwargs level to avoid conflict
                # with tok_kwargs
                mm_kwargs["audio_kwargs"] = {
                    "truncation": mm_kwargs.pop("truncation", False)
                }
                mm_kwargs["text_kwargs"] = {
                    "truncation": tok_kwargs.pop("truncation", False)
                }

        hf_inputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        if (
            "audio_feature_lengths" in hf_inputs
            and "feature_attention_mask" in hf_inputs
            and (audios := mm_data.get("audio", []))
        ):
            audio_num_frames = []
            for _, audio in enumerate(audios):
                audio_length = len(audio[0]) if isinstance(audio, tuple) else len(audio)
                num_frame = (
                    (audio_length // hop_length)
                    if audio_length % hop_length == 0
                    else (audio_length // hop_length - 1)
                )
                if mm_kwargs.get("truncation", False):
                    num_frame = min(
                        num_frame, feature_extractor.n_samples // hop_length
                    )
                audio_num_frames.append(num_frame)
            hf_inputs["feature_attention_mask"] = [
                torch.ones(num_frame) for num_frame in audio_num_frames
            ]
            hf_inputs["audio_feature_lengths"] = torch.tensor(audio_num_frames)
        
        return hf_inputs

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], str, Mapping[str, list[PlaceholderFeaturesInfo]]]:
        """
        Qwen3-Omni reimplements this function to handle `use_audio_in_video`.
        """
        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)

        use_audio_in_video = False
        if "video" in mm_kwargs:
            for item in mm_kwargs["video"]:
                if item and item["use_audio_in_video"].data:
                    use_audio_in_video = True
                else:
                    use_audio_in_video = False

        # normal case with `use_audio_in_video=False`
        if is_update_applied:
            mm_placeholders = self._find_mm_placeholders(
                prompt_ids,
                mm_prompt_updates,
            )
            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts,
            )
        else:
            if use_audio_in_video and "audio" in mm_prompt_updates:
                filtered_updates = {
                    k: v for k, v in mm_prompt_updates.items() if k != "audio"
                }
                prompt_ids, mm_placeholders = self._apply_prompt_updates(
                    prompt_ids,
                    filtered_updates,
                )
                # Derive audio placeholders from video placeholders
                mm_placeholders = self._derive_audio_from_video_placeholders(
                    mm_placeholders, mm_prompt_updates
                )
            else:
                prompt_ids, mm_placeholders = self._apply_prompt_updates(
                    prompt_ids,
                    mm_prompt_updates,
                )

            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts,
            )

        return prompt_ids, mm_placeholders

    def get_updates_use_audio_in_video(
        self,
        thinker_config: PretrainedConfig,
        audio_len: int,
        video_grid_thw: list[int] | torch.Tensor,
        video_second_per_grid_t: float,
    ) -> list[int]:
        shift = 0
        audio_token_id = thinker_config.audio_token_id
        video_token_id = thinker_config.video_token_id
        audio_start_token_id = thinker_config.audio_start_token_id
        audio_end_token_id = thinker_config.audio_end_token_id
        spatial_merge_size = thinker_config.vision_config.spatial_merge_size
        position_id_per_seconds = thinker_config.position_id_per_seconds
        audio_token_indices = np.arange(next(iter([audio_len])))
        curr_video_grid_thw = next(iter([video_grid_thw]))
        height = curr_video_grid_thw[1] // spatial_merge_size
        width = curr_video_grid_thw[2] // spatial_merge_size
        video_token_indices = np.arange(curr_video_grid_thw[0]).reshape(-1, 1, 1)
        video_token_indices = np.broadcast_to(
            video_token_indices, (video_token_indices.shape[0], height, width)
        ).reshape(-1)
        video_token_indices = (
            (video_token_indices + shift)
            * next(iter([video_second_per_grid_t]))
            * position_id_per_seconds
        )
        video_data_index, audio_data_index = 0, 0
        updates = [audio_start_token_id]
        while video_data_index < len(video_token_indices) and audio_data_index < len(
            audio_token_indices
        ):
            if (
                video_token_indices[video_data_index]
                <= audio_token_indices[audio_data_index]
            ):
                updates += [video_token_id]
                video_data_index += 1
            else:
                updates += [audio_token_id]
                audio_data_index += 1
        if video_data_index < len(video_token_indices):
            updates += [video_token_id] * (len(video_token_indices) - video_data_index)
        if audio_data_index < len(audio_token_indices):
            updates += [audio_token_id] * (len(audio_token_indices) - audio_data_index)
        updates += [audio_end_token_id]
        return updates

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        vocab = tokenizer.get_vocab()

        audio_token = processor.audio_token
        image_token = processor.image_token
        video_token = processor.video_token
        audio_token_id = vocab[audio_token]
        image_token_id = vocab[image_token]
        video_token_id = vocab[video_token]

        out_mm_data = out_mm_kwargs.get_data()
        audio_feature_lengths = out_mm_data.get("audio_feature_lengths")
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        if audio_feature_lengths is None and feature_attention_mask is None:
            audio_output_lengths = []
        elif audio_feature_lengths is not None:
            _, audio_output_lens = _get_feat_extract_output_lengths(
                audio_feature_lengths
            )
            audio_output_lengths = audio_output_lens.tolist()
        elif feature_attention_mask is not None:
            assert isinstance(feature_attention_mask, torch.Tensor)
            _, audio_output_lens = _get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
            audio_output_lengths = audio_output_lens.tolist()

        # number of audios read from video.
        audio_in_video_item_idx = 0
        audio_item_idx = 0

        def get_replacement_qwen2_audio(item_idx: int):
            nonlocal audio_item_idx
            item_idx += audio_in_video_item_idx

            audio_item_idx += 1

            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio = audios.get(item_idx)
                raise ValueError(
                    f"The audio {audio} (len={len(audio)}) is too short "
                    "to be represented inside the model"
                )

            return [audio_token_id] * num_features

        def get_replacement_qwen2_vision(item_idx: int, modality: str):
            grid_thw = out_mm_data[f"{modality}_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)
            merge_length = image_processor.merge_size**2

            token_id = image_token_id if modality == "image" else video_token_id
            return [token_id] * (int(grid_thw.prod()) // merge_length)

        use_audio_in_video = hf_processor_mm_kwargs.get("use_audio_in_video", False)
        thinker_config = self.info.get_hf_config()

        def get_replacement_qwen2_use_audio_in_video(item_idx: int):
            nonlocal audio_in_video_item_idx
            audio_num_features = audio_output_lengths[
                audio_in_video_item_idx + item_idx
            ]
            video_grid_thw = out_mm_data["video_grid_thw"][item_idx]

            audio_in_video_item_idx += 1

            second_per_grid_ts = hf_processor_mm_kwargs.get("second_per_grid_ts", None)
            if second_per_grid_ts:
                video_second_per_grid_t = second_per_grid_ts[item_idx]
            else:
                video_second_per_grid_t = 2.0

            placeholder = self.get_updates_use_audio_in_video(
                thinker_config=thinker_config,
                audio_len=audio_num_features,
                video_grid_thw=video_grid_thw,
                video_second_per_grid_t=video_second_per_grid_t,
            )
            return PromptUpdateDetails.select_token_id(
                placeholder, embed_token_id=video_token_id
            )

        video_replacement_fn = (
            get_replacement_qwen2_use_audio_in_video
            if use_audio_in_video
            else partial(get_replacement_qwen2_vision, modality="video")
        )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_qwen2_audio,
            ),
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=partial(get_replacement_qwen2_vision, modality="image"),
            ),
            PromptReplacement(
                modality="video",
                target=video_token,
                replacement=video_replacement_fn,
            ),
        ]

    def _derive_audio_from_video_placeholders(
        self,
        placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
        """
        Helper to derive audio placeholders from video placeholders when
        use_audio_in_video=True.
        """
        if "video" not in placeholders:
            return placeholders

        # Validate audio and video counts match
        num_videos = len(placeholders["video"])
        num_audios = len(mm_prompt_updates.get("audio", []))
        if num_audios != num_videos:
            raise ValueError(
                f"use_audio_in_video requires equal number of audio and video items, "
                f"got {num_audios=}, {num_videos=}"
            )

        tokenizer = self.info.get_tokenizer()
        processor = self.info.get_hf_processor()
        audio_token_id = tokenizer.get_vocab()[processor.audio_token]

        result_placeholders = dict(placeholders)
        audio_placeholders = []

        # Each video is paired with one audio
        for video_idx, video_placeholder in enumerate(placeholders["video"]):
            # Create is_embed mask selecting only audio tokens
            audio_is_embed = torch.tensor(video_placeholder.tokens) == audio_token_id

            audio_placeholder = PlaceholderFeaturesInfo(
                modality="audio",
                item_idx=video_idx,
                start_idx=video_placeholder.start_idx,
                tokens=video_placeholder.tokens,
                is_embed=audio_is_embed,
            )
            audio_placeholders.append(audio_placeholder)

        result_placeholders["audio"] = audio_placeholders
        return result_placeholders

    def _validate_mm_placeholders(
        self,
        mm_placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
        mm_item_counts: Mapping[str, int],
    ) -> None:
        BaseMultiModalProcessor[
            ValleyOmniProcessingInfo
        ]._validate_mm_placeholders(self, mm_placeholders, mm_item_counts)

    def _get_raw_input_ids(
        self,
        token_ids: list[int],
        use_audio_in_video: bool = False,
    ) -> list[int]:
        tokenizer = self.info.get_tokenizer()
        vision_bos_token = tokenizer.encode(tokenizer.vision_bos_token)[0]
        vision_eos_token = tokenizer.encode(tokenizer.vision_eos_token)[0]
        audio_bos_token = tokenizer.encode(tokenizer.audio_bos_token)[0]
        audio_eos_token = tokenizer.encode(tokenizer.audio_eos_token)[0]
        audio_token = tokenizer.encode("<|audio_pad|>")[0]
        image_token = tokenizer.encode("<|image_pad|>")[0]
        video_token = tokenizer.encode("<|video_pad|>")[0]

        result = token_ids[:]
        if use_audio_in_video:
            while True:
                start = None
                for i in range(len(result) - 1):
                    if result[i : i + 2] == [vision_bos_token, audio_bos_token]:
                        start = i
                        break
                if start is not None:
                    end = None
                    for i in range(start + 2, len(result) - 1):
                        if result[i : i + 2] == [audio_eos_token, vision_eos_token]:
                            end = i
                            break
                    if end is not None:
                        result = (
                            result[:start]
                            + [vision_bos_token, video_token, vision_eos_token]
                            + result[end + 2 :]
                        )
                else:
                    break

        for mm_token in [audio_token, image_token, video_token]:
            compressed = []
            for x in result:
                if x != mm_token or (not compressed or compressed[-1] != mm_token):
                    compressed.append(x)
            result = compressed

        return result

# ---------------------------------------------------------------------------
# ValleyOmni model (vLLM-side)
# ---------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    ValleyOmniMultiModalProcessor,
    info=ValleyOmniProcessingInfo,
    dummy_inputs=ValleyOmniDummyInputsBuilder,
)
class ValleyOmniForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsLoRA, SupportsPP, SupportsMRoPE
):
    """vLLM implementation of ValleyOmni.

    This class is intentionally lightweight and focuses on composing
    existing vLLM modules:

    - ``Qwen3_VisionTransformer`` for images / videos
    - ``Qwen3LLMForCausalLM`` for the language backbone
    - A simple audio encoder + connector that project audio features to
      the language hidden size
    """

    # Packed projection mappings for weight loading (same as Qwen3-VL).
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    supports_encoder_tp_data = True
    merge_by_field_config = True
    
    # Map HF weights to vLLM submodules.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Visual / language parts are the same as Qwen3-VL.
            "model.visual.": "visual.",
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
            # Audio encoder / connector live under ``model.`` in HF.
            "model.audio_encoder.": "audio_encoder.",
            "model.audio_connector.": "audio_connector.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            # Qwen3-style vision wrappers.
            return "<|vision_start|><|IMAGE|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|VIDEO|><|vision_end|>"
        if modality.startswith("audio"):
            return f"Audio {i}: <|audio_bos|><|AUDIO|><|audio_eos|>"

        raise ValueError("Unsupported modality: {modality!r}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__()
        hf_config: ValleyOmniConfig = vllm_config.model_config.hf_config
        quant_config: QuantizationConfig = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = hf_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        # -------------------- Visual tower --------------------
        if multimodal_config.get_limit_per_prompt("image") or multimodal_config.get_limit_per_prompt(
            "video"
        ):
            self.visual = Qwen3_VisionTransformer(
                hf_config.get_vision_config(),
                norm_eps=getattr(hf_config.get_text_config(), "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                # use_data_parallel=self.use_data_parallel,
            )
        else:
            self.visual = None

        # Deepstack (multi-scale vision features).
        self.use_deepstack = hasattr(hf_config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(hf_config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        if self.use_deepstack and self.visual is not None:
            self.deepstack_input_embeds: list[torch.Tensor] = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    hf_config.get_text_config().hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
        else:
            self.deepstack_input_embeds = None
        self.visual_dim = hf_config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        # -------------------- Audio path --------------------
        audio_limit = multimodal_config.get_limit_per_prompt("audio")
        self.audio_encoder: Optional[ValleyOmniAudioEncoder]
        self.audio_connector: Optional[ValleyOmniAudioConnector]

        self.audio_encoder = None
        self.audio_connector = None

        if audio_limit != 0:
            encoder_type = getattr(hf_config, "audio_encoder_type", None)

            if encoder_type == "qwen3_omni_moe":
                # HF-like audio encoder: Qwen3-Omni-MoE.
                # ae_cfg = ValleyOmniAudioEncoderConfig(hf_config.audio_config)
                self.audio_encoder = ValleyOmniAudioEncoder(hf_config)
                audio_hidden_in = self.audio_encoder.output_dim
                audio_hidden_out = hf_config.get_text_config().hidden_size
                # NOTE: The submodule names in ValleyOmniAudioConnector
                # (proj_in/proj_mid/proj_out/ln) are aligned with the
                # transformers-side AudioConnector so that weights can be
                # loaded without renaming.
                self.audio_connector = ValleyOmniAudioConnector(
                    input_dim=audio_hidden_in,
                    output_dim=audio_hidden_out,
                    hidden_dim=hf_config.connector_hidden_size,
                    downsample=getattr(hf_config, "apply_audio_downsample", False),
                )
            else:
                # For other encoder types (e.g. ``qwen2_5_omni``) we
                # currently do not instantiate an audio branch. The
                # corresponding weights will be skipped during loading to
                # avoid shape mismatches.
                self.audio_encoder = None
                self.audio_connector = None

        # -------------------- Language model --------------------
        self.language_model = Qwen3LLMForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )


    def _get_deepstack_input_embeds(self, num_tokens: int) -> IntermediateTensors:
        # get deepstack_input_embeds from buffer, and clear the buffer
        return IntermediateTensors(
            {
                f"deepstack_input_embeds_{idx}": self.deepstack_input_embeds[idx][
                    :num_tokens
                ]
                for idx in range(self.deepstack_num_level)
            }
        )

    def _set_deepstack_input_embeds(self, deepstack_input_embeds: torch.Tensor) -> None:
        # set deepstack_input_embeds to buffer
        num_tokens = deepstack_input_embeds.size(1)
        if num_tokens > self.deepstack_input_embeds[0].size(0):
            self.deepstack_input_embeds = [
                torch.zeros(
                    num_tokens,
                    self.config.text_config.hidden_size,
                    device=self.deepstack_input_embeds[0].device,
                    dtype=self.deepstack_input_embeds[0].dtype,
                )
                for _ in range(self.deepstack_num_level)
            ]
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[idx][:num_tokens].copy_(
                deepstack_input_embeds[idx]
            )

    def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
        # clear deepstack_input_embeds in buffer
        if num_tokens > 0:
            for idx in range(self.deepstack_num_level):
                self.deepstack_input_embeds[idx][:num_tokens].zero_()


    # ------------------------------------------------------------------
    # Helpers for multimodal inputs
    # ------------------------------------------------------------------

    def _validate_and_reshape_mm_tensor(self, mm_input: object, name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(
                    f"{name} should be 2D or batched 3D tensor. Got ndim: {mm_input.ndim} (shape={mm_input.shape})"
                )
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    # ---- Image / video ----

    def _parse_and_validate_image_input(
        self,
        **kwargs: object,
    ) -> Optional[dict[str, torch.Tensor]]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None:
            return None

        pixel_values = self._validate_and_reshape_mm_tensor(pixel_values, "image pixel values")
        image_grid_thw = self._validate_and_reshape_mm_tensor(image_grid_thw, "image grid_thw")

        return {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def _parse_and_validate_video_input(
        self,
        **kwargs: object,
    ) -> Optional[dict[str, torch.Tensor]]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None:
            return None

        pixel_values_videos = self._validate_and_reshape_mm_tensor(
            pixel_values_videos, "video pixel values"
        )
        video_grid_thw = self._validate_and_reshape_mm_tensor(video_grid_thw, "video grid_thw")

        return {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

    def _process_image_input(self, image_input: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if self.visual is None:
            return tuple()

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual,
                pixel_values,
                grid_thw_list,
                rope_type="rope_3d",
            )

        image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)

        merge_size = self.visual.spatial_merge_size
        sizes = (torch.tensor(grid_thw_list, dtype=torch.long).prod(-1) // (merge_size * merge_size)).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(self, video_input: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if self.visual is None:
            return tuple()

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual,
                pixel_values_videos,
                grid_thw_list,
                rope_type="rope_3d",
            )

        video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw_list)

        merge_size = self.visual.spatial_merge_size
        sizes = (torch.tensor(grid_thw_list, dtype=torch.long).prod(-1) // (merge_size * merge_size)).tolist()
        return video_embeds.split(sizes)

    # ---- Audio ----

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> ValleyOmniAudioFeatureInputs | None:
        input_audio_features = kwargs.pop("input_audio_features", None)
        audio_feature_lengths = kwargs.pop("audio_feature_lengths", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        if input_audio_features is None:
            return None

        return ValleyOmniAudioFeatureInputs(
            type="audio_features",
            input_features=input_audio_features,
            audio_feature_lengths=audio_feature_lengths,
            feature_attention_mask=feature_attention_mask,
        )


    def _process_audio_input(
        self,
        audio_input: ValleyOmniAudioFeatureInputs,
        audio_hashes: list[str] | None = None,
        cached_audio_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]

        audio_feat_lengths, audio_output_lengths = _get_feat_extract_output_lengths(
            audio_feature_lengths
        )

        audio_outputs = self.audio_encoder(
            input_features=input_features,
            feature_lens=audio_feature_lengths
        )
        projected = self.audio_connector(audio_outputs)
        # Split to one embedding tensor per audio item.
        return torch.split(projected, audio_output_lengths.tolist())

    # ------------------------------------------------------------------
    # Public multimodal API
    # ------------------------------------------------------------------

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
                    **kwargs
                )
            if (
                input_key in ("input_audio_features")
                and "audio" not in mm_input_by_modality
            ):
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(
                    **kwargs
                )
        return mm_input_by_modality

    def get_language_model(self) -> nn.Module:
        return self.language_model
    
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += tuple(video_embeddings)
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                multimodal_embeddings += tuple(audio_embeddings)
        return multimodal_embeddings


    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        # handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
            # handle_oov_mm_token=handle_oov_mm_token,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        deepstack_input_embeds = None
        # TODO (ywang96): support overlapping modalitiy embeddings so that
        # `use_audio_in_video` will work on V1.
        # split the feat dim to obtain multi-scale visual feature
        has_vision_embeddings = [
            embeddings.shape[-1] != self.config.text_config.hidden_size
            for embeddings in multimodal_embeddings
        ]
        if self.visual.deepstack_visual_indexes is not None and any(
            has_vision_embeddings
        ):
            multiscale_len = len(self.visual.deepstack_visual_indexes)
            multimodal_embeddings_multiscale = []
            is_vision = torch.zeros_like(is_multimodal)
            mm_positions = torch.nonzero(is_multimodal, as_tuple=True)[0]
            mm_position_idx = 0
            for index, embeddings in enumerate(multimodal_embeddings):
                num_tokens = embeddings.shape[0]
                current_positions = mm_positions[
                    mm_position_idx : mm_position_idx + num_tokens
                ]

                # Vision embeddings
                if embeddings.shape[-1] != self.config.text_config.hidden_size:
                    visual_dim = embeddings.shape[-1] // (multiscale_len + 1)
                    multi_dim = visual_dim * multiscale_len
                    embeddings_main, embeddings_multiscale = torch.split(
                        embeddings, [visual_dim, multi_dim], dim=-1
                    )
                    multimodal_embeddings[index] = embeddings_main
                    multimodal_embeddings_multiscale.append(embeddings_multiscale)
                    is_vision[current_positions] = True

                # Audio embeddings
                else:
                    is_vision[current_positions] = False

                mm_position_idx += num_tokens

            deepstack_input_embeds = inputs_embeds.new_zeros(
                inputs_embeds.size(0), multiscale_len * inputs_embeds.size(1)
            )
            deepstack_input_embeds = _merge_multimodal_embeddings(
                inputs_embeds=deepstack_input_embeds,
                multimodal_embeddings=multimodal_embeddings_multiscale,
                is_multimodal=is_vision,
            )
            deepstack_input_embeds = (
                deepstack_input_embeds.view(
                    inputs_embeds.shape[0], multiscale_len, visual_dim
                )
                .permute(1, 0, 2)
                .contiguous()
            )
            self._set_deepstack_input_embeds(deepstack_input_embeds)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds
        
    # ------------------------------------------------------------------
    # vLLM model interface
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        if (
            self.use_deepstack
            and inputs_embeds is not None
            and get_pp_group().is_first_rank
        ):
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0)
            )
        else:
            deepstack_input_embeds = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            # args for deepstack
            deepstack_input_embeds=deepstack_input_embeds,
        )

        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes: list[str] = []
        if self.visual is None:
            skip_prefixes.append("visual.")
        # Only skip audio weights when the entire audio branch is absent.
        if self.audio_encoder is None and self.audio_connector is None:
            skip_prefixes.extend(["audio_encoder.", "audio_connector."])

        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """Return mapping for multimodal weight packing.

        - ``language_model``: text transformer
        - ``tower_model``: visual tower and (optionally) audio encoder
        - ``connector``: audio connector projection
        """

        tower_keys: list[str] = []
        if self.visual is not None:
            tower_keys.append("visual")
        if self.audio_encoder is not None:
            tower_keys.append("audio_encoder")

        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="audio_connector",
            tower_model=tower_keys,
        )
    
    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {
                "image_grid_thw",
                "video_grid_thw",
                "second_per_grid_ts",
                "audio_feature_lengths",
                "use_audio_in_video",
            },
        )
        image_grid_thw = kwargs.get("image_grid_thw", [])
        video_grid_thw = kwargs.get("video_grid_thw", [])
        second_per_grid_ts = kwargs.get("second_per_grid_ts", [])
        audio_feature_lengths = kwargs.get("audio_feature_lengths", [])
        use_audio_in_video = any(kwargs.get("use_audio_in_video", []))

        image_grid_thw = (torch.stack if image_grid_thw else torch.tensor)(
            image_grid_thw
        )
        video_grid_thw = (torch.stack if video_grid_thw else torch.tensor)(
            video_grid_thw
        )

        input_ids = torch.tensor(input_tokens)
        if input_ids is None or input_ids.ndim != 1:
            raise ValueError("_omni3_get_input_positions_tensor expects 1D input_ids")

        seq_len = input_ids.shape[0]

        if isinstance(audio_feature_lengths, list):
            audio_feature_lengths = torch.tensor(
                audio_feature_lengths, dtype=torch.long
            )

        if not len(second_per_grid_ts) and len(video_grid_thw):
            second_per_grids = torch.ones(len(video_grid_thw), dtype=torch.float32)
        else:
            second_per_grids = torch.tensor(second_per_grid_ts, dtype=torch.float32)

        config = self.config
        spatial_merge_size = config.vision_config.spatial_merge_size
        image_token_id = config.image_token_id
        video_token_id = config.video_token_id
        audio_token_id = config.audio_token_id
        vision_start_token_id = config.vision_start_token_id
        audio_start_token_id = config.audio_start_token_id
        position_id_per_seconds = config.position_id_per_seconds

        vision_start_indices = torch.argwhere(
            input_ids == vision_start_token_id
        ).squeeze(1)
        if vision_start_indices.numel() > 0:
            vision_tokens = input_ids[vision_start_indices + 1]
        else:
            vision_tokens = input_ids.new_empty((0,), dtype=input_ids.dtype)
        audio_nums = torch.sum(input_ids == audio_start_token_id)
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (
            (vision_tokens == audio_start_token_id).sum()
            if use_audio_in_video
            else (vision_tokens == video_token_id).sum()
        )

        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0
        image_idx = 0
        video_idx = 0
        audio_idx = 0
        remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums  # noqa: E501
        multimodal_nums = (
            image_nums + audio_nums
            if use_audio_in_video
            else image_nums + video_nums + audio_nums
        )  # noqa: E501

        for _ in range(multimodal_nums):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
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

            if min_ed == ed_audio_start:
                text_len = min_ed - st
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len, dtype=torch.long)
                        .view(1, -1)
                        .expand(3, -1)
                        + st_idx
                    )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                bos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(bos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                _, audio_len = _get_feat_extract_output_lengths(
                    audio_feature_lengths[audio_idx]
                )
                llm_pos_ids = (
                    torch.arange(audio_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                llm_pos_ids_list.append(llm_pos_ids)
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                eos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(eos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st += text_len + bos_len + audio_len + eos_len
                audio_idx += 1
                remain_audios -= 1
            elif (
                min_ed == ed_vision_start
                and input_ids[ed_vision_start + 1] == image_token_id
            ):
                text_len = min_ed - st
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len, dtype=torch.long)
                        .view(1, -1)
                        .expand(3, -1)
                        + st_idx
                    )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                bos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(bos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                grid_t = image_grid_thw[image_idx][0]
                grid_hs = image_grid_thw[:, 1]
                grid_ws = image_grid_thw[:, 2]
                t_index = torch.arange(grid_t) * position_id_per_seconds
                llm_pos_ids = get_llm_pos_ids_for_vision(
                    st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                )
                image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                llm_pos_ids_list.append(llm_pos_ids)
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                eos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(eos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st += text_len + bos_len + image_len + eos_len
                image_idx += 1
                remain_images -= 1
            elif (
                min_ed == ed_vision_start
                and input_ids[ed_vision_start + 1] == video_token_id
                and not use_audio_in_video
            ):
                text_len = min_ed - st
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len, dtype=torch.long)
                        .view(1, -1)
                        .expand(3, -1)
                        + st_idx
                    )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                bos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(bos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                grid_t = video_grid_thw[video_idx][0]
                grid_hs = video_grid_thw[:, 1]
                grid_ws = video_grid_thw[:, 2]
                t_index = (
                    torch.arange(grid_t)
                    * float(second_per_grids[video_idx].item())
                    * position_id_per_seconds
                )
                llm_pos_ids = get_llm_pos_ids_for_vision(
                    st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                )
                video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                llm_pos_ids_list.append(llm_pos_ids)
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                eos_len = 1
                llm_pos_ids_list.append(
                    torch.arange(eos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                st += text_len + bos_len + video_len + eos_len
                video_idx += 1
                remain_videos -= 1
            elif (
                min_ed == ed_vision_start
                and ed_vision_start + 1 == ed_audio_start
                and use_audio_in_video
            ):
                text_len = min_ed - st
                if text_len != 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len, dtype=torch.long)
                        .view(1, -1)
                        .expand(3, -1)
                        + st_idx
                    )
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                bos_len = 1
                bos_block = (
                    torch.arange(bos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                llm_pos_ids_list.append(bos_block)
                llm_pos_ids_list.append(bos_block)
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                _, audio_len = _get_feat_extract_output_lengths(
                    audio_feature_lengths[audio_idx]
                )
                audio_llm_pos_ids = (
                    torch.arange(audio_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                grid_t = video_grid_thw[video_idx][0]
                grid_hs = video_grid_thw[:, 1]
                grid_ws = video_grid_thw[:, 2]
                t_index = (
                    torch.arange(grid_t)
                    * float(second_per_grids[video_idx].item())
                    * position_id_per_seconds
                )
                video_llm_pos_ids = get_llm_pos_ids_for_vision(
                    st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                )
                video_data_index, audio_data_index = 0, 0
                while (
                    video_data_index < video_llm_pos_ids.shape[-1]
                    and audio_data_index < audio_llm_pos_ids.shape[-1]
                ):
                    if (
                        video_llm_pos_ids[0][video_data_index]
                        <= audio_llm_pos_ids[0][audio_data_index]
                    ):
                        llm_pos_ids_list.append(
                            video_llm_pos_ids[
                                :, video_data_index : video_data_index + 1
                            ]
                        )
                        video_data_index += 1
                    else:
                        llm_pos_ids_list.append(
                            audio_llm_pos_ids[
                                :, audio_data_index : audio_data_index + 1
                            ]
                        )
                        audio_data_index += 1
                if video_data_index < video_llm_pos_ids.shape[-1]:
                    llm_pos_ids_list.append(
                        video_llm_pos_ids[
                            :, video_data_index : video_llm_pos_ids.shape[-1]
                        ]
                    )
                if audio_data_index < audio_llm_pos_ids.shape[-1]:
                    llm_pos_ids_list.append(
                        audio_llm_pos_ids[
                            :, audio_data_index : audio_llm_pos_ids.shape[-1]
                        ]
                    )
                video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                eos_len = 1
                eos_block = (
                    torch.arange(eos_len, dtype=torch.long).view(1, -1).expand(3, -1)
                    + st_idx
                )
                llm_pos_ids_list.append(eos_block)
                llm_pos_ids_list.append(eos_block)
                st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2  # noqa: E501
                audio_idx += 1
                video_idx += 1
                remain_videos -= 1
                remain_audios -= 1

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1)
                + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if llm_positions.shape[1] != seq_len:
            raise RuntimeError("Position ids length mismatch with input ids length")

        mrope_position_delta = llm_positions.max() + 1 - seq_len
        return llm_positions, mrope_position_delta
