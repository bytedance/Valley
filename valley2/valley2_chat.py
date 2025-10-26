#  Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
import io
import torch
import re
import numpy as np
from typing import Dict, List, Union
from PIL import Image
from qwen_vl_utils import fetch_image
from transformers import AutoTokenizer, SiglipImageProcessor, Qwen2VLImageProcessor
from transformers import set_seed

from . import conversation as conversation_lib
from .valley_utils import disable_torch_init
from .model.language_model.valley_qwen2 import ValleyQwen2ForCausalLM
from .util.data_util import dynamic_preprocess, preprocess
from .util.mm_utils import process_anyres_image
from .util.vision_encoder_config import siglip_processor_config, qwen2vl_processor_config

logging.basicConfig(level=logging.INFO)


# Init the constants
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VI_START_TOKEN = "<vi_start>"
DEFAULT_VI_END_TOKEN = "<vi_end>"
BLACK_IMG_ENV = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x03\x00\x00\x00\x03\x08\x02\x00\x00\x00\xd9J"\xe8\x00\x00\x00\x12IDAT\x08\x1dcd\x80\x01F\x06\x18`d\x80\x01\x00\x00Z\x00\x04we\x03N\x00\x00\x00\x00IEND\xaeB`\x82'


def preprocess_multimodal(
    conversations,
    img_num,
    data_args,
) -> Dict:
    for sentence in conversations:
        if DEFAULT_VIDEO_TOKEN in sentence["value"]:
            if data_args.use_special_start_end_token:
                video_replace_token = (DEFAULT_VI_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_VI_END_TOKEN) * img_num
            else:
                video_replace_token = DEFAULT_IMAGE_TOKEN * img_num
            sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, "").strip()
            sentence["value"] = video_replace_token + "\n" + sentence["value"]
        else:
            segs = re.split(DEFAULT_IMAGE_TOKEN, sentence["value"])
            if data_args.use_special_start_end_token:
                sentence["value"] = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN).join(
                    segs[: img_num + 1]
                ) + "".join(segs[img_num + 1 :])
            else:
                sentence["value"] = DEFAULT_IMAGE_TOKEN.join(segs[: img_num + 1]) + "".join(segs[img_num + 1 :])

    return conversations


class Valley2Chat:
    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.float16,
        padding_side: str = "left",
        use_fast: bool = True,
        trust_remote_code: bool = True,
        output_logits=False,
        conversation_tag="qwen2",
        max_new_tokens: int = 768,
        seed: int = 42,
        black_img: bytes = BLACK_IMG_ENV,
    ):

        # Init the env
        disable_torch_init()
        set_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_logits = output_logits
        self.conversation_tag = conversation_tag
        conversation_lib.default_conversation = conversation_lib.conv_templates[self.conversation_tag]

        # Load model and tokenizer
        logging.info(f"Start loading valley model from {model_path}")
        self.model_path = model_path
        self.model = ValleyQwen2ForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
        self.model = self.model.to(self.device).half()
        self.model.config.min_tile_num = 1
        self.model.config.max_tile_num = 9
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast, trust_remote_code=trust_remote_code)
        self.tokenizer.padding_side = padding_side
        self.max_new_tokens = max_new_tokens

        # Load image preprocessor
        self.black_img = black_img
        self.image_processor = SiglipImageProcessor.from_dict(siglip_processor_config)
        self.qwen2vl_processor = Qwen2VLImageProcessor.from_dict(qwen2vl_processor_config, max_pixels=1280 * 28 * 28)
        self.image_processor.crop_size = self.image_processor.size["height"]

    def preprocess_images(self, image_binary_list) -> torch.FloatTensor:
        byte2image = lambda byte_data: Image.open(io.BytesIO(byte_data)).convert("RGB")
        images = []
        for binary in image_binary_list:
            if isinstance(binary, Image.Image):
                images.append(binary.convert("RGB") )
            elif isinstance(binary, bytes):
                images.append(byte2image(binary))
            else:
                raise ValueError("unsupported type")
        video_pad = []
        for img in images:
            if self.model.config.anyres:
                image = process_anyres_image(img, self.image_processor, self.model.config.grid_pinpoints)
            else:
                image = self.image_processor(img, return_tensors="pt")["pixel_values"][0]

            video_pad.append(image)

        video_pad = [self.black_img] if len(video_pad) == 0 else video_pad

        if not self.model.config.anyres:
            video = torch.stack(video_pad, dim=0)
        else:
            video = [torch.stack(img, dim=0) for img in video_pad]
        return video

    def __call__(self, request):
        # preprocess images
        if "images" not in request or not request["images"] or not request["images"][0]:
            images_binary = [self.black_img]
        else:
            images_binary = request["images"][:8]

        video_images_tensor = self.preprocess_images(images_binary)
        img_length = len(video_images_tensor)
        video_images_tensor = [video_images_tensor]

        # Process system prompt and image input
        messages = []
        chat_history = request["chat_history"]
        if chat_history[0]["role"] == "system":
            if chat_history[0]["content"]:
                conversation_lib.default_conversation.system = chat_history[0]["content"]
            chat_history = chat_history[1:]
        chat = chat_history[0]
        assert chat["role"] == "user"

        if images_binary and "<image>" not in chat["content"]:
            image_token = "".join(["<image>"] * len(images_binary))
            chat["content"] = f"{image_token}\n{chat['content']}"

        messages.append({"from": "human", "value": chat["content"]})
        text = chat["content"]

        # add all other chat_history to messages
        for chat in chat_history[1:]:
            if chat["role"] == "user":
                messages.append({"from": "human", "value": chat["content"]})
            elif chat["role"] == "assistant":
                messages.append({"from": "gpt", "value": chat["content"]})
            else:
                raise Exception(f"unknow role {chat['role']} in multi round")

        # get eagle image features
        messages_qwen = []
        image_list = []
        if isinstance(images_binary[0], Image.Image):
            images_pil = [img.convert("RGB") for img in images_binary]
        elif isinstance(images_binary[0], bytes):
            images_pil = [Image.open(io.BytesIO(img)).convert("RGB") for img in images_binary]
        image_sizes = [[x.size for x in images_pil]]
        for image_file in images_pil:
            image = fetch_image({"image": image_file})
            image_list.append(image)
        data_dict_qwen2vl = self.qwen2vl_processor(image_list, return_tensors="pt")

        # process messages, get tensors which will be input to model
        source = preprocess_multimodal(messages, img_length, self.model.config)
        data_dict = preprocess(
            source,
            self.tokenizer,
            has_image=True,
            only_mask_system=False,
            inference=True,
        )
        input_ids = data_dict["input_ids"]
        input_ids = input_ids.unsqueeze(0).to(self.device)
        if img_length:
            images = [[item.to(self.device).half() for item in img] for img in video_images_tensor]

        # model inference
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=images,
                image_sizes=image_sizes,
                pixel_values=data_dict_qwen2vl["pixel_values"].to(self.device),
                image_grid_thw=data_dict_qwen2vl["image_grid_thw"].to(self.device),
                pixel_values_videos=None,
                video_grid_thw=None,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        input_token_len = input_ids.shape[1]
        generation_text = self.tokenizer.batch_decode(output_ids.sequences[:, input_token_len:])[0]
        generation_text = generation_text.replace("<|im_end|>", "")
        return generation_text