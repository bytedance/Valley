import argparse
from transformers import AutoTokenizer
from valley.model.valley_model import ValleyLlamaForCausalLM
import torch
from enum import Enum

from valley.util.config import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_VIDEO_FRAME_TOKEN,
    DEFAULT_VI_START_TOKEN,
    DEFAULT_VI_END_TOKEN,
    DEFAULT_VIDEO_TOKEN,
)


class ModelPath(Enum):
    Valley2_7b = "luoruipu1/Valley2-7b"


parser = argparse.ArgumentParser(description="Process some video.")
parser.add_argument("video_file", type=str, help="The path to the video file")
args = parser.parse_args()
video_file = args.video_file


def init_vision_token(model, tokenizer):
    vision_config = model.get_model().vision_tower.config
    (
        vision_config.im_start_token,
        vision_config.im_end_token,
    ) = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    (
        vision_config.vi_start_token,
        vision_config.vi_end_token,
    ) = tokenizer.convert_tokens_to_ids([DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN])
    vision_config.vi_frame_token = tokenizer.convert_tokens_to_ids(
        DEFAULT_VIDEO_FRAME_TOKEN
    )
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN]
    )[0]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input the query
query = f"{DEFAULT_VIDEO_TOKEN} Describe the video concisely."

# input the system prompt
system_prompt = "You are Valley, a large language and vision assistant trained by ByteDance. You are able to understand the visual content or video that the user provides, and assist the user with a variety of tasks using natural language. Follow the instructions carefully and explain your answers in detail."


model_path = ModelPath.Valley2_7b
# pulls model from HF given path
model = ValleyLlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)


tokenizer = AutoTokenizer.from_pretrained(model_path)

init_vision_token(model, tokenizer)

model = model.to(device)
model.eval()


# we support openai format input
message = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Hi!"},
    {"role": "assistent", "content": "Hi there! How can I help you today?"},
    {"role": "user", "content": query},
]

gen_kwargs = dict(
    do_sample=True,
    temperature=0.2,
    max_new_tokens=1024,
)


response = model.completion(tokenizer, video_file, message, gen_kwargs, device)

