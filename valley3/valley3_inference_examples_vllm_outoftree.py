from PIL import Image
import torch
from transformers.models.valley_omni import (
    ValleyOmniConfig, ValleyOmniForConditionalGeneration, ValleyOmniProcessor, AutoProcessor
)
from qwen_omni_utils import process_mm_info
from typing import Any
from vllm import LLM, SamplingParams, ModelRegistry
from inference_valley3.vllm_0_18_0.valley_omni_oot import ValleyOmniForConditionalGeneration
ModelRegistry.register_model("ValleyOmniForConditionalGeneration", ValleyOmniForConditionalGeneration)

# model_path = "/path/to/Valley3-8B-Instruct-0401"
model_path = "/path/to/Valley3-32B-Instruct-0401"

model_tp = 1

example_image_path = "./example_data/example_image.jpg"
example_audio_path = "./example_data/example_audio.wav"
example_video_path = "./example_data/example_video.mp4"
example_video_audio_path = "./example_data/example_video_with_audio_short.mp4"

text_only_message = [{"role": "user", "content": [{"type": "text", "text": "Please tell me which number is larger: 3.11 or 3.8?"}],}]
image_text_message = [{"role": "user", "content": [{"type": "image", "image": example_image_path}, {"type": "text", "text": "Describe the content."}],}]
audio_text_message = [{"role": "user", "content": [{"type": "audio", "audio": example_audio_path}, {"type": "text", "text": "Fully transcribe the content of the audio."}],}]
video_text_message = [{"role": "user", "content": [{"type": "video", "video": example_video_path}, {"type": "text", "text": "Describe the video in brief."}],}]
video_audio_text_message = [{"role": "user", "content": [{"type": "video", "video": example_video_audio_path}, {"type": "text", "text": "Please describe the video in brief and transcribe what the person in the video is saying."}],}]

test_cases = [text_only_message, image_text_message, audio_text_message, video_text_message, video_audio_text_message]

if __name__ == "__main__":
    
    print(model_path)

    model = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=model_tp,
        limit_mm_per_prompt={'image': 10, 'video': 10, 'audio': 10},
        max_num_seqs=8,
        max_model_len=128000,
        seed=1234,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
    )

    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = ValleyOmniProcessor.from_pretrained(
        model_path, min_pixels=min_pixels, max_pixels=max_pixels
    )

    for i in range(len(test_cases)):
        messages = test_cases[i]
        use_audio_in_video = (i == len(test_cases) - 1)
        print(messages)
        chat_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audio_inputs, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = {
            'prompt': chat_input,
            'multi_modal_data': {},
            "mm_processor_kwargs": {
                "use_audio_in_video": use_audio_in_video,
            },
        }

        if image_inputs is not None:
            inputs['multi_modal_data']['image'] = image_inputs
        if video_inputs is not None:
            inputs['multi_modal_data']['video'] = video_inputs
        if audio_inputs is not None:
            inputs['multi_modal_data']['audio'] = audio_inputs

        outputs = model.generate([inputs], sampling_params=sampling_params)
        text_output = outputs[0].outputs[0].text
        
        print("-"*30)
        print("Model output")
        print("-"*30)
        print(text_output)
        print("\n\n")
