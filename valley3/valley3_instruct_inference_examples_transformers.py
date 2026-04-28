from PIL import Image
import torch
from transformers.models.valley_omni import (
    ValleyOmniConfig, ValleyOmniForConditionalGeneration, ValleyOmniProcessor
)
from qwen_omni_utils import process_mm_info
from typing import Any

# model_path = "/path/to/Valley3-8B-Instruct-0401"
model_path = "/path/to/Valley3-32B-Instruct-0401"

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

    model = ValleyOmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = ValleyOmniProcessor.from_pretrained(
        model_path, min_pixels=min_pixels, max_pixels=max_pixels
    )

    for i in range(len(test_cases)):
        messages = test_cases[i]
        print(messages)
        use_audio_in_video = (i == len(test_cases) - 1)
        chat_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audio_inputs, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = processor(
            text=[chat_input],
            audio=audio_inputs,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video
        ).to(model.device).to(model.dtype)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=4096, use_audio_in_video=use_audio_in_video)

        generated_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = processor.batch_decode(generated_trimmed, skip_special_tokens=True)[0]
        print("-"*30)
        print("Model output")
        print("-"*30)
        print(output)
        print("\n\n")
