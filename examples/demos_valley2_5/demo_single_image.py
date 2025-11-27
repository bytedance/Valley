import torch
import urllib
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, AutoModel

GTHINKER_SYS_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>. In the reasoning process enclosed within <think> </think>,"
    " each specific visual cue is enclosed within <vcues_*>...</vcues_*>, where * indicates the index of the specific cue. "
    "Before concluding the final answer, pause for a quick consistency check: verify whether the visual cues support the reasoning "
    "and whether each step logically follows from what is seen. If correct, conclude the answer; otherwise, revise the visual cues and reasoning, then conclude."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(
    "bytedance-research/Valley2.5", 
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "bytedance-research/Valley2.5", 
    only_navit=True,
    max_pixels=28*28*16384,
    min_pixels=28*28*4,
    trust_remote_code=True
)

url = "https://images.unsplash.com/photo-1734640113825-24dd7c056052"
img = urllib.request.urlopen(url=url, timeout=5).read()
img = Image.open(BytesIO(img)).convert("RGB")
res = processor(
    {
        "conversations": 
        [
            {"role": "system", "content": GTHINKER_SYS_PROMPT},
            {"role": "user", "content": "Describe the given image."},
        ], 
        "images": [img]
    }, 
    enable_thinking=True
)

with torch.inference_mode():
    model.to(dtype=torch.bfloat16, device=device)
    output_ids = model.generate(
        input_ids=res["input_ids"].to(device),
        image_sizes=res["image_sizes"],
        pixel_values=res["pixel_values"].to(dtype=torch.bfloat16, device=device),
        image_grid_thw=res["image_grid_thw"].to(device),
        do_sample=False,
        max_new_tokens=4096,
        repetition_penalty=1.0,
        return_dict_in_generate=True,
        output_scores=True
    )

input_token_len = res["input_ids"].shape[1]
generation_text = processor.batch_decode(output_ids.sequences[:, input_token_len:])[0]
generation_text = generation_text.replace("<|im_end|>", "")
print(generation_text)