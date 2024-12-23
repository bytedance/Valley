# Valley 2.0
## Introduction
Valley is a cutting-edge multimodal large model designed to handle a variety of tasks involving text, images, and video data, which is developed by ByteDance. Our model not only

- Achieved the best results in the inhouse e-commerce and short-video benchmarks
- Demonstrated comparatively outstanding performance in the OpenCompass (average scores > 67) tests

when evaluated against models of the same scale.

<!-- <div style="display:flex;">
  <img src="assets/open_compass_1223.jpg" alt="opencompass" style="height:300px;" />
  <img src="assets/tts_inhouse_benchmark_1223.jpg" alt="inhouse" style="height:300px;" />
</div> -->


## Valley-Eagle
The foundational version of Valley is a multimodal large model aligned with Siglip and Qwen2.5, incorporating LargeMLP and ConvAdapter to construct the projector. 

- In the final version, we also referenced [Eagle](https://arxiv.org/pdf/2408.15998), introducing an additional VisionEncoder that can flexibly adjust the number of tokens and is parallelized with the original visual tokens. 
- This enhancement supplements the model’s performance in extreme scenarios, and we chose the Qwen2vl VisionEncoder for this purpose. 

and the model structure is shown as follows:

<div style="display:flex;">
  <img src="assets/valley_structure.jpeg" alt="opencompass" style="height:600px;" />
</div>


## Release
- [12/23] 🔥 Announcing [Valley-Qwen2.5-7B](https://huggingface.co/ByteDance)!

## Environment Setup
``` bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Inference Demo
- Single image
``` python
from valley_eagle_chat import ValleyEagleChat
model = ValleyEagleChat(
    model_path='bytedance-research/Valley-Eagle-7B',
    padding_side = 'left',
)

url = 'http://p16-goveng-va.ibyteimg.com/tos-maliva-i-wtmo38ne4c-us/4870400481414052507~tplv-wtmo38ne4c-jpeg.jpeg'
img = urllib.request.urlopen(url=url, timeout=5).read()

request = {
    "chat_history": [
        {'role': 'system', 'content': 'You are Valley, developed by ByteDance. Your are a helpfull Assistant.'},
        {'role': 'user', 'content': 'Describe the given image.'},
    ],
    "images": [img],
}

result = model(request)
print(f"\n>>> Assistant:\n")
print(result)
```

- Video
``` python
from valley_eagle_chat import ValleyEagleChat
import decord
import requests
import numpy as np
from torchvision import transforms

model = ValleyEagleChat(
    model_path='bytedance-research/Valley-Eagle-7B',
    padding_side = 'left',
)

url = 'https://videos.pexels.com/video-files/29641276/12753127_1920_1080_25fps.mp4'
video_file = './video.mp4'
response = requests.get(url)
if response.status_code == 200:
    with open("video.mp4", "wb") as f:
        f.write(response.content)
else:
    print("download error!")
    exit(1)

video_reader = decord.VideoReader(video_file)
decord.bridge.set_bridge("torch")
video = video_reader.get_batch(
    np.linspace(0,  len(video_reader) - 1, 8).astype(np.int_)
).byte()
print([transforms.ToPILImage()(image.permute(2, 0, 1)).convert("RGB") for image in video])

request = {
    "chat_history": [
        {'role': 'system', 'content': 'You are Valley, developed by ByteDance. Your are a helpfull Assistant.'},
        {'role': 'user', 'content': 'Describe the given video.'},
    ],
    "images": [transforms.ToPILImage()(image.permute(2, 0, 1)).convert("RGB") for image in video],
}
result = model(request)
print(f"\n>>> Assistant:\n")
print(result)
```

## License Agreement
All of our open-source models are licensed under the Apache-2.0 license.


## Citation
Coming Soon!
