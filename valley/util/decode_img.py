import json
import os
import base64
from PIL import Image
import random
import io


def image_preprocess(image_str):
    image = _load_image(b64_decode(image_str))

    return image


def b64_decode(string):
    if isinstance(string, str):
        string = string.encode()
    return base64.decodebytes(string)


def _load_image(buffer):
    img = Image.open(io.BytesIO(buffer))
    img = img.convert('RGB')
    return img


path = 'datas/GB_val.txt'  # 测试集路径
save_path = 'datas/gambling/'  # 保存路径

if not os.path.exists(save_path):
    os.mkdir(save_path)

with open(path, 'r') as f:
    lines = f.readlines()

for idx, ex in enumerate(lines):
    data = json.loads(ex)
    # print(data.keys())  TODO: 看看有哪些字段，选择一些保存
    title = data['title']
    # asr = data['asr']
    merge_ocr = data['merge_ocr']
    # product_title = data['product_title']
    # video_desp = data['video_desp']
    gt_label = data['gt_label']
    video_frame = data['video_frame']
    # text = {'asr': asr, 'merge_ocr': merge_ocr, 'product_title': product_title, 'video_desp': video_desp,
    #         'gt_label': gt_label}
    text = {'merge_ocr': merge_ocr, 'title': title, 'gt_label': gt_label}
    with open(os.path.join(save_path, f'{idx}.json'), 'w') as f:
        f.write(json.dumps(text, indent=4))

    for i, video in enumerate(video_frame):
        if random.random() < 0.5:
            image = image_preprocess(video)
            image.save(os.path.join(save_path, f'{idx}_{i}.jpg'), quality=95)

    if idx > 50:
        break
