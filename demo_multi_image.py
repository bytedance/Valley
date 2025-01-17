from valley_eagle_chat import ValleyEagleChat
import urllib
from io import BytesIO
from PIL import Image

model = ValleyEagleChat(
    model_path="bytedance-research/Valley-Eagle-7B",
    padding_side="left",
)

urls = [
    "https://plus.unsplash.com/premium_photo-1661632559307-902ac3f6174c",
    "https://plus.unsplash.com/premium_photo-1661632559713-a478160cd72e",
    "https://plus.unsplash.com/premium_photo-1661607772173-54f7b8263c27",
    "https://plus.unsplash.com/premium_photo-1661607115685-36b2a7276389",
    "https://plus.unsplash.com/premium_photo-1661607103369-e799ee7ef954",
    "https://plus.unsplash.com/premium_photo-1661628841460-1c9d7e6669ec",
    "https://plus.unsplash.com/premium_photo-1661602273588-f213a4155caf",
    "https://plus.unsplash.com/premium_photo-1661602247160-d42d7aba6798"
]

url2img = lambda url: Image.open(
    BytesIO(urllib.request.urlopen(url=url, timeout=5).read())
).convert("RGB")

imgs = [url2img(url) for url in urls]

request = {
    "chat_history": [
        {"role": "system", "content": "You are Valley, developed by ByteDance. Your are a helpfull Assistant."},
        {"role": "user", "content": "Describe the given images."},
    ],
    "images": imgs,
}
result = model(request)
print(f"\n>>> Assistant:\n")
print(result)
