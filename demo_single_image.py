from valley_eagle_chat import ValleyEagleChat
import urllib

model = ValleyEagleChat(
    model_path="bytedance-research/Valley-Eagle-7B",
    padding_side="left",
)

url = "http://p16-goveng-va.ibyteimg.com/tos-maliva-i-wtmo38ne4c-us/4870400481414052507~tplv-wtmo38ne4c-jpeg.jpeg"
img = urllib.request.urlopen(url=url, timeout=5).read()
request = {
    "chat_history": [
        {"role": "system", "content": "You are Valley, developed by ByteDance. Your are a helpfull Assistant."},
        {"role": "user", "content": "Describe the given image."},
    ],
    "images": [img],
}
result = model(request)
print(f"\n>>> Assistant:\n")
print(result)
