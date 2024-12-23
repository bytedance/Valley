from valley_eagle_chat import ValleyEagleChat
import urllib

model = ValleyEagleChat(
    model_path="/mnt/bn/yangmin-priv/czh/checkpoints/valley/valley_b6_eagle_qwen2_5_siglip_ovis_convadapter_vocab32k_stage1_5_5200w_pack_part0_stage2_v3_50_200_250w_aquila_s3_4_anneal",
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
