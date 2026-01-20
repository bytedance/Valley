import dataclasses
from enum import auto, Enum
from typing import List, Tuple
import base64

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    mode: str = None
    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])
    
    
    def get_video(self,):
        videos = []
        path_list = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, video_path, image_process_mode = msg
                    path_list.append(video_path)
                    with open(video_path, "rb") as videoFile:
                        video_b64_str = base64.b64encode(videoFile.read())
                    videos.append(video_b64_str)
        return videos, path_list
    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    msg, image_list, image_process_mode = msg
                    if type(image_list) is not list:
                        image_list = [image_list]
                    for image in image_list:
                        if image_process_mode == "Pad":
                            def expand2square(pil_img, background_color=(122, 116, 104)):
                                width, height = pil_img.size
                                if width == height:
                                    return pil_img
                                elif width > height:
                                    result = Image.new(pil_img.mode, (width, width), background_color)
                                    result.paste(pil_img, (0, (width - height) // 2))
                                    return result
                                else:
                                    result = Image.new(pil_img.mode, (height, height), background_color)
                                    result.paste(pil_img, ((height - width) // 2, 0))
                                    return result
                            image = expand2square(image)
                        elif image_process_mode == "Crop":
                            pass
                        elif image_process_mode == "Resize":
                            image = image.resize((224, 224))
                        else:
                            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
                        max_hw, min_hw = max(image.size), min(image.size)
                        aspect_ratio = max_hw / min_hw
                        max_len, min_len = 800, 400
                        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                        longest_edge = int(shortest_edge * aspect_ratio)
                        W, H = image.size
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        image = image.resize((W, H))
                        if return_pil:
                            images.append(image)
                        else:
                            buffered = BytesIO()
                            image.save(buffered, format="JPEG")
                            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                            images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, image, image_process_mode = msg
                    img_str = ''
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    # image = image.resize((224, 224))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = img_str+f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = msg.replace('<image>', '')+img_str
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def video_to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    
                    msg, video, image_process_mode = msg
                    with open(video, "rb") as videoFile:
                        video_b64_str = base64.b64encode(videoFile.read()).decode("utf-8") 
                    img_str = ''
                    img_str = img_str+f'''<video controls align="left" style="height: 200px;" src="data:video/mp4;base64,{video_b64_str}">
                                            The “video” tag is not supported by your browser. Click [here] to download the video file.
                                            </video>'''
                    msg = msg.replace('<video>', '')+img_str
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }
conv_v1_2 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

simple_conv_video = Conversation(
    system="You are Valley, a large language and vision assistant trained by ByteDance."
           "You are able to understand the visual content or video that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hi!"),
        ("Assistant", "Hi there!  How can I help you today?\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
default_conversation = simple_conv_video
conv_templates = {
    "v1":conv_v1_2,
    "multimodal_video":simple_conv_video,
}