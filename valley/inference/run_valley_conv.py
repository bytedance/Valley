import argparse
import torch
import sys
sys.path.append('./valley')
from transformers import LlamaTokenizer
from valley.model.valley_model import ValleyLlamaForCausalLM
import torch
import os
from valley.utils import disable_torch_init
from transformers import CLIPImageProcessor
import os
import random
from conversation import conv_templates, SeparatorStyle
from util.config import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_VIDEO_FRAME_TOKEN, DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN
from util.data_util import  KeywordsStoppingCriteria
import decord
from torchvision import transforms
from data import video_transform
import numpy as np
from pathlib import Path
from PIL import Image
from constants import SHELL_UI_HEADER
def load_video(path,image_processor):
    if os.path.isfile(path):
        video_reader = decord.VideoReader(path, num_threads=1, ctx=decord.cpu(0))
        decord.bridge.set_bridge('torch')
        video_len = len(video_reader)
        video = video_reader.get_batch(np.linspace(0, video_len - 1, 8).astype(np.int_)).byte()#8, height,width,3
        video = video.permute(3, 0, 1, 2) # 3 x 8 x height x width
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        crop_size, scale_size = 224, 256
        trans = transforms.Compose([
            video_transform.TensorToNumpy(),
            video_transform.Resize(scale_size),
            video_transform.CenterCrop(crop_size),
            video_transform.ClipToTensor(channel_nb=3),
            video_transform.Normalize(mean=input_mean, std=input_std)
        ])
        video = trans(video)
    else:
        video_frames = list(Path(path).rglob('*'))
        video_frames = [Image.open(path) for path in video_frames]
        if len(video_frames) >8:
            video_frames = [video_frames[i] for i in np.linspace(0, len(video_frames) - 1, 8).astype(np.int_)]
        # if 1 <= video_frames[0].size[1]/video_frames[0].size[0]:
        #     min_length = min(video_frames[0].size)
        #     resize = transforms.Resize([min_length,min_length])
        #     video_frames = [resize(frame) for frame in video_frames]
        #     test_frame = video_frames[0]
        video = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
        video = video.permute(1,0, 2, 3)
    return video
def assistant_out(model,conv,tokenizer,input_ids,image_tensor):

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    keywords = ['###']
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),# 1,8,3,224,224
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)

    out = outputs[0]
    while True:
        cur_len = len(out)
        out = out.strip()
        for pattern in ['###', 'Assistant:', 'Response:','LLaVA:']:
            if out.startswith(pattern):
                out = out[len(pattern):].strip()
        if len(out) == cur_len:
            break
    try:
        index = out.index(conv.sep)
    except ValueError:
        out += conv.sep
        index = out.index(conv.sep)

    out = out[:index].strip()
    string_out = out.replace('\n','') + '\n'
    return string_out

def inference(args):
    
    random.seed(42)

    disable_torch_init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model_name = os.path.expanduser(args.model_name)
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    
    # load model

    print('load model')
    model = ValleyLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    print('load end')

    
    
    model.model.multi_image = True
    model.model.multi_image_mode = 'concatenate'
    # if torch.cuda.is_available():
    # model = model.to('cuda:'+str(this_rank_gpu_index))
    model = model.to(device)
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    
    vision_tower = model.get_model().vision_tower   
    vision_tower.to(device, dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        vision_config.vi_start_token, vision_config.vi_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN])
        vision_config.vi_frame_token = tokenizer.convert_tokens_to_ids(DEFAULT_VIDEO_FRAME_TOKEN)
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    
    video_path = ''
    os.system('cls' if os.name == 'nt' else 'clear')
    print(SHELL_UI_HEADER)
    while True:
        try:
            if not video_path:
                video_path = input("Assistant: please input video path. path: ")
                if video_path == '':
                    video_path = args.video_file
                print()
                video = load_video(video_path,image_processor) # 3, 8, 224, 224
                video = video.permute(1,0,2,3)# 8,3,224,224
                video_length = video.shape[0]
                test_image = video[0]
                image_tensor = video.to(device)
                conv = conv_templates[args.conv_mode].copy()
            
            qs = input("human:     ")
            print()
            if qs == 'change video':
                video_path = input("Assistant: please input video path. path: ")
                print()
                video = load_video(video_path,image_processor) # 3, 8, 224, 224
                video = video.permute(1,0,2,3)# 8,3,224,224
                video_length = video.shape[0]
                test_image = video[0]
                image_tensor = video.to(device)
                conv = conv_templates[args.conv_mode].copy()
                qs = input("human:     ")
                print()
            if qs == 'quit':
                break

            if not conv.has_video:
                if mm_use_im_start_end:
                    qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN + DEFAULT_VI_START_TOKEN + DEFAULT_VIDEO_FRAME_TOKEN * video_length + DEFAULT_VI_END_TOKEN
                else:
                    qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                conv.has_video = True
            
            conv.append_message(conv.roles[0], qs)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt])

            # load video
            
            input_ids = torch.as_tensor(inputs.input_ids).cuda()
            responce = assistant_out(model,conv,tokenizer,input_ids,image_tensor)
            conv.append_message(conv.roles[1], responce)
            print('Assistant: '+responce.strip()+'\n')
        except Exception as e:
            print('Assistant: '+str(e)+'\n')
def gather_result(args):
    num_worker = args.world_size
    with open(args.out_path,'a+') as f:
        for i in range(num_worker):
            with open(args.out_path+".worker_"+str(i),'r') as tf:
                tmp_result = tf.readlines()
            f.writelines(tmp_result)
            os.remove(args.out_path+".worker_"+str(i))
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="../checkpoints/stable-valley-13b-v1/")
    parser.add_argument("--query", type=str, required=False,default="Describe the following video concisely.")
    parser.add_argument("--video_file", type=str, required=False,default="/mnt/bn/luoruipu-disk/LLaVa-personal/save_images/live_frame/1")
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="v1")
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    inference(args)

    

