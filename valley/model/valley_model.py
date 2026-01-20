from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, CLIPImageProcessor, CLIPVisionModel

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from valley.util.data_util import load_video
from valley.util.data_util import  KeywordsStoppingCriteria
from tokenizers import AddedToken
from valley.util.config import *


class ValleyConfig(LlamaConfig):
    model_type = "valley"

class ValleyLlamaModel(LlamaModel):
    config_class = ValleyConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):
        super(ValleyLlamaModel, self).__init__(config)

        self.patch_pooling_method = "mean"

        if hasattr(config, "mm_vision_tower"):
            # HACK: for FSDP
            # self.vision_tower = [CLIPVisionModel.from_pretrained(config.mm_vision_tower)]
            if 'chinese' in config.mm_vision_tower:
                from transformers import ChineseCLIPVisionModel as CLIPVisionModel
                from transformers import ChineseCLIPImageProcessor as CLIPImageProcessor
            else:
                from transformers import CLIPVisionModel, CLIPImageProcessor

            self.vision_tower = CLIPVisionModel.from_pretrained(config.mm_vision_tower)
            
        if hasattr(config, "use_patch_importance_pooling") and config.use_patch_importance_pooling:
            print('using temporal linear pooling')
            self.pooling_layer = nn.Linear(self.config.hidden_size * 256, 1)
            self.patch_pooling_method = "temporal_importance"
        
        if hasattr(config, "use_delta_transformer") and config.use_delta_transformer:
            print('using temporal transformer delta adding')
            self.transforemr_adding_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8, batch_first=True)
            self.transformer_delta_encoder = nn.TransformerEncoder(self.transforemr_adding_layer, num_layers=1)
            self.patch_pooling_method = "temporal_transformer"
            # self.position_matrix = torch.nn.Parameter(self.getPositionEncoding(seq_len=2048, d = config.hidden_size))
            self.position_matrix = torch.nn.Parameter(torch.zeros(2048,config.hidden_size))
            self.position_matrix.requires_grad = False

        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
            print(config.mm_hidden_size, config.hidden_size)


    def initialize_vision_modules(self, vision_tower, mm_vision_select_layer,
                                  pretrain_mm_mlp_adapter=None, use_patch_importance_pooling=False, use_delta_transformer=False):
        self.config.mm_vision_tower = vision_tower

        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        if not hasattr(self, 'vision_tower'):
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        else:
            vision_tower = self.vision_tower
        vision_tower.requires_grad_(False)
        # vision_tower = vision_tower.to(torch.float16)
        self.vision_tower = vision_tower

        vision_config = vision_tower.config
        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.use_patch_importance_pooling = use_patch_importance_pooling
        self.config.use_delta_transformer = use_delta_transformer
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        if not hasattr(self, 'pooling_layer') and use_patch_importance_pooling:
            self.pooling_layer = nn.Linear(self.config.hidden_size * 256, 1)
            self.patch_pooling_method = "temporal_importance"

        if not hasattr(self, 'transformer_delta_encoder') and use_delta_transformer:
            self.transforemr_adding_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=8, batch_first=True)
            self.transformer_delta_encoder = nn.TransformerEncoder(self.transforemr_adding_layer, num_layers=1)
            self.patch_pooling_method = "temporal_transformer"
            self.position_matrix = torch.nn.Parameter(self.getPositionEncoding(seq_len=2048, d = self.config.hidden_size))
            self.position_matrix.requires_grad = False

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config
        )
    def getPositionEncoding(self, seq_len=2048, d=5120, n=10000):
        P = torch.zeros((seq_len, d))
        for k in range(seq_len):
            for i in torch.arange(int(d/2)):
                denominator = torch.pow(n, 2*i/d)
                P[k, 2*i] = torch.sin(k/denominator)
                P[k, 2*i+1] = torch.cos(k/denominator)
        return P

    def text_importance_pooling(self,patch_feature):# 8, 256, 5120
        # print(patch_feature.shape)
        patch_feature_flatten = torch.flatten(patch_feature,start_dim=1)
        score = nn.functional.softmax(self.pooling_layer(patch_feature_flatten), dim=0)
        # print(score.shape)
        score = score.unsqueeze(2)
        patch_feature = score*patch_feature
        patch_feature = torch.sum(patch_feature, dim=0)
        return patch_feature
    
    def temporal_tranforemr_delta_adding(self,patch_feature):# 8, 256, 5120
        patch_feature = patch_feature.permute(1,0,2) # 256,8,5120
        sequence_length = patch_feature.shape[1]
        patch_number = patch_feature.shape[0]
        position_embedding = self.position_matrix[:sequence_length,:].unsqueeze(0).type_as(patch_feature)# 1,8,5120
        position_embedding = position_embedding.repeat(patch_number,1,1).to(patch_feature.device) # 256,8,5120
        patch_feature_pos = patch_feature+position_embedding
        patch_feature_delta = self.transformer_delta_encoder(patch_feature_pos)[:,-1,:]
        patch_feature_mean = torch.mean(patch_feature, dim=1) # 256 , 4096
        patch_feature = patch_feature_delta + patch_feature_mean
        return patch_feature

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # HACK: replace back original embeddings for Valley pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            # print(torch.max(input_ids))
            # print(input_ids.shape)
            # print(torch.min(input_ids))
            # print(self.embed_tokens.weight.shape)
            inputs_embeds = self.embed_tokens(input_ids)
            

        vision_tower = getattr(self, 'vision_tower', None)
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            # vision_tower = vision_tower[0]  # HACK: for FSDP
            with torch.no_grad():
                if type(images) is list:
                    # variable length images
                    image_features = []
                    for image in images:
                        image_forward_out = vision_tower(image, output_hidden_states=True)
                        select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                        select_hidden_state = image_forward_out.hidden_states[select_hidden_state_layer]
                        image_feature = select_hidden_state[:, :]
                        image_features.append(image_feature)
                else:
                    image_features = []
                    for batch_id in range(len(images)):
                        image_forward_outs = vision_tower(images[batch_id], output_hidden_states=True)# 8,3,224,224
                        select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                        select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                        image_features.append(select_hidden_state[:, :])
                    image_features = torch.stack(image_features)


            if type(images) is list:
                image_features = [self.mm_projector(image_feature) for image_feature in image_features]
            else:
                image_features = self.mm_projector(image_features)

            dummy_image_features = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0 # this index is for batch 
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue

                cur_image_features = image_features[cur_image_idx]
                # patch pooling method ( mean, max, text relative pooling )
                if self.patch_pooling_method == 'mean':
                    mean_image_features = torch.mean(cur_image_features[:,1:,:],dim=0) # 256 , 4096
                elif self.patch_pooling_method == 'max':
                    mean_image_features = torch.max(cur_image_features[:,1:,:],dim=0)[0] # 256 , 4096
                elif self.patch_pooling_method == 'temporal_importance':
                    mean_image_features = self.text_importance_pooling(cur_image_features[:,1:,:]) # 256 , 4096
                elif self.patch_pooling_method == 'temporal_transformer':
                    mean_image_features = self.temporal_tranforemr_delta_adding(cur_image_features[:,1:,:]) # 256 , 4096

                frame_image_features = cur_image_features[:,0,:]# frame_length, 4096
                num_patches = mean_image_features.shape[0]
                # print(mean_image_features.shape)

                if (cur_input_ids == vision_tower.config.im_start_token).sum() != (cur_input_ids == vision_tower.config.im_end_token).sum():
                    raise ValueError("The number of im_start_token and im_end_token should be the same")
                image_start_tokens = torch.where(cur_input_ids == vision_tower.config.im_start_token)[0]
                multi_iamge_index = 0 # this index is for multi_image
                cur_new_input_embeds = cur_input_embeds.clone() # to save the new embed
                for image_start_token_pos in image_start_tokens: #this loop is for multi_image in one piece
                    cur_image_features = mean_image_features.to(device=cur_input_embeds.device)
                    if cur_input_ids[image_start_token_pos + num_patches + 1] != vision_tower.config.im_end_token:
                        raise ValueError("Seems that the image is cut.")
                    cur_new_input_embeds = torch.cat((cur_new_input_embeds[:image_start_token_pos+1], cur_image_features, cur_new_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
                    multi_iamge_index+=1

                try:
                    if (cur_input_ids == vision_tower.config.vi_start_token).sum() != (cur_input_ids == vision_tower.config.vi_end_token).sum():
                        raise ValueError("The number of vi_start_token and vi_end_token should be the same")
                    video_start_tokens = torch.where(cur_input_ids == vision_tower.config.vi_start_token)[0]
                    num_frame = frame_image_features.shape[0]
                    assert (cur_input_ids == vision_tower.config.vi_frame_token).sum() == num_frame
                    cur_video_input_embeds = cur_new_input_embeds.clone() # to save the new embed
                    for video_start_token_pos in video_start_tokens: #this loop is for multi_image in one piece
                        frame_image_features = frame_image_features.to(device=cur_input_embeds.device)
                        if cur_input_ids[video_start_token_pos + num_frame + 1] != vision_tower.config.vi_end_token:
                            raise ValueError("Seems that the image is cut.")
                        cur_video_input_embeds = torch.cat((cur_video_input_embeds[:video_start_token_pos+1], frame_image_features, cur_video_input_embeds[video_start_token_pos + num_frame + 1:]), dim=0)
                except:
                    cur_video_input_embeds = cur_new_input_embeds.clone()
                new_input_embeds.append(cur_video_input_embeds)
                cur_image_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(ValleyLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class ValleyLlamaForCausalLM(LlamaForCausalLM):
    config_class = ValleyConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = ValleyLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, tokenizer):
        vision_config = self.get_model().vision_tower.config
        vision_config.use_im_start_end = True
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN,DEFAULT_VIDEO_FRAME_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VI_START_TOKEN,DEFAULT_VI_END_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        vision_config.vi_start_token, vision_config.vi_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN])
        vision_config.vi_frame_token = tokenizer.convert_tokens_to_ids(DEFAULT_VIDEO_FRAME_TOKEN)
        
        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]

    def build_inputs(self,tokenizer, messages):
        prompt = ''
        for m in messages:
            if m['role'] == 'system':
                prompt += m['content'] +'\n\n' + '###'
            elif m['role'] == 'user':
                replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + \
                DEFAULT_IM_END_TOKEN + DEFAULT_VI_START_TOKEN + \
                DEFAULT_VIDEO_FRAME_TOKEN * 8 + DEFAULT_VI_END_TOKEN
                if '<video>'  in m['content'] or '<image>' in m['content']:
                    message = m['content'].replace('<video>',replace_token)
                    message = message.replace('<image>',replace_token)
                    prompt += ' ' + 'Human' + ": " + message+' \n' + '###'
            elif m['role'] == 'assistent':
                prompt += ' ' + 'Assistent' + ": " + m['content']+' \n' + '###'
            else:
                raise ValueError("Role is only suport \"assistent\", \"human\" and \"system\".")
        if DEFAULT_IM_START_TOKEN not in prompt:
            raise ValueError("You need to specify the <video> token in the query")
        tokenizer.padding_side = 'left'
        input_id = tokenizer([prompt], padding=True)
        return input_id
    
    def process_response(self,outputs):
        output = []
        for i, out in enumerate(outputs):
            while True:
                cur_len = len(out)
                out = out.strip()
                for pattern in ['###', 'Assistant:', 'Response:', 'Valley:']:
                    if out.startswith(pattern):
                        out = out[len(pattern):].strip()
                if len(out) == cur_len:
                    break
            try:
                index = out.index('###')
            except ValueError:
                out += '###'
                index = out.index("###")
            out = out[:index].strip()
            output.append(out)
        return output

    @torch.no_grad()
    def completion(self, tokenizer, video: str, message: list, gen_kwargs:dict, device):
        inputs = self.build_inputs(tokenizer, message)
        input_ids = torch.as_tensor(inputs.input_ids).to(device)
        images = load_video(video)
        images = images.permute(1, 0, 2, 3)
        images = images.unsqueeze(0).half().to(device)
        stopping_criteria = KeywordsStoppingCriteria(['###'], tokenizer, input_ids)
        output_ids = self.generate(input_ids = input_ids, images = images, stopping_criteria=[stopping_criteria],**gen_kwargs)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        response = self.process_response(outputs)
        return response
    
AutoConfig.register("valley", ValleyConfig)
AutoModelForCausalLM.register(ValleyConfig, ValleyLlamaForCausalLM)