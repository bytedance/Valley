import pathlib
from peft import get_peft_model, LoraConfig, TaskType
import torch
import transformers
from transformers import Trainer, TrainerCallback
from valley.train.trainner import LLMCallback
from valley.model.valley_model import ValleyLlamaForCausalLM
from valley.util.data_util import smart_tokenizer_and_embedding_resize, safe_save_model_for_hf_trainer
from valley.data.dataset import make_video_supervised_data_module
from valley.util.config import *
import argparse
from dataclasses import dataclass, field
from typing import Optional
import os
from valley.utils import print_trainable_params


os.environ['NCCL_DEBUG']=''
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    tune_llm_layer: str=field(default= None)
    patch_pooling_method: str=field(default='mean')# v1
    use_patch_importance_pooling: bool=field(default=False)# v2
    use_delta_transformer: bool=field(default=False)# v3


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    fashion_data_path: str = field(default = None,
                            metadata={"help": "Path to the new construct training data."})
    video_data_path:str = field(default = None,
                            metadata={"help": "Path to the video training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    sep_image_conv_front: bool = False
    image_token_len: int = 0
    eval_num: int = 400
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    fashion_image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    num_image: int=field(default=4)
    multi_image: bool=field(default=True)
    multi_image_mode: str=field(default='concatenate')
    use_fashion: bool = field(default=False)
    fast_epoch: bool = field(default=False)
    conv_mode:str = field(default = 'v1')
    only_mask_system: str = field(default= True)
    project_name: str = field(default='valley')

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_patch_pooling_matrix: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    vis_lora: bool = field(default=False)
    lora_lr: float = field(default=None)
    lora_save_strategy: str=field(default = 'no')
    prediction_file_name: Optional[str] = field(
        default=None,
        metadata={"help": ("The `prediction_file_name` to be use for output results")},
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    deepspeed: str = field(default=None)
    output_dir: str = field(default='./checkpoints')
    lora:str = field(default=False)

def train(args):
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file(args.conf,allow_extra_keys=True)
    training_args.learning_rate = float(training_args.learning_rate)
    os.environ['WANDB_PROJECT'] = data_args.project_name

    model = ValleyLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens({
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        })

    tokenizer.add_tokens([DEFAULT_VIDEO_FRAME_TOKEN,
                          DEFAULT_VI_START_TOKEN,
                          DEFAULT_VI_END_TOKEN], 
                          special_tokens=True)

    if model_args.vision_tower is not None:
        model_vision_dict = model.get_model().initialize_vision_modules(
            vision_tower=model_args.vision_tower,
            mm_vision_select_layer=model_args.mm_vision_select_layer,
            pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter,
            use_patch_importance_pooling = model_args.use_patch_importance_pooling,
            use_delta_transformer = model_args.use_delta_transformer
        )
        dtype = torch.float32
        if training_args.fp16:
            dtype = torch.float16
        if training_args.bf16:
            dtype = torch.bfloat16
        model.get_model().vision_tower.to(dtype=dtype, device=training_args.device)
        vision_config = model_vision_dict['vision_config']

        data_args.image_token_len = model_vision_dict['image_token_len']
        data_args.image_processor = model_vision_dict['image_processor']
        data_args.is_multimodal = True

        

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        vision_config.use_im_start_end = training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.sep_image_conv_front = data_args.sep_image_conv_front
        model.initialize_vision_tokenizer(tokenizer=tokenizer)


    if training_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.lora:
        target_modules=['model.layers.'+str(i)+'.'+ k for i in range(40) for k in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj","mlp.down_proj","mlp.up_proj"]]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=target_modules
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter
    if training_args.tune_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        model.get_model().orig_embeds_params = [model.get_input_embeddings().weight.data.clone().to(device=training_args.device)]
        for p in model.get_input_embeddings().parameters():
            p.requires_grad = True
        for p in model.get_output_embeddings().parameters():
            p.requires_grad = False
    
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
    
    

    data_module = make_video_supervised_data_module(tokenizer=tokenizer,
                                            data_args=data_args)
    if training_args.lora:
        callback_class =  LLMCallback
    else:
        callback_class =  TrainerCallback

    
    trainer = Trainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    callbacks=[callback_class],
                    **data_module, 
                    )

    print_trainable_params(model)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        # Lora model is not support this resume branch, make sure your lora out_dir is empty.
        print('resume')
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str,
                        default="valley/configs/experiment/valley_debug.yaml")
    args = parser.parse_args()
    train(args)
