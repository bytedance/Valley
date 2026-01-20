import copy
import json
import os
import math
import numpy as np
from transformers import (
    TrainerCallback,
    TrainingArguments,
)
from torch import nn
import datasets
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset
from transformers import Seq2SeqTrainer, is_datasets_available, PreTrainedModel
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from valley.utils import get_logger
import torch.distributed as dist
from transformers.trainer_utils import EvalPrediction
import torch
import re
from valley.util.data_util import  KeywordsStoppingCriteria
from transformers.data.data_collator import DataCollator
import evaluate
logger = get_logger("Trainer")

class LLMCallback(TrainerCallback):
    "A callback that output infomation and do some operators"

    
    def output_log(self, args: TrainingArguments, state: TrainerState):
        def loss_log(data):
            try:
                loss_ = data["loss"]
                learning_rate_ = data["learning_rate"]
                step_ = data["step"]
                loss_log_str = f"step: {step_:<8} || learning_rate: {learning_rate_:<25} || loss: {loss_:<10}"
            except:
                loss_log_str = json.dumps(data)
            return loss_log_str

        output_file = os.path.join(args.output_dir, "trainer.log")
        log_history = map(loss_log, state.log_history)
        with open(output_file, "w") as f:
            for line in log_history:
                f.write(line + "\n")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # TODO: support deepspeed zero3 save extra weights not all llm weights
        if args.lora_save_strategy == 'steps' and state.global_step%args.save_steps == 0:
            self.output_log(args, state)
            peft_str = "peft"
            model_ = kwargs["model"]
            save_number = str(state.global_step)
            if (
                args.lora
                and peft_str in str(type(model_))
                and not is_deepspeed_zero3_enabled()
            ):
                # if model is peft-based, save the extra weights, zero3 not supprot
                epoch = "steps_" + save_number
                output_dir = os.path.join(args.output_dir, epoch)
                os.makedirs(output_dir, exist_ok=True)
                model_.save_pretrained(output_dir)
            if (
                args.lora
                and peft_str in str(type(model_))
                and args.tune_mm_mlp_adapter
                and not is_deepspeed_zero3_enabled()
            ):
                epoch = "steps_" + save_number
                output_dir = os.path.join(args.output_dir, epoch)
                os.makedirs(output_dir, exist_ok=True)
                model_.base_model.model.save_pretrained(output_dir)
                
        return super().on_step_end(args, state, control, **kwargs)
    
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # TODO: support deepspeed zero3 save extra weights not all llm weights
        if args.lora_save_strategy == 'epoch' :
            self.output_log(args, state)
            peft_str = "peft"
            model_ = kwargs["model"]
            save_number = str(state.global_step)
            if (
                args.lora
                and peft_str in str(type(model_))
                and not is_deepspeed_zero3_enabled()
            ):
                # if model is peft-based, save the extra weights, zero3 not supprot
                epoch = "epoch_" + save_number
                output_dir = os.path.join(args.output_dir, epoch)
                os.makedirs(output_dir, exist_ok=True)
                model_.save_pretrained(output_dir)
            if (
                args.lora
                and peft_str in str(type(model_))
                and args.tune_mm_mlp_adapter
                and not is_deepspeed_zero3_enabled()
            ):
                epoch = "epoch_" + save_number
                output_dir = os.path.join(args.output_dir, epoch)
                os.makedirs(output_dir, exist_ok=True)
                model_.base_model.model.save_pretrained(output_dir)
        return super().on_epoch_end(args, state, control, **kwargs)

    def merge_files(self, prediction_file_name):
        from pathlib import Path
        old_files = list(Path(prediction_file_name).parent.glob('*.worker*'))
        prediction_file_name = '.'.join(str(old_files[0]).split('.')[:-1])
        metrics = []
        with open(prediction_file_name, "w") as writer:
            for file_name in old_files:
                with open(file_name, "r") as reader:
                    for line in reader:
                        metrics.append(torch.tensor(self.compute_dev_metric(json.loads(line))))
                        writer.write(f"{line}")
            metrics = torch.mean(torch.stack(metrics, dim = 1 ),dim=1)
            bleu, rouge1, rouge2, rougeL, rougeLsum, bert_score_precision, bert_score_recall, bert_score_f1 = metrics.numpy().tolist()
        with open(prediction_file_name.replace('.jsonl','_metric.txt'), "w") as metric_writer:
            metric_str = json.dumps([bleu, rouge1, rouge2, rougeL, rougeLsum, bert_score_precision, bert_score_recall, bert_score_f1])
            metric_writer.write(metric_str)
        for file in old_files:
            os.system(f"rm {file}")
        return dict(bleu = bleu, 
                    rouge1 = rouge1, 
                    rouge2 = rouge2, 
                    rougeL = rougeL, 
                    rougeLsum = rougeLsum, 
                    bert_score_precision = bert_score_precision, 
                    bert_score_recall = bert_score_recall, 
                    bert_score_f1 = bert_score_recall)

    def compute_dev_metric(self,d):
        bert_score = d['bert_score']
        metric_score = d['metric_score']
        bleu = sum([metric['bleu'] for metric in metric_score])/len(metric_score) if len(metric_score)!= 0 else 0
        rouge1 = sum([metric['rouge1'] for metric in metric_score])/len(metric_score)if len(metric_score)!= 0 else 0
        rouge2 = sum([metric['rouge2'] for metric in metric_score])/len(metric_score)if len(metric_score)!= 0 else 0
        rougeL = sum([metric['rougeL'] for metric in metric_score])/len(metric_score)if len(metric_score)!= 0 else 0
        rougeLsum = sum([metric['rougeLsum'] for metric in metric_score])/len(metric_score)if len(metric_score)!= 0 else 0
        bert_score_precision = sum(bert_score['precision'])/len(bert_score['precision']) if len(bert_score['precision'])!= 0 else 0
        bert_score_recall = sum(bert_score['recall'])/len(bert_score['recall']) if len(bert_score['recall'])!= 0 else 0
        bert_score_f1 = sum(bert_score['precision'])/len(bert_score['precision']) if len(bert_score['precision'])!= 0 else 0
        return bleu, rouge1, rouge2, rougeL, rougeLsum, bert_score_precision, bert_score_recall, bert_score_f1 

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if dist.get_rank() == 0:
            # due to multiprocess, just do evaluation in rank 0
            metric = self.merge_files(args.prediction_file_name)
            kwargs['metrics'].update(metric)
            control.should_log = True
        else:
            control.should_log = False
        return super().on_evaluate(args, state, control, **kwargs)

    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        **kwargs,
    ):
        if dist.get_rank() == 0:
                # due to multiprocess, just do evaluation in rank 0
            self.merge_files(args.prediction_file_name)

        return super().on_predict(args, state, control, metrics, **kwargs)
    

class ValleyTrainer(Seq2SeqTrainer):
    def __init__(self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,):

        self.clf_metrics = evaluate.combine(["bleu", "rouge"])
        self.bertscore = evaluate.load("bertscore")

        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics,)
    
    def log(self, logs: Dict[str, float], eval = False) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if not eval:
            if self.state.epoch is not None:
                logs["epoch"] = round(self.state.epoch, 2)

            # logs['lora_lr'] = self.optimizer.param_groups[0]['lr']
            # logs['other_lr'] = self.optimizer.param_groups[1]['lr']
            output = {**logs, **{"step": self.state.global_step}}
            self.state.log_history.append(output)
            self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
        else:
            if self.state.epoch is not None:
                logs["epoch"] = round(self.state.epoch, 2)
            self.state.log_history.append(output)
            self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def create_optimizer(self,):
        if self.args.lora and self.args.tune_mm_mlp_adapter:
            from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
            from transformers.trainer_pt_utils import get_parameter_names
            decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if ('lora' in n and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    'lr': float(self.args.lora_lr) if self.args.lora_lr else self.args.learning_rate
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n in decay_parameters and 'lora' not in n and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls, optimizer_kwargs = ValleyTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        else:
            self.optimizer = super().create_optimizer()
        return self.optimizer

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        results = super().evaluate(
            eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs
        )
        if dist.get_rank() == 0:
            # due to multiprocess, just do evaluation in rank 0
            # print(results)
            self.log(results)
        return results


    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        # logger.info(f"rank: {dist.get_rank()}-{inputs['input_ids'].size()}-{inputs['ground_truth_labels'].size()}")

        # evalset is format as input_ids, labels, and label_index, labels represent each turn converation length, and label_index is assistent reponse index
        # inputs:{ 'input_ids', 'attention_mask', 'labels', 'images', 'label_index' }
        
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )
        
        turn_number = len(inputs['label_index'][0])
        inputs['labels'] = tuple(inputs['labels'][0].numpy().tolist())
        input_ids_split = torch.split(inputs['input_ids'], inputs['labels'], dim = 1)

        system_input_id = input_ids_split[0]
        # human input list, and the last item is <s>###, need to be ignore
        
        human_input_id_list = [input_id for i,input_id in enumerate(input_ids_split) if i%2==1]
        begin_ids = human_input_id_list[-1]
        human_input_id_list = human_input_id_list[:-1]
        response_input_id_list = [input_id for i,input_id in enumerate(input_ids_split) if i%2==0 and i!=0]
        assert len(human_input_id_list) == len(response_input_id_list)
        if len(response_input_id_list)>5:
            kwargs = {"device": self.args.device}
            generated_tokens = torch.tensor([1]).to(**kwargs)
            loss = None
            labels = None
            return loss, generated_tokens, labels
        last_id = system_input_id

        generated_tokens = []
        gd_truth_tokens = []
        for turn_index in range(turn_number):
            input_ids = torch.concat([last_id, human_input_id_list[turn_index], begin_ids],dim=1)
            if input_ids.shape[1] > self.args.generation_max_length:
                gd_truth_this_turn = response_input_id_list[turn_index][:,1:]
                gd_truth_tokens.append(gd_truth_this_turn[0])
                generated_tokens.append(torch.tensor([0]))
                continue
            inputs = dict(
                input_ids = input_ids,
                attention_mask = torch.ones_like(input_ids),
                images = inputs['images']
            )
            inputs = self._prepare_inputs(inputs)

            # XXX: adapt synced_gpus for fairscale as well
            # Priority (handled in generate):
            # gen_kwargs > model.generation_config > default GenerationConfig()
            gen_kwargs = self._gen_kwargs.copy()
            if (
                gen_kwargs.get("max_length") is None
                and gen_kwargs.get("max_new_tokens") is None
            ):
                gen_kwargs["max_length"] = self.model.config.max_length
            gen_kwargs["num_beams"] = (
                gen_kwargs["num_beams"]
                if gen_kwargs.get("num_beams") is not None
                else self.model.config.num_beams
            )
            default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
            gen_kwargs["synced_gpus"] = (
                gen_kwargs["synced_gpus"]
                if gen_kwargs.get("synced_gpus") is not None
                else default_synced_gpus
            )

            gen_kwargs['stopping_criteria'] =  [KeywordsStoppingCriteria(['###'], self.tokenizer, inputs['input_ids'])]
            outputs = self.model.generate(**inputs, **gen_kwargs, return_dict_in_generate=True)
            # because upper code has implement the begin token, so don't need to add ,so [input_ids.shape[1]:-1]
            outputs_ids = outputs.sequences[:,input_ids.shape[1]:]
            gd_truth_this_turn = response_input_id_list[turn_index][:,1:]
            gd_truth_tokens.append(gd_truth_this_turn[0])
            generated_tokens.append(outputs_ids[0])
            # add generate token to tail of last id
            last_id = torch.concat([input_ids,outputs_ids.cpu()],dim=1)

        self._output_generate_results(
            generated_tokens, gd_truth_tokens
        )
        kwargs = {"device": self.args.device}
        generated_tokens = torch.tensor([1]).to(**kwargs)
        loss = None
        labels = None
        return loss, generated_tokens, labels

    def _pad_tensors_to_max_len(self, tensor, max_length):
        return super()._pad_tensors_to_max_len(tensor, max_length)

    def decode(self, token_list):
        ignore_tokens = [
            self.tokenizer.bos_token,
            self.tokenizer.eos_token,
            self.tokenizer.pad_token,
            "\n",
            # "\x20",
        ]
        sub_re = re.compile("|".join(ignore_tokens))
        return list(map(lambda x: sub_re.sub("", self.tokenizer.decode(x)), token_list))

    def _output_generate_results(
        self,
        generated_tokens,
        gd_truth
    ):
        """output the greneted results to target file

        Parameters
        ----------
        generated_tokens : List
            generated tokens
        gd_truth : List
            the ground truth, by default None
        Returns
        ----------
        _type_
            _description_
        """
        generate_response_list = []
        gd_truth_list = []
        metric_score = []
        for generate_str,gd_truth_str in zip(generated_tokens,gd_truth):
            try:
                generate_response_list.append(self.tokenizer.decode(generate_str))
                gd_truth_list.append(self.tokenizer.decode(gd_truth_str))
                if generate_response_list[-1].strip() == "":
                    generate_response_list[-1] = 'aaaaaaaaaa'
                metric_score.append(self.clf_metrics.compute(predictions=[generate_response_list[-1]], references=[gd_truth_list[-1]]))
            except:
                print(generate_response_list[-1])
                continue
        bert_score = self.bertscore.compute(predictions=generate_response_list, references=gd_truth_list, lang='en')
        
        assert len(generate_response_list) == len(gd_truth_list)

        json_arr = []
        
        json_arr.append(
            dict(
                generate_response=generate_response_list,
                ground_truth=gd_truth_list,
                bert_score = bert_score,
                metric_score = metric_score
            )
        )

        self.jsonl_write(json_arr)

    def jsonl_write(self, json_arr: Dict[str, str]):
        """jsonl write

        Parameters
        ----------
        json_arr : Dict[str, str]
            json dict list
        output_file : str
            the output file
        """
        rank = dist.get_rank()
        json_str_arr = map(json.dumps, json_arr)
        global_step = 'step'+str(self.state.global_step)
        path = self.args.prediction_file_name.split('/')
        path[-1] = global_step + '_' + path[-1]
        path = '/'.join(path)
        output_file = path.replace(".jsonl", f".jsonl.worker{rank}")
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as writer:
            for line in json_str_arr:
                writer.write(f"{line}\n")
