import os
import torch
import torch.nn as nn
import math

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    is_peft_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
)
import safetensors
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from peft import PeftModel
from typing import Optional
import numpy as np
from transformers.processing_utils import ProcessorMixin
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from typing import Any, Dict, Optional, Tuple, Union
from transformers.trainer import *

class HardNegativePool:
    def __init__(self, pool_size=512, temperature=0.05):
        self.pool_size = pool_size
        self.temperature = temperature
        self.context_queue = None
        self.hidden_dim = 3584
        self.queue_ptr = 0  # 环形队列指针
        self.full = -1
        # self.context_queue = torch.randn(64, self.hidden_dim).cuda().bfloat16()
        self._init_queue()

    def _init_queue(self):
        self.context_queue = torch.randn(self.pool_size, self.hidden_dim).cuda().bfloat16()
        self.context_queue = F.normalize(self.context_queue, dim=-1)

    def update(self, negatives):
        batch_size = negatives.size(0)
        ptr = self.queue_ptr
        
        
        if ptr + batch_size > self.pool_size:
            self.full = 1
            remain = self.pool_size - ptr
            self.context_queue[ptr:] = negatives[:remain]
            self.context_queue[:batch_size - remain] = negatives[remain:]
            self.queue_ptr = (self.queue_ptr + batch_size) % self.pool_size
        else:
            self.context_queue[ptr:ptr+batch_size] = negatives  # 直接插入
            self.queue_ptr += batch_size  # 更新队列指针

    # def get_negatives(self, q_reps, pos, hard_num):
    #     with torch.no_grad():
    #         scores = torch.matmul(q_reps, self.context_queue.transpose(0, 1))  # [B, B]

    #         neg_scores = scores.clone()
    #         # diagonal_elements = neg_scores.diagonal()
    #         # closest_scores = torch.stack([row[torch.topk(torch.abs(row - diag), 2, largest=False).indices] 
    #         #                 for row, diag in zip(neg_scores, diagonal_elements)])

    #         closest_negative_indices = torch.stack([torch.topk(torch.abs(row - diag), hard_num, largest=False).indices
    #                                     for row, diag in zip(neg_scores, pos)])
    #         hard_negatives = torch.gather(scores, dim=1, index=closest_negative_indices)

    #         # flattened_indices = closest_negative_indices.reshape(-1)
    #         # counts = torch.bincount(flattened_indices)
    #         # top_k_indices = torch.topk(counts, 128).indices
    #         # sorted_top_k_indices = top_k_indices.sort()[0]

    #         # batch_size = scores.size(0)
    #         # num_negatives = 128
    #         # sampled_indices = torch.randint(0, scores.size(1), (batch_size, num_negatives)).to(scores.device)
    #         # hard_negatives = torch.gather(scores, dim=1, index=sampled_indices)
    #     return hard_negatives

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class QwenTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(QwenTrainer, self).__init__(*args, **kwargs)

        self.process_rank = dist.get_rank()

        self.world_size = dist.get_world_size()
        # self.neg_pool = HardNegativePool(pool_size=1536)
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
        # self.shuffle = self.args.shuffle

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []
            layer_low_lr = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]
            if self.args.layer_low_lr is not None:
                lr_mapper['low_lr'] = self.args.layer_low_lr
                layer_low_lr = [name for name, _ in opt_model.named_parameters() if "fusion.layers.0" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters + layer_low_lr
                
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

                if layer_low_lr:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in layer_low_lr and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.layer_low_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in layer_low_lr and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.layer_low_lr,
                            },
                        ]
                    )
                
                if visual_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )
                
                if merger_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            self.save_model(output_dir, _internal_call=True)

            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Determine the new best metric / best model checkpoint
            if metrics is not None and self.args.metric_for_best_model is not None:
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics[metric_to_check]

                operator = np.greater if self.args.greater_is_better else np.less
                if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
                ):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

            # Save the Trainer state
            if self.args.should_save:
                # Update the `TrainerControl` state to where we are currently
                self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

        else:
            super(QwenTrainer, self)._save_checkpoint(model, trial, metrics)
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")

            supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            if not isinstance(self.model, supported_classes):
                if state_dict is None:
                    state_dict = self.model.state_dict()

                if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                    self.accelerator.unwrap_model(self.model).save_pretrained(
                        output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                    )
                else:
                    logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                    if self.args.save_safetensors:
                        safetensors.torch.save_file(
                            state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                        )
                    else:
                        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            else:
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        torch.cuda.synchronize()
        dist.barrier()

        all_tensors = [torch.empty_like(t) for _ in range(self.args.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    # def dist_gather_tensor(self, t: Optional[torch.Tensor]):
    #     if t is None or t.numel() == 0:
    #         print(f"Rank {self.process_rank}: t is empty before all_gather")
    #         return None

    #     t = t.contiguous()

    #     torch.cuda.synchronize()
    #     dist.barrier()

    #     all_tensors = [torch.empty_like(t) for _ in range(self.args.world_size)]
    #     dist.all_gather(all_tensors, t)  # collect rank data

    #     # concat a bath
    #     all_tensors = torch.cat(all_tensors, dim=0)  # [256]

    #     world_size = self.args.world_size
    #     batch_size_per_rank = all_tensors.shape[0] // world_size  # 256 // 4 = 64
    #     start_index = self.process_rank * batch_size_per_rank
    #     end_index = start_index + batch_size_per_rank
    #     gathered_tensor = all_tensors[start_index:end_index]  # get rank data

    #     print(f"Rank {self.process_rank}: Final gathered shape = {gathered_tensor.shape}")

    #     return gathered_tensor
    
    def sync_loss(self, loss: torch.Tensor):
        if loss.dim() > 0:
            loss = loss.mean()
        sync_loss = loss.clone()
        dist.all_reduce(sync_loss, op=dist.ReduceOp.SUM)
        sync_loss /= self.args.world_size
    
        return sync_loss

    def prepare_model_input(self, inputs, i=-1, micro_bsz=-1):
        if i == -1 and micro_bsz == -1:
            start_idx, end_idx = 0, inputs['input_ids'].shape[0]
        else:
            start_idx = i * micro_bsz
            end_idx = (i + 1) * micro_bsz

        q_inputs = {}
        q_inputs['input_ids'] = inputs['input_ids'][start_idx: end_idx, :]
        q_inputs['attention_mask'] = inputs['attention_mask'][start_idx: end_idx, :]
        q_inputs['labels'] = inputs['labels'][start_idx: end_idx, :]
        pixel_values = inputs['pixel_values'][start_idx: end_idx]
        pixel_values = [
            pixel for pixel in pixel_values 
            if not (pixel == 0).all()
        ]
        if pixel_values:
            q_inputs['pixel_values'] = torch.cat(pixel_values, dim=0)
        else:
            q_inputs['pixel_values'] = None

        image_grid_thw = inputs['image_grid_thw'][start_idx: end_idx]
        image_grid_thw = [
            thw for thw in image_grid_thw
            if not (thw == -1).all()
        ]
        if image_grid_thw: 
            q_inputs['image_grid_thw'] = torch.cat(image_grid_thw, dim=0)
        else:
            q_inputs['image_grid_thw'] = None

        c_inputs = {}
        c_inputs['input_ids'] = inputs['cand_input_ids'][start_idx: end_idx, :]
        c_inputs['attention_mask'] = inputs['cand_attention_mask'][start_idx: end_idx, :]
        c_inputs['labels'] = inputs['cand_labels'][start_idx: end_idx]
        c_inputs['is_cand'] = True
        pixel_values = inputs['cand_pixel_values'][start_idx: end_idx]
        pixel_values = [
            pixel for pixel in pixel_values 
            if not (pixel == 0).all()  # 跳过全零张量
        ]
        if pixel_values:  # 如果存在有效数据
            c_inputs['pixel_values'] = torch.cat(pixel_values, dim=0)
        else:
            c_inputs['pixel_values'] = None

        image_grid_thw = inputs['cand_image_thw'][start_idx: end_idx]
        image_grid_thw = [
            thw for thw in image_grid_thw
            if not (thw == -1).all()
        ]
        if image_grid_thw:
            c_inputs['image_grid_thw'] = torch.cat(image_grid_thw, dim=0)
        else:
            c_inputs['image_grid_thw'] = None
        
        return q_inputs, c_inputs
    
    def separate_input(self, inputs):
        q_inputs = {}
        q_inputs['input_ids'] = inputs['input_ids']
        q_inputs['attention_mask'] = inputs['attention_mask']
        q_inputs['pixel_values'] = inputs['pixel_values']
        q_inputs['image_grid_thw'] = inputs['image_grid_thw']

        c_inputs = {}
        c_inputs['input_ids'] = inputs['cand_input_ids']
        c_inputs['attention_mask'] = inputs['cand_attention_mask']
        c_inputs['is_cand'] = True
        c_inputs['pixel_values'] = inputs['cand_pixel_values']
        c_inputs['image_grid_thw'] = inputs['cand_image_thw']

        return q_inputs, c_inputs
    
    def label_encoder(self, labels):
        label_to_index = {
            'image': 0,
            'text': 1,
            'image,text': 2,
        }

        encoded_labels = [label_to_index.get(label) for label in labels]
        
        return encoded_labels

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        model.train()

        inputs = self._prepare_inputs(inputs)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self.args.grad_cache_enable:
            mini_bsz = self.args.per_device_train_batch_size
            micro_bsz = self.args.grad_cache_micro_batch_size
            global_micro_bsz = micro_bsz * self.world_size
            assert mini_bsz % micro_bsz == 0, "batch size should be divisible by --grad_cache_mini_batch_size"
            num_microbatches = mini_bsz // micro_bsz

            q_reps_cache = []
            c_reps_cache = []
            rng_state_cache = []

            with torch.no_grad():
                for i in range(0, num_microbatches):
                    q_inputs, c_inputs = self.prepare_model_input(inputs, i, micro_bsz)
                        
                    rng_state_ = torch.get_rng_state() # but have to save the rng_state for w/ grad forward
                    rng_state_cache.append(rng_state_)

                    q_outputs, c_outputs = self.model(q_inputs=q_inputs, c_inputs=c_inputs)
                    q_reps_ = q_outputs.reps_[:, -1, :]
                    c_reps_ = c_outputs.reps_[:, -1, :]

                    q_reps_ = self.dist_gather_tensor(q_reps_)
                    c_reps_ = self.dist_gather_tensor(c_reps_)

                    q_reps_cache.append(q_reps_)
                    c_reps_cache.append(c_reps_)

            q_reps_cache = torch.cat(q_reps_cache, dim=0)
            c_reps_cache = torch.cat(c_reps_cache, dim=0)

            for i in range(0, num_microbatches):
                if self.state.global_step == 0:
                    logger.info(f"* with gradient forward # {i}")
                
                if i == (num_microbatches - 1):
                    context = nullcontext()
                    if self.state.global_step == 0:
                        logger.info("    this is the last micro-batch iteration, no_sync is disabled.")
                else:
                    context = self.accelerator.no_sync(model) # <- is a feature of accelerate, and for pure DDP, it should be model.no_sync()
                    if self.state.global_step == 0:
                        logger.info("    this is not the last micro-batch, no_sync is enabled.")
                
                with context:
                    q_inputs, c_inputs = self.prepare_model_input(inputs, i, micro_bsz)
                    
                    with self.compute_loss_context_manager():
                        rng_state_ = rng_state_cache.pop(0)
                        torch.set_rng_state(rng_state_) # for dropout..
                        q_outputs, c_outputs = self.model(q_inputs=q_inputs, c_inputs=c_inputs)

                        q_reps_ = q_outputs.reps_[:, -1, :]
                        c_reps_ = c_outputs.reps_[:, -1, :]
                        
                        if self.state.global_step == 0:
                            logger.info(f"    process #{self.process_rank}: {i * global_micro_bsz + self.process_rank * micro_bsz} -> {i * global_micro_bsz + (self.process_rank + 1) * micro_bsz}")
                        
                        # for this, please refer to the above "Composition of q_reps_"
                        q_reps_tmp = q_reps_cache.clone()
                        q_reps_tmp[i * global_micro_bsz + self.process_rank * micro_bsz: i * global_micro_bsz + (self.process_rank + 1) * micro_bsz] = q_reps_
                        c_reps_tmp = c_reps_cache.clone()
                        c_reps_tmp[(i * global_micro_bsz + self.process_rank * micro_bsz) : (i * global_micro_bsz + (self.process_rank + 1) * micro_bsz)] = c_reps_
                        
                        # compute loss L2-norm
                        q_reps_tmp = F.normalize(q_reps_tmp, p=2, dim=-1)
                        c_reps_tmp = F.normalize(c_reps_tmp, p=2, dim=-1)
                        scores = torch.matmul(q_reps_tmp, c_reps_tmp.transpose(0, 1)) # [bs, bs]
                        if self.state.global_step == 0:
                            logger.info(f"    scores.shape = {scores.shape}")
                            logger.info(f"    softmax_temperature = {self.args.softmax_temperature}")
                            
                        scores = scores / self.args.softmax_temperature
                        
                        # one query, multiple passage (only one positive passage, others are all negatives)
                        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long) # shape [B, 1] where 1 is an index from 0 to B*n_train_docs
                        
                        loss = F.cross_entropy(scores, target, reduction='mean')
                        
                        # In distributed training, global CE backward will give one gpu its own gradient, 
                        # But DDP will average the gradient of all gpus, here, an extra mean is introduced. 
                        # then, we need to multiply loss by self.world_size to make the loss the same AS IF it is trained on a single LARGE GPU
                        loss = loss * self.world_size
                    if self.args.deepspeed is not None:
                        if self.state.global_step == 0:
                            logger.info("    You are using deepspeed+accelerate+transformers to train the model, calling self.accelerator.deepspeed_engine_wrapped.engine.backward(loss)")
                        self.accelerator.deepspeed_engine_wrapped.engine.backward(loss)
                    else:
                        if self.state.global_step == 0:
                            logger.info("    You are using DDP+accelerate+transformers to train the model.")
                        self.accelerator.backward(loss)

            if self.args.deepspeed is not None:
                self.accelerator.deepspeed_engine_wrapped.engine.step()
                if self.state.global_step == 0:
                    logger.info("    You are using deepspeed+accelerate+transformers to train the model, calling self.accelerator.deepspeed_engine_wrapped.engine.step()")
            
        # Training code for PUMA
        # Separating the calculation of query and candidate embedding may save GPU memory
        else:
            with self.compute_loss_context_manager():
                q_inputs, c_inputs = self.separate_input(inputs)
    
                q_outputs = self.model(**q_inputs)
                c_outputs = self.model(**c_inputs)
                q_reps_ = q_outputs.reps_
                c_reps_ = c_outputs.reps_

                q_reps_gather = self.dist_gather_tensor(q_reps_)
                c_reps_gather = self.dist_gather_tensor(c_reps_)

                q_reps = F.normalize(q_reps_gather, p=2, dim=-1)
                c_reps = F.normalize(c_reps_gather, p=2, dim=-1)

                current_epoch = self.state.epoch
                total_epochs = self.args.num_train_epochs
                
                norm_temp = self.args.softmax_temperature
                if self.args.training_stage == 'v1':
                    temperature = self.args.softmax_temperature
                else:
                    # hard_temperature = round(norm_temp * math.exp(-self.args.decay_rate * (current_epoch / total_epochs)), 3)
                    hard_temperature = round(norm_temp * math.exp(-self.args.decay_rate * min(0.5, (current_epoch / total_epochs))*2), 3)
                    
                scores = torch.matmul(q_reps, c_reps.transpose(0, 1))
                
                distill_loss = 0.
                
                if self.args.distill:
                    q_ori_reps = q_outputs.ori_reps
                    c_ori_reps = c_outputs.ori_reps

                    q_ret_list = q_outputs.ret_list
                    c_ret_list = c_outputs.ret_list

                    q_ret_list_gather = self.dist_gather_tensor(q_ret_list)
                    c_ret_list_gather = self.dist_gather_tensor(c_ret_list)

                    q_ori_reps_gather = self.dist_gather_tensor(q_ori_reps)
                    c_ori_reps_gather = self.dist_gather_tensor(c_ori_reps)

                    q_ori_reps = F.normalize(q_ori_reps_gather, p=2, dim=-1)
                    c_ori_reps = F.normalize(c_ori_reps_gather, p=2, dim=-1)

                    q_ret_list = F.normalize(q_ret_list_gather, p=2, dim=-1)
                    c_ret_list = F.normalize(c_ret_list_gather, p=2, dim=-1)
        
                    if self.args.mse:
                        distill_loss = (F.mse_loss(q_reps_gather, q_ori_reps_gather) + F.mse_loss(c_reps_gather, c_ori_reps_gather)) * self.args.scale

                if self.args.training_stage == 'v2' and self.args.hard_norm:
                    modality_target_ = inputs['modality_target']
                    modality_target_ = torch.tensor(self.label_encoder(modality_target_)).to(scores.device)
                    modality_target = self.dist_gather_tensor(modality_target_)

                    modality_matrix = modality_target.unsqueeze(0) == modality_target.unsqueeze(1)
                    matrix_temp = torch.where(modality_matrix, hard_temperature, norm_temp)
                    scores = scores / matrix_temp
                else:
                    scores = scores / norm_temp

                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                info_loss = F.cross_entropy(scores, target, reduction='mean')

                if self.args.mse:
                    # paper setting
                    loss = info_loss * 0.9 + distill_loss * 0.1

                    # selection - dynamic weight
                    # alpha = current_epoch / total_epochs
                    # info_weight = 0.5 + 0.4 * alpha
                    # distill_weight = 1.0 - info_weight
                    # loss = info_loss * info_weight + distill_loss * distill_weight
                    
                else:
                    loss = info_loss
            self.accelerator.backward(loss, scale_wrt_gas=False)

        return loss.detach() / self.args.gradient_accumulation_steps