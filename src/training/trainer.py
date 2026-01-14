import os
import torch
import torch.nn as nn

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
import copy
from contextlib import nullcontext
from peft import PeftModel
from typing import Optional
import numpy as np
from transformers.processing_utils import ProcessorMixin
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from typing import Any, Dict, Optional, Tuple, Union

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

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters
                
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

        all_tensors = [torch.empty_like(t) for _ in range(self.args.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
    
    def sync_loss(self, loss: torch.Tensor):
        # 确保 loss 是标量张量
        if loss.dim() > 0:
            loss = loss.mean()  # 简化处理
        
        # 创建一个张量用于同步
        sync_loss = loss.clone()
        dist.all_reduce(sync_loss, op=dist.ReduceOp.SUM)
        
        # 如果需要平均值而不是总和，可以除以 world_size
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
            if not (thw == -1).all()  # 跳过特殊标记
        ]
        if image_grid_thw:  # 如果存在有效数据
            c_inputs['image_grid_thw'] = torch.cat(image_grid_thw, dim=0)
        else:
            c_inputs['image_grid_thw'] = None
        
        return q_inputs, c_inputs


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

            # step2: compute query, corresponding passage (both with gradient) and calculate loss, and perform backward to get gradient for parameters
            for i in range(0, num_microbatches):
                if self.state.global_step == 0:
                    logger.info(f"* with gradient forward # {i}")
                
                # we only want to synchronize gradient in the last iteration, because for non-last iteration, to all-reduce the gradient is trivial, a waste of time.
                if i == (num_microbatches - 1):
                    context = nullcontext()
                    if self.state.global_step == 0:
                        logger.info("    this is the last micro-batch iteration, no_sync is disabled.")
                else:
                    context = self.accelerator.no_sync(model) # <- is a feature of accelerate, and for pure DDP, it should be model.no_sync()
                    if self.state.global_step == 0:
                        logger.info("    this is not the last micro-batch, no_sync is enabled.")
                
                with context:
                    # step 2a: prepare micro-batch again
                    q_inputs, c_inputs = self.prepare_model_input(inputs, i, micro_bsz)
                    
                    with self.compute_loss_context_manager():
                        rng_state_ = rng_state_cache.pop(0)
                        torch.set_rng_state(rng_state_) # for dropout..
                        q_outputs, c_outputs = self.model(q_inputs=q_inputs, c_inputs=c_inputs)

                        q_causal_loss = q_outputs.loss
                        c_causal_loss = c_outputs.loss

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
                        loss = loss * self.world_size + (q_causal_loss + c_causal_loss) / 3
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
            
            # Compute accuracy of global mini-batch, only once is fine.
            # with torch.no_grad():
            #     predicted_indices = torch.argmax(scores, axis=1)
            #     accuracy = torch.mean((predicted_indices == target).float())
            # self.metric_hook["accuracy"].append(accuracy.item())

        else:
            with self.compute_loss_context_manager():
                q_inputs, c_inputs = self.prepare_model_input(inputs)
                # q_outputs = self.model(**q_inputs)
                # q_reps_ = q_outputs.reps_

                # with torch.no_grad():
                #     c_outputs = self.model(**c_inputs)
                #     c_reps_ = c_outputs.reps_

                q_outputs, c_outputs = self.model(q_inputs=q_inputs, c_inputs=c_inputs)
                q_reps_ = q_outputs.reps_[:, -1, :]
                c_reps_ = c_outputs.reps_[:, -1, :]

                q_causal_loss = self.sync_loss(q_outputs.loss)
                c_causal_loss = self.sync_loss(c_outputs.loss)

                q_reps = self.dist_gather_tensor(q_reps_) # <- all of the gathered large tensor do not have gradient, except the partition of this gpu
                c_reps = self.dist_gather_tensor(c_reps_) # <- all of the gathered large tensor do not have gradient, except the partition of this gpu

                q_reps = F.normalize(q_reps, p=2, dim=-1)
                c_reps = F.normalize(c_reps, p=2, dim=-1)

                
                scores = torch.matmul(q_reps, c_reps.transpose(0, 1))
                logger.info(f"scores.shape = {scores.shape}")
                scores = scores / self.args.softmax_temperature
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                
                loss = F.cross_entropy(scores, target, reduction='mean')
                
                # In distributed training, global CE backward will give one gpu its own gradient, 
                # But DDP will average the gradient of all gpus, here, an extra mean is introduced. 
                # then, we need to multiply loss by self.world_size to make the loss the same AS IF it is trained on a single LARGE GPU
                loss = loss * self.world_size + (q_causal_loss + c_causal_loss) / 2
                
                with torch.no_grad():
                    predicted_indices = torch.argmax(scores, axis=1)
                    accuracy = torch.mean((predicted_indices == target).float())
            
            # hacked metric logging:
            # self.metric_hook["accuracy"].append(accuracy.item())

            self.accelerator.backward(loss)
    
        return loss.detach() / self.args.gradient_accumulation_steps