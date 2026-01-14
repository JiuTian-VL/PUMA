import os
import sys
sys.path.append('')

import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
from transformers import AutoProcessor
from peft import PeftModel
import argparse
from training.constants import *
from eval.utils import hash_did, hash_qid
from training.ret_oneword import InferenceQuery
from pathlib import Path
from accelerate import Accelerator

from qwen_vl_utils import process_vision_info
from training.model_file.modeling_pre import *

def get_data_pool(data_path):
    if 'jsonl' in data_path:
        list_data_dict = []
        with open(data_path, "r") as fin:
            for line in fin:
                list_data_dict.append(json.loads(line))
    else:    
        list_data_dict = json.load(open(data_path, "r"))
    data_pool = list_data_dict
    return data_pool

def cand_collate_fn(batch):
    texts = [item['txt'] if item['txt'] is not None else "" for item in batch]
    img_paths = [item['img_path'] if item['img_path'] is not None else None for item in batch]
    ids = [item['did'] if item['did'] is not None else None for item in batch]
    return {"texts": texts, "img_paths": img_paths, 'ids': ids}

def custom_collate_fn(batch):
    messages = [item[0] for item in batch]
    ids = [item[1] for item in batch]
    return {'messages': messages, 'ids': ids}


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def load_model(args):
    global processor, model, device
    device = args.device

    kwargs = {"device_map": args.device}
    kwargs['torch_dtype'] = torch.bfloat16
    kwargs['_attn_implementation'] = 'flash_attention_2'

    if 'lora' in args.model_path.lower() and args.model_base is not None:
        # lora_cfg_pretrained = AutoConfig.from_pretrained(args.model_path)
        # if hasattr(lora_cfg_pretrained, 'quantization_config'):
        #     del lora_cfg_pretrained.quantization_config
        processor = AutoProcessor.from_pretrained(args.model_base)
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_base, low_cpu_mem_usage=True, layer=args.layer_num, **kwargs)
        
        non_lora_trainables = torch.load(os.path.join(args.model_path, 'non_lora_state_dict.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.fusion.') for k in non_lora_trainables):
            non_lora_trainables = {
                (k[6:] if k.startswith('model.') else k): v
                for k, v in non_lora_trainables.items()
                if 'visual' not in k
            }
        model.load_state_dict(non_lora_trainables, strict=False)
    
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()
    else:
        processor = AutoProcessor.from_pretrained(args.model_base)
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_path, low_cpu_mem_usage=True, **kwargs).to(device)
    model.config.compression = False
    model.config.drop = args.drop
    model.config.router = args.router

    processor.tokenizer.add_tokens(["[RET]"])
    # model.resize_token_embeddings(len(processor.tokenizer))
    model.config.RET_token_id = processor.tokenizer("[RET]", add_special_tokens=False).input_ids[0]
    if args.img_token:
        processor.tokenizer.add_tokens(["[IMG]"])
        model.config.IMG_token_id = processor.tokenizer("[IMG]", add_special_tokens=False).input_ids[0]
    return model

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def inference(dataloader, model):
    data_embed = []
    data_ids = []
    weights_list = []
    i = 0

    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device 
    is_main_process = accelerator.is_main_process

    # model.eval()
    with torch.no_grad():
        dataloader, model = accelerator.prepare(dataloader, model)
        for batch in tqdm(dataloader):

            infer_messages = batch['messages']
            texts = processor.apply_chat_template(
                infer_messages, tokenize=False, add_generation_prompt=False
            )
            image_inputs, _ = process_vision_info(infer_messages)
            
            #  image_inputs = [img_path if img_path is not None else None for img_path in img_paths]
            inputs = processor(
                text=texts,
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            
            outputs = model(**inputs)

            # hidden_states = outputs['hidden_states'][-1]  # last layer [bs, seq, hiddenstate]
            

            eos_token = accelerator.gather_for_metrics(outputs.reps_)
            eos_token = F.normalize(eos_token, dim=-1)
            # if outputs.router_weights is not None:
            #     router_weights = outputs.router_weights
            #     weights_list.append(router_weights.cpu().numpy())
            
            ids_gather = accelerator.gather_for_metrics(batch['ids'])
            data_embed.extend(eos_token.cpu().numpy())
            data_ids.append(ids_gather)
        
    # data_embed = np.concatenate(data_embed, axis=0)
    weights = []
    # weights = np.concatenate(weights_list, axis=1)
    # weights = np.mean(weights, axis=1).squeeze()

    return data_embed, data_ids, weights

def main(args):
    model = load_model(args)
    model.eval()
    model.config.img_token = args.img_token
    
    args.max_pixels = 300 * 28 * 28
    args.min_pixels = 4 * 28 * 28
    
    # if args.num_chunks != -1:
    #     cand_file = [json.loads(q) for q in open(args.cand_path, "r")]
    #     cand_pool = get_chunk(cand_file, args.num_chunks, args.chunk_idx)
    # else:
    #     cand_pool = get_data_pool(args.cand_path)
    print(args.cand_path)
    if 'mscoco' in args.cand_path:
        args.cand_path = args.cand_path.replace('_cand_pool.jsonl', '_test_cand_pool.jsonl')
    infer_name = "_".join(args.cand_path.split('/')[-1].split('_')[:3])
    cand_dataset = InferenceQuery(args.cand_path, processor, args)
    cand_dataloader = DataLoader(cand_dataset, batch_size=args.batch_size, num_workers=48, shuffle=False, collate_fn=custom_collate_fn)    
    data_embed, data_ids, weights = inference(cand_dataloader, model)
    data_ids = [hash_did(id) for sublist in data_ids for id in sublist]

    answer_file = os.path.join(args.save_path, 'cand')
    os.makedirs(answer_file, exist_ok=True)

    if 'mscoco' in args.cand_path:
        save_cand_path = args.cand_path.replace('_test_cand_pool', '_cand_pool')
        np.save(f"{answer_file}/{Path(save_cand_path).stem}_embed.npy", data_embed)
        np.save(f"{answer_file}/{Path(save_cand_path).stem}_ids.npy", data_ids)
    else:
        np.save(f"{answer_file}/{Path(args.cand_path).stem}_embed.npy", data_embed)
        np.save(f"{answer_file}/{Path(args.cand_path).stem}_ids.npy", data_ids)

    print('-'*50)

    query_dataset = InferenceQuery(args.query_path, processor, args, args.query_cand_path)
    query_dataloader = DataLoader(query_dataset, batch_size=args.batch_size, num_workers=48, shuffle=False, collate_fn=custom_collate_fn)
    data_embed, data_ids, weights = inference(query_dataloader, model)

    data_ids = [hash_qid(id) for sublist in data_ids for id in sublist]
    answer_file = os.path.join(args.save_path, 'query')
    os.makedirs(answer_file, exist_ok=True)
    np.save(f"{answer_file}/{Path(args.query_path).stem}_embed.npy", data_embed)
    np.save(f"{answer_file}/{Path(args.query_path).stem}_ids.npy", data_ids)
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--cand-path", type=str, default=None)
    parser.add_argument("--query-cand-path", type=str, default=None)
    parser.add_argument("--query-path", type=str, default=None)
    parser.add_argument("--inst-path", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=-1)
    parser.add_argument("--chunk-idx", type=int, default=-1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--infer-query", type=bool)
    parser.add_argument("--infer-cand", type=bool)
    parser.add_argument("--answer-file", type=str, default=False)
    parser.add_argument("--img-token", action="store_true", default=False)
    parser.add_argument("--drop", action="store_true", default=False)
    parser.add_argument("--router", action="store_true", default=False)
    parser.add_argument("--layer-num", type=int, default=6)
    args = parser.parse_args()

    replace_qwen_training_modality_adaptive()
    main(args)
