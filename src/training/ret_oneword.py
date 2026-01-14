import os
from typing import Dict
import torch
import random
import transformers
import ujson as json
from torch.utils.data import Dataset

from qwen_vl_utils import process_vision_info
from .params import DataArguments
from .constants import *
from utils import format_string

def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output



def get_video_info(video_path, max_pixels, fps):
    # Using this because of process_vision_info function
    # Need to fix this in the future

    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "video", 
                "video": video_path,
                "max_pixels": max_pixels,
                "fps": fps
            }
            ]
        }
    ]

    _, video_input = process_vision_info(messages)

    return video_input[0]

from torch.utils.data import Sampler, DataLoader
from collections import defaultdict
import numpy as np
class NormalBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=16, mean_datasets=4, std_datasets=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.mean_datasets = mean_datasets
        self.std_datasets = std_datasets
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        np.random.seed(self.epoch*42)
        dataset_samples = defaultdict(list)
        for idx, (dataset_name, _) in enumerate(self.dataset.samples):
            dataset_samples[dataset_name].append(idx)

        all_batches = []
        
        for dataset in dataset_samples:
            random.shuffle(dataset_samples[dataset])
        dataset_names = list(dataset_samples.keys())


        while sum(len(v) for v in dataset_samples.values()) >= self.batch_size:
            dataset_names = [dataset for dataset in dataset_names if len(dataset_samples[dataset]) > 0]

            if len(dataset_names) == 0:
                break
            # if len(dataset_names) == 1:
            #     print(1)
            
            num_datasets = int(np.clip(np.random.normal(self.mean_datasets, self.std_datasets), 1, len(dataset_names)))
            # num_datasets = 2
            chosen_datasets = np.random.choice(dataset_names, num_datasets, replace=False)
            
            batch = []
            for dataset in chosen_datasets:
                if len(dataset_samples[dataset]) == 0:
                    continue
                num_samples_per_dataset = max(1, self.batch_size // num_datasets)  # 确保合理分配
                num_samples_per_dataset = min(num_samples_per_dataset, len(dataset_samples[dataset]))  # 避免超出范围
                batch.extend(dataset_samples[dataset][:num_samples_per_dataset])
                dataset_samples[dataset] = dataset_samples[dataset][num_samples_per_dataset:]
            
            while len(batch) < self.batch_size:
                # remaining_samples = []
                # for dataset in dataset_samples:
                #     if len(dataset_samples[dataset]) > 0:
                #         remaining_samples.extend(dataset_samples[dataset])
                # np.random.shuffle(remaining_samples)
                # batch.extend(remaining_samples[:self.batch_size - len(batch)])
                remain_name = np.random.choice(dataset_names, 1, replace=False)[0]
                remain_samples = dataset_samples[remain_name]
                remain_num = min(self.batch_size - len(batch), len(remain_samples))
                batch.extend(remain_samples[:remain_num])
                dataset_samples[remain_name] = dataset_samples[remain_name][remain_num:]
            np.random.shuffle(batch)
            all_batches.append(batch[:self.batch_size])

        np.random.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        return len(self.dataset) // self.batch_size
    

from torch.utils.data import DistributedSampler


class DatasetOnlyDistributedSampler(DistributedSampler):
    def __init__(self, dataset, batch_size=16, mean_datasets=3, std_datasets=1, num_replicas=None, rank=None):
        super(DatasetOnlyDistributedSampler, self).__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.dataset = dataset
        self.batch_size = batch_size
        self.mean_datasets = mean_datasets
        self.std_datasets = std_datasets

    def __iter__(self):
        dataset_samples = defaultdict(list)
        for idx, (dataset_name, _) in enumerate(self.dataset.samples):
            dataset_samples[dataset_name].append(idx)

        all_batches = []
        dataset_names = list(dataset_samples.keys())

        while sum(len(v) for v in dataset_samples.values()) >= self.batch_size:
            dataset_names = [dataset for dataset in dataset_names if len(dataset_samples[dataset]) > 0]

            if len(dataset_names) == 0:
                break

            # Select number of datasets based on normal distribution
            num_datasets = int(np.clip(np.random.normal(self.mean_datasets, self.std_datasets), 1, len(dataset_names)))
            chosen_datasets = np.random.choice(dataset_names, num_datasets, replace=False)

            # Make sure the batch contains samples only from the chosen datasets
            batch = []
            for dataset in chosen_datasets:
                if len(dataset_samples[dataset]) == 0:
                    continue
                num_samples_per_dataset = max(1, self.batch_size // num_datasets)
                num_samples_per_dataset = min(num_samples_per_dataset, len(dataset_samples[dataset]))
                batch.extend(dataset_samples[dataset][:num_samples_per_dataset])
                dataset_samples[dataset] = dataset_samples[dataset][num_samples_per_dataset:]

            if len(batch) < self.batch_size:
                remaining_samples = []
                for dataset in dataset_samples:
                    if len(dataset_samples[dataset]) > 0:
                        remaining_samples.extend(dataset_samples[dataset])
                np.random.shuffle(remaining_samples)
                batch.extend(remaining_samples[:self.batch_size - len(batch)])

            # Shuffle the batch to ensure randomness
            np.random.shuffle(batch)
            all_batches.append(batch[:self.batch_size])

        # Split batches across different distributed processes (using the DistributedSampler state)
        start_index = self.rank * ((len(self.dataset)//self.batch_size) // self.num_replicas)
        end_index = (self.rank + 1) * ((len(self.dataset)//self.batch_size) // self.num_replicas)
        indices = all_batches[start_index:end_index]

        # all_batches = all_batches[self.rank::self.num_replicas]
        # self.local_batches = len(all_batches)
        return iter(indices)

    def __len__(self):
        return (len(self.dataset) // self.batch_size) // self.num_replicas * self.num_replicas * self.num_replicas


class RetrievalDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        cand_path: str,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        padding=True,
    ):
        super(RetrievalDataset, self).__init__()
        if isinstance(data_path, str):
            if 'jsonl' in data_path:
                list_data_dict = []
                with open(data_path, "r") as fin:
                    for line in fin:
                        list_data_dict.append(json.loads(line))
            else:    
                list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.min_pixel = data_args.min_pixels
        self.max_pixel = data_args.max_pixels
        # self.fps = data_args.fps
        self.cand_path = cand_path
        self.inst_path = data_args.inst_path
        self.img_token = data_args.img_token

        self._get_cand_pool(self.cand_path)
        self._load_query_instructions(self.inst_path)

        self.datasets = defaultdict(list)
        self.get_samples()
        self.samples = [(dname, i) for dname in self.datasets for i in range(len(self.datasets[dname]))]

    def get_samples(self):
        for item in self.list_data_dict:
            qid = item.get('qid', '')
            task_number = int(qid.split(':')[0])
            self.datasets[f'task_{task_number}'].append(item)

    def _load_query_instructions(self, instructions_path):
        """Validate and load instructions."""
        prompts_dict = {}
        with open(instructions_path, "r") as f:
            next(f)  # Skip the header line
            for line in f.readlines():
                parts = line.strip().split("\t")
                # Construct the key to be dataset_id, query_modality, cand_modality
                key = f"{parts[3]}, {parts[0]}, {parts[1]}"
                prompts = [p for p in parts[4:] if p]  # Filters out any empty prompts
                prompts_dict[key] = prompts
        self.query_instructions = prompts_dict

    def _get_random_query_prompt(self, dataset_id, query_modality, cand_modality):
        key = f"{dataset_id}, {query_modality}, {cand_modality}"
        prompts = self.query_instructions.get(key, [])
        assert prompts, f"Cannot find prompts for {key}"
        prompt = format_string(random.choice(prompts))
        assert prompt, f"Prompt is empty for {key}"
        return prompt


    def get_image_info(self, text, image_paths, min_pixel, max_pixel, query_prompt=None):
        content_list = []
        # if isinstance(image_paths, str):
        #     image_paths = [image_paths]

        if query_prompt is not None:
            text = query_prompt + '\n' + text

        text = format_string(text)
        query_txt_with_prompt = self.processor.tokenizer(text, truncation=True, max_length=450, padding=False, return_tensors=None, add_special_tokens=False)
        text = self.processor.tokenizer.decode(query_txt_with_prompt['input_ids'])
        
        if len(text) > 0 and len(image_paths) > 0:
            # i,t
            text += '\nSummarize above image and sentence in one word: '
        elif len(image_paths) == 0:
            # t
            text += '\nSummarize above sentence in one word: '
        else:
            # i
            text += '\nSummarize above image in one word: '
    
        if self.img_token and image_paths != '':
            text = '[IMG] ' + text
        
        # print(text)
        

        for path in [image_paths]:
            if len(path) > 0:
                content_list.append(
                    {
                        "type": "image",
                        "image": path,
                        "min_pixels": min_pixel,
                        "max_pixels": max_pixel,
                    },
                )
        content_list.append(
            {"type": "text", "text": text},
        )
        messages = [
            {
                "role": "user",
                "content": content_list
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "[RET]."}
                ]
            },
        ]
        # text = self.processor.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
        # image_input, _ = process_vision_info(messages)

        return messages

    def _get_cand_pool(self, cand_path):
        if 'jsonl' in cand_path:
            list_data_dict = []
            with open(cand_path, "r") as fin:
                for line in fin:
                    list_data_dict.append(json.loads(line))
        else:    
            list_data_dict = json.load(open(cand_path, "r"))
        self.cand_pool = list_data_dict
        cand_pool_dict = {}
        # able to use 'get' method
        for cand_pool_entry in self.cand_pool:
            did = cand_pool_entry.get("did")
            assert did, f"Cannot find did for {cand_pool_entry}"
            cand_pool_dict[did] = cand_pool_entry
        self.cand_pool = cand_pool_dict

    def convert_qwen_format(self, sources):
        if sources["query_img_path"] is not None:
            image_files = os.path.join(self.data_args.image_folder, sources["query_img_path"])
        else:
            image_files = ''

        qid = sources.get("qid", None)
        query_dataset_id = qid.split(":")[0] if qid else None
        query_modality = sources['query_modality']

        selected_pos_cand_did = random.choice(sources['pos_cand_list'])
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        assert pos_cand, f"Cannot find positive candidate {selected_pos_cand_did} for {sources}"
        
        pos_cand_modality = pos_cand.get("modality", None)

        query_prompt = self._get_random_query_prompt(query_dataset_id, query_modality, pos_cand_modality)

        pos_cand_txt = pos_cand.get("txt") or ""
        query_txt = sources.get('query_txt') or ''

        pos_cand_img = pos_cand.get("img_path", None)
        cand_image_files = os.path.join(self.data_args.cand_folder, pos_cand_img) if pos_cand_img is not None else ''
        
        query_messages = self.get_image_info(query_txt, image_files, self.min_pixel, self.max_pixel, query_prompt=query_prompt)
        cand_messages = self.get_image_info(pos_cand_txt, cand_image_files, self.min_pixel, self.max_pixel)
        

        return query_messages, cand_messages, pos_cand_modality
    
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # sources = self.list_data_dict[i]
        dataset_name, sample_idx = self.samples[i]
        sources = self.datasets[dataset_name][sample_idx]
          
        query_input, cand_input, modality_target = self.convert_qwen_format(sources)

        data_dict = dict(
            query_inputs=query_input,
            cand_inputs=cand_input,
            modality_target=modality_target,
            qid=sources['qid']
        )
     
        return data_dict

class DataCollatorForRetrievalDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        batch_query_inputs = []
        batch_cand_inputs = []
        # query_attention_mask = []
        # cand_attention_mask = []
        # for example in examples:
            # query_input_ids.append(example['query_inputs']['input_ids'])
            # query_attention_mask.append(example['query_inputs']['attention_mask'])
            # cand_inputs_ids.append(example['cand_inputs']['input_ids'])
            # cand_attention_mask.append(example['cand_inputs']['attention_mask'])
        
        batch_modality_target = [
            example['modality_target']
            for example in examples
        ]
        
        
        batch_qid = [
            example['qid']
            for example in examples
        ]
        parsed_indices = [tuple(map(int, item.split(':'))) for item in batch_qid]

        batch_qid = torch.tensor(parsed_indices)
        
        batch_query_inputs = [
            example['query_inputs']
            for example in examples
        ]
        batch_cand_inputs = [
            example['cand_inputs']
            for example in examples
        ]
        query_inputs = {
            'pixel_values': None,
            'image_grid_thw': None
        }

        cand_inputs = {
            'pixel_values': None,
            'image_grid_thw': None
        }

        query_text = self.processor.apply_chat_template(
            batch_query_inputs, tokenize=False, add_generation_prompt=False
        )
       
        query_image_inputs, video_inputs = process_vision_info(batch_query_inputs)
        query_inputs = self.processor(
            text=query_text,
            images=query_image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if 'pixel_values' not in query_inputs:
            query_inputs['pixel_values'] = None
            query_inputs['image_grid_thw'] = None


        cand_text = self.processor.apply_chat_template(
            batch_cand_inputs, tokenize=False, add_generation_prompt=False
        )
        cand_image_inputs, video_inputs = process_vision_info(batch_cand_inputs)
        cand_inputs = self.processor(
            text=cand_text,
            images=cand_image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if 'pixel_values' not in cand_inputs:
            cand_inputs['pixel_values'] = None
            cand_inputs['image_grid_thw'] = None
            

        data_dict = {
            'input_ids': query_inputs['input_ids'],
            'cand_input_ids': cand_inputs['input_ids'],
            'attention_mask': query_inputs['attention_mask'],
            'cand_attention_mask': cand_inputs['attention_mask'],
            'pixel_values': query_inputs['pixel_values'],
            'image_grid_thw': query_inputs['image_grid_thw'],
            'cand_pixel_values': cand_inputs['pixel_values'],
            'cand_image_thw': cand_inputs['image_grid_thw'],
            'modality_target': batch_modality_target,
            'batch_qid': batch_qid
        }
            
        return data_dict


def make_retrieval_data_module(processor, data_args):
    sft_dataset = RetrievalDataset(
        data_path=data_args.data_path, cand_path=data_args.cand_path, processor=processor, data_args=data_args
    )
    data_collator = DataCollatorForRetrievalDataset(processor)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)

class InferenceQuery(RetrievalDataset):
    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        cand_path: str = None,
        padding=True,
    ):
        super(RetrievalDataset, self).__init__()
        if isinstance(data_path, str):
            if 'jsonl' in data_path:
                list_data_dict = []
                with open(data_path, "r") as fin:
                    for line in fin:
                        list_data_dict.append(json.loads(line))
            else:    
                list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.min_pixel = data_args.min_pixels
        self.max_pixel = data_args.max_pixels
        # self.fps = data_args.fps
        self.cand_path = cand_path
        self.inst_path = data_args.inst_path
        if self.cand_path is not None:
            self._get_cand_pool(self.cand_path)
            self._load_query_instructions(self.inst_path)
        self.img_token = data_args.img_token
        

    def query(self, sources):
        if sources["query_img_path"] is not None:
            image_files = os.path.join(self.data_args.image_folder, sources["query_img_path"])
        else:
            image_files = ''

        qid = sources.get("qid", None)
        query_dataset_id = qid.split(":")[0] if qid else None
        query_modality = sources['query_modality']

        selected_pos_cand_did = random.choice(sources['pos_cand_list'])
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        assert pos_cand, f"Cannot find positive candidate {selected_pos_cand_did} for {sources}"
        
        pos_cand_modality = pos_cand.get("modality", None)

        query_prompt = self._get_random_query_prompt(query_dataset_id, query_modality, pos_cand_modality)
        query_txt = sources.get('query_txt') or ''
   
        query_messages = self.get_image_info(query_txt, image_files, self.min_pixel, self.max_pixel, query_prompt=query_prompt)        

        return query_messages

    def cand(self, sources):
        pos_cand_txt = sources.get("txt") or ""

        pos_cand_img = sources.get("img_path", None)
        cand_image_files = os.path.join(self.data_args.image_folder, pos_cand_img) if pos_cand_img is not None else ''
        
        cand_messages = self.get_image_info(pos_cand_txt, cand_image_files, self.min_pixel, self.max_pixel)
        
        return cand_messages

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        
        if self.cand_path is not None:
            # infer query
            messages = self.query(sources)
            ids = sources['qid']
        else:
            # infer cand
            messages = self.cand(sources)
            ids = sources['did']
     
        return messages, ids
