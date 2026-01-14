import copy
import os
from dataclasses import dataclass, field
from typing import Dict
import torch
import random
import transformers
import ujson as json
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
import csv

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



class PretrainDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        cand_path: str,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        padding=True,
    ):
        super(PretrainDataset, self).__init__()
        if isinstance(data_path, str):
            with open(data_path, mode='r', newline='', encoding='utf-8') as file:
                list_data_dict = []
                csv_reader = csv.reader(file)
                next(csv_reader)
                for row in csv_reader:
                    list_data_dict.append(row)

        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.min_pixel = data_args.min_pixels
        self.max_pixel = data_args.max_pixels
        # self.fps = data_args.fps
        # self.cand_path = cand_path
        self.inst_path = data_args.inst_path
        self.img_token = data_args.img_token

        # self._get_cand_pool(self.cand_path)
        # self._load_query_instructions(self.inst_path)



    def get_image_info(self, text, image_paths, min_pixel, max_pixel, query_prompt=None):
        content_list = []
        # if isinstance(image_paths, str):
        #     image_paths = [image_paths]

        # if query_prompt is not None:
        #     text = query_prompt + '\n' + text

        # query_txt_with_prompt = self.processor.tokenizer(text, truncation=True, max_length=450, padding=False, return_tensors=None, add_special_tokens=False)
        # text = self.processor.tokenizer.decode(query_txt_with_prompt['input_ids'])
        text = format_string(text)
        text += '\nSummarize above sentence in one word: '
            
        # if self.img_token and image_paths != '':
        #     text = '[IMG] ' + text
        
        # print(text)
        

        # for path in [image_paths]:
        #     if len(path) > 0:
        #         content_list.append(
        #             {
        #                 "type": "image",
        #                 "image": path,
        #                 "min_pixels": min_pixel,
        #                 "max_pixels": max_pixel,
        #             },
        #         )
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
    
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
          
        query_input = self.get_image_info(sources[0], '', self.min_pixel, self.max_pixel)
        cand_input = self.get_image_info(sources[1], '', self.min_pixel, self.max_pixel)

        data_dict = dict(
            query_inputs=query_input,
            cand_inputs=cand_input
        )
     
        return data_dict

class DataCollatorForPretrainDataset(object):
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
        
        # batch_modality_target = [
        #     example['modality_target']
        #     for example in examples
        # ]
        
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
            batch_query_inputs, tokenize=False, add_generation_prompt=True
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
            batch_cand_inputs, tokenize=False, add_generation_prompt=True
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
            'modality_target': None
        }
            
        return data_dict


def make_pretrain_data_module(processor, data_args):
    sft_dataset = PretrainDataset(
        data_path=data_args.data_path, cand_path='', processor=processor, data_args=data_args
    )
    data_collator = DataCollatorForPretrainDataset(processor)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)
