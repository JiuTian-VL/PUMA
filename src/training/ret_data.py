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

from .params import DataArguments
from .constants import *

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

def get_image_info(image_path, min_pixel, max_pixel):
    # Using this because of process_vision_info function
    # Need to fix this in the future    
    
    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "image", 
                "image": image_path,
                "min_pixel": min_pixel,
                "max_pixel": max_pixel
            }
            ]
        }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]

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
        self.fps = data_args.fps
        self.cand_path = cand_path

        self._get_cand_pool(self.cand_path)

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
    
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                images.append(get_image_info(image_file, self.min_pixel, self.max_pixel))

        else:
            grid_key = None
            pixel_key = None
            images = None
            videos = None

        query_modality = sources['query_modality']
        selected_pos_cand_did = random.choice(sources['pos_cand_list'])

        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        assert pos_cand, f"Cannot find positive candidate {selected_pos_cand_did} for {sources}"
        # Note: pos_cand_dataset_id should be the same as query_dataset_id but for OVEN and INFOSEEK it is not.
        pos_cand_dataset_id = selected_pos_cand_did.split(":")[0]
        pos_cand_modality = pos_cand.get("modality", None)
        pos_cand_txt = pos_cand.get("txt") or ""
        
        pos_cand_res = random.choice(['It is [RET].', '[RET].', 'Try [RET].'])
        cand_folder = self.data_args.cand_folder
        cand_images = []
        # able to process mulit-image
        if 'image' in pos_cand_modality:
            pos_cand_txt = VISION_START_TOKEN+DEFAULT_IMAGE_TOKEN+VISION_END_TOKEN + pos_cand_txt
            pos_cand_img = pos_cand.get("img_path", None)
            cand_images.append(get_image_info(os.path.join(cand_folder, pos_cand_img), self.min_pixel, self.max_pixel))
        else:
            cand_images = None

        sources = sources['conversations']

        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_cand_pixel_values = []
        all_cand_image_grid_thw = []

        all_cand_input_ids = []
        all_cand_labels = []

        # Qwen2-VL uses a default system message so I've added this.
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{RETRIEVAL_TEXT}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))
        
        if len(CAND_TEXT) > 0:
            cand_message = f"{DEFAULT_IM_START_TOKEN}system\n{CAND_TEXT}{DEFAULT_IM_END_TOKEN}\n"
            cand_message_input_ids = processor.tokenizer(cand_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            cand_system_labels = torch.full_like(cand_message_input_ids, IGNORE_INDEX) 

            all_cand_input_ids.append(cand_message_input_ids.squeeze(0))
            all_cand_labels.append(cand_system_labels.squeeze(0))

        for idx, j in enumerate(range(0, len(sources), 2)):
            user_input_ = sources[j]
            gpt_response_ = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input_['role']}\n{user_input_['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response_['role']}\n"
            gpt_response = f"{gpt_response_['content']}\n{DEFAULT_IM_END_TOKEN}\n"
            
            # visual_prompt = f'{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}' if cand_images is not None else ''
            cand_input = f"{DEFAULT_IM_START_TOKEN}{user_input_['role']}\n{pos_cand_txt}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response_['role']}\n"
            cand_response = f"{pos_cand_res}\n{DEFAULT_IM_END_TOKEN}\n"

            # Query
            if idx == 0:
                inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                if pixel_key and grid_key:
                    all_pixel_values.append(inputs[pixel_key])
                    all_image_grid_thw.append(inputs[grid_key])
                else:
                    all_pixel_values.append(-1)
                    all_image_grid_thw.append(-1)
            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            # cand
            if idx == 0:
                cand_inputs = processor(text=[cand_input], images=cand_images, videos=videos, padding=False, return_tensors='pt')
                cand_prompt_input_ids = cand_inputs['input_ids']
                if cand_images is not None:
                    all_cand_pixel_values.append(cand_inputs[pixel_key])
                    all_cand_image_grid_thw.append(cand_inputs[grid_key])
                else:
                    # 特殊处理无效数据，或插入默认值
                    all_cand_pixel_values.append(torch.zeros(1, 1176))  # 替换为零张量
                    all_cand_image_grid_thw.append(torch.tensor([-1,-1,-1]))
            else:
                cand_prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
            cand_response_ids = processor.tokenizer(cand_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            cand_final_ids = torch.cat([cand_prompt_input_ids, cand_response_ids], dim=1).squeeze(0)
            cand_lables = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(cand_prompt_input_ids[0])),  
                    cand_response_ids.squeeze(0),
                ],
                dim=0,
            )
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)

            all_cand_labels.append(cand_lables)
            all_cand_input_ids.append(cand_final_ids)

        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        cand_labels = torch.cat(all_cand_labels, dim=0).to(torch.long)
        cand_input_ids = torch.cat(all_cand_input_ids, dim=0).to(torch.long)

        pixel_values = torch.cat(all_pixel_values, dim=0)
        image_thw = torch.cat(all_image_grid_thw, dim=0)
        
        
        cand_pixel_values = torch.cat(all_cand_pixel_values)
        cand_image_thw = torch.cat(all_cand_image_grid_thw)

        attention_mask = (input_ids > -1000000).to(torch.long)
        cand_attention_mask = (cand_input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cand_attention_mask=cand_attention_mask,
            labels=labels,
            cand_labels=cand_labels,
            cand_input_ids=cand_input_ids,
        )

        
        data_dict[pixel_key] = pixel_values
        data_dict[grid_key] = image_thw
        
        data_dict['cand_pixel_values'] = cand_pixel_values
        data_dict['cand_image_thw'] = cand_image_thw
        
        return data_dict

class DataCollatorForRetrievalDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_cand_label_ids = []
        batch_pixel_values = []
        batch_image_thw = []
        batch_cand_input_ids = []
        batch_cand_pixel_values = []
        batch_cand_image_thw = []

        for example in examples:
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            batch_cand_input_ids.append(example["cand_input_ids"])
            batch_cand_label_ids.append(example["cand_labels"])

            if 'pixel_values' in example:
                batch_pixel_values.append(example['pixel_values'])
                batch_image_thw.append(example['image_grid_thw'])
            
            if 'cand_pixel_values' in example:
                batch_cand_pixel_values.append(example['cand_pixel_values'])
                batch_cand_image_thw.append(example['cand_image_thw'])
            
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )
        cand_input_ids = pad_sequence(
            batch_cand_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        cand_attention_mask = cand_input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)
        cand_labels = pad_sequence(batch_cand_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'cand_labels': cand_labels,
            'cand_attention_mask': cand_attention_mask,
            'cand_input_ids': cand_input_ids,
            'attention_mask': attention_mask,
        }
            
        if len(batch_pixel_values) > 0:
            # pixel_values = torch.cat(batch_pixel_values, dim=0)
            # image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict['pixel_values'] = batch_pixel_values
            data_dict['image_grid_thw'] = batch_image_thw

        if len(batch_cand_pixel_values) > 0:
            data_dict['cand_pixel_values'] = batch_cand_pixel_values
            data_dict['cand_image_thw'] = batch_cand_image_thw

        return data_dict


def make_retrieval_data_module(processor, data_args):
    sft_dataset = RetrievalDataset(
        data_path=data_args.data_path, cand_path=data_args.cand_path, processor=processor, data_args=data_args
    )
    data_collator = DataCollatorForRetrievalDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)