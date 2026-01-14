import os 
from transformers import AutoProcessor
import sys 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)
from src.training.model_file.modeling_pre import *
import argparse
import torch.nn.functional as F 
from accelerate import Accelerator
import accelerate
from peft import PeftModel 
import shutil 


replace_qwen_training_modality_adaptive()

def eval(args):
    original_model_id = args.original_model_id
    model_id = args.model_id 
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        original_model_id, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, 
        device_map='cpu',
        layer=args.layer_num
    )

    lora_model = PeftModel.from_pretrained(model, model_id)
    merged_model = lora_model.merge_and_unload()

    # processor is not changed so we still load from the original model repo
    processor = AutoProcessor.from_pretrained(original_model_id)
    
    # print(merged_model.RET_embed)
    # merged_model.save_pretrained
    merged_model.save_pretrained(args.save_path)
    processor.save_pretrained(args.save_path)

    # copy the chat_template.json file
    source_chat_file = os.path.join(args.original_model_id, "chat_template.json")
    target_chat_file = os.path.join(args.save_path, "chat_template.json")
    shutil.copy(source_chat_file, target_chat_file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--layer_num', type=int)

    args = parser.parse_args()
    eval(args)