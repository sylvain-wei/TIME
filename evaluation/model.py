import transformers
import vllm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import time
import torch
import os

def get_visible_gpu_count():
    """
    获取当前可见的GPU数量
    根据CUDA_VISIBLE_DEVICES环境变量动态计算
    """
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices is None or cuda_visible_devices.strip() == '':
        return 1
    return len([x for x in cuda_visible_devices.split(',') if x.strip()])

class VLLM_MODEL():
    def __init__(self, model_path):
        self.model_path = model_path
        
        # 根据模型大小自动调整gpu_memory_utilization和tensor_parallel_size
        memory_util = 0.9  # 默认值
            
        # 获取GPU数量
        tp_size = get_visible_gpu_count()
        print(f"初始化模型 {self.model_path} 使用设置: gpu_memory_utilization={memory_util}, tensor_parallel_size={tp_size}")
        
        
        self.llm = LLM(
            model=self.model_path,
            gpu_memory_utilization=memory_util,
            enable_prefix_caching=True,
            device="cuda",
            dtype="bfloat16",
            tensor_parallel_size=tp_size,
            enforce_eager=True,
            enable_chunked_prefill=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def generate(self, texts):
        prompts = []
        if "instruct" in self.model_path.lower():   # Instruct model
            for text in texts:
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                prompts.append(prompt)
        else:   # NOTE Base model
            prompts = texts

        sampling_params = SamplingParams(
            n=1, # 每个prompt生成1个output
            temperature=0.0, 
            max_tokens=10000, 
            stop=[self.tokenizer.eos_token], # 生成到eos_token就停止，也可以加入更多终止字符串
            repetition_penalty=1.2, # 重复惩罚
            min_tokens=1,   # 至少生成一个token
        )

        outputs = self.llm.generate(
                    prompts,
                    sampling_params=sampling_params,
                )

        return outputs