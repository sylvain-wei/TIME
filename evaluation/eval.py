# 评估脚本

import os
import json
import argparse
from model import VLLM_MODEL
from utils import *

# load dataset(JSON)
def load_dataset(args):
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # 筛选出符合task和dataset_name的prompt
    if args.task:
        dataset = [d for d in dataset if d["task"] == args.task]
    if args.dataset_name:
        dataset = [d for d in dataset if d["dataset_name"] == args.dataset_name]
    if 'news' in args.dataset_name.lower() and args.retriever:
        dataset = [d for d in dataset if d["Setting"] == args.retriever]
    
    assert 'news' in args.dataset_name.lower() and args.retriever is not None and args.retriever in ["bm25", "vector", 'hybrid'], f"For news dataset, retriever must be specified and must be one of ['bm25', 'vector', 'hybrid']"
        
    assert len(dataset) > 0, f"No dataset found for task: {args.task} and dataset_name: {args.dataset_name}"
    
    return dataset

# prompt
def get_all_prompts(dataset, args):
    # 将每个batch中的context和question提取出来
    prompts = []
    for d in dataset:
        prompt = get_prompt(d["Context"], d["Question"], d['Task'], d["Dataset Name"])
        prompts.append(prompt)
    return prompts

def get_prompt(context, question, task, dataset_name):
    prompt_template = get_prompt_template(task, dataset_name)
    prompt = prompt_template.format(context=context, question=question)
    return prompt

def get_prompt_template(task, dataset_name):
    base_dir = "../prompts/evaluation"
    if "wiki" in dataset_name.lower():
        base_dir = os.path.join(base_dir, "base")
        if task in ["Extract"]:
            fp = os.path.join(base_dir, "multi_choice_qa.txt")
        elif task in ["Localization"]:
            fp = os.path.join(base_dir, "free_form_qa_for_time_expression.txt")
        elif task in ["Computation", "Order_Reasoning", "Timeline", "Explicit_Reasoning"]:
            fp = os.path.join(base_dir, "free_form_qa.txt")
        elif task in ["Order_Compare", "Duration_Compare"]:
            fp = os.path.join(base_dir, "single_choice_qa.txt")
        elif task in ["Counterfactual"]:
            fp = os.path.join(base_dir, "free_form_qa_for_false_premise.txt")
        elif task in ["Relative_Reasoning", "Co_temporality"]:
            fp = os.path.join(base_dir, "free_form_qa_with_refusal.txt")
        else:
            raise ValueError(f"Unknown task: {task}")
    
    elif "news" in dataset_name.lower():
        base_dir = os.path.join(base_dir, "RAG")
        if task in ["Extract"]:
            fp = os.path.join(base_dir, "multi_choice_qa.txt")
        elif task in ["Localization"]:
            fp = os.path.join(base_dir, "free_form_qa_for_time_expression.txt")
        elif task in ["Computation", "Timeline"]:
            fp = os.path.join(base_dir, "free_form_qa.txt")
        elif task in ["Explicit_Reasoning", "Order_Reasoning", "Relative_Reasoning", "Order_Compare", "Duration_Compare", "Co_temporality"]:
            fp = os.path.join(base_dir, "single_choice_qa.txt")
        elif task in ["Counterfactual"]:
            fp = os.path.join(base_dir, "single_choice_qa_for_false_premise.txt")
        else:
            raise ValueError(f"Unknown task: {task}")
    
    elif "dial" in dataset_name.lower():
        base_dir = os.path.join(base_dir, "base")
        if task in ["Extract"]:
            fp = os.path.join(base_dir, "multi_choice_qa.txt")
        elif task in ["Localization"]:
            fp = os.path.join(base_dir, "free_form_qa_for_time_expression.txt")
        elif task in ["Computation", "Order_Reasoning", "Timeline", "Explicit_Reasoning"]:
            fp = os.path.join(base_dir, "free_form_qa.txt")
        elif task in ["Order_Compare", "Duration_Compare"]:
            fp = os.path.join(base_dir, "single_choice_qa.txt")
        elif task in ["Counterfactual"]:
            fp = os.path.join(base_dir, "free_form_qa_for_false_premise.txt")
        elif task in ["Relative_Reasoning", "Co_temporality"]:
            fp = os.path.join(base_dir, "free_form_qa_with_refusal.txt")
        else:
            raise ValueError(f"Unknown task: {task}")
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    with open(fp, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    return prompt_template

# inference_batch
def inference_batch(model, batch_prompts):
    batch_outputs = model.generate(batch_prompts)
    return batch_outputs

# 计算评估指标
def compute_metric(dataset, args):
    compute_score(dataset, args)

# evaluation逻辑（也要保存中间结果）（在内部分batch）
def evaluation(model, dataset, args):
    prompts = get_all_prompts(dataset, args)
    
    # 分batch
    batch_size = args.batch_size
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    
    all_outputs = []
    for batch in batches:
        batch_outputs = inference_batch(model, batch)
        all_outputs.extend(batch_outputs)
    
    for d, o in zip(dataset, all_outputs):
        d["Response"] = o
    
    # 保存中间结果
    if "news" not in args.dataset_name.lower():
        output_path = os.path.join(args.output_dir, f"{args.model_path.split('/')[-1]}_{args.dataset_name}_{args.task}.json")
    else:
        output_path = os.path.join(args.output_dir, f"{args.model_path.split('/')[-1]}_{args.dataset_name}_{args.task}_{args.retriever}.json")
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
        
    # 计算metric，获得指标文件夹
    compute_metric(dataset, args)

# 命令行参数
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, default=None)  # models for vLLM
    parser.add_argument("--dataset_path", type=str, required=True, default=None)  # dataset path for evaluation
    parser.add_argument("--output_dir", type=str, required=True, default="../responses")  # output path for evaluation
    parser.add_argument("--result_dir", type=str, required=True, default="../metric_results")  # result path for evaluation
    
    parser.add_argument("--dataset_name", type=str, required=True, default=None)    # 必须选定，否则无法得到正确的dataset
    parser.add_argument("--task", type=str, required=True, default=None)    # 必须选定，否则无法得到正确的dataset
    parser.add_argument("--retriever", type=str, required=True, default=None)    # 对于news数据集，需要指定retriever
    
    parser.add_argument("--batch_size", type=int, required=True, default=32)
    return parser.parse_args()


# main函数
def main():
    args = get_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    model = VLLM_MODEL(args.model_path)
    dataset = load_dataset(args)
    evaluation(model, dataset, args)

if __name__ == "__main__":
    main()