# 本文件主要定义了评估指标的计算函数。输入应该是问题类型、预测答案、真实答案；输出是评估指标的值。
# 主要包含free-form和multi-choice两个问题类型
# 1. free-form问题：输入是问题类型、预测答案、真实答案；输出是评估指标的值。
# 2. multi-choice问题：输入是问题类型、预测答案、真实答案；输出是评估指标的值。

import os
import json
import re


def extract_options_from_pred_answer(text):
    # 去除可能存在的引号
    text = text.strip('"\'')

    # 从第一行或第一段提取答案
    first_line = text.split('\n')[0].strip()

    # 特殊处理：A,C 格式
    if ',' in first_line and all(c in "ABCDE," for c in first_line):
        return ' '.join(re.findall(r'[A-E]', first_line))
    
    # 匹配形如 [A B] 的格式
    multi_letter_match = re.search(r'\[(A B|B A|[A-E] [A-E])\]', text)
    if multi_letter_match:
        letters = re.findall(r'[A-E]', multi_letter_match.group(1))
        return ' '.join(letters)
    
    # 匹配形如 [[A B C D]] 的格式（双方括号中包含多个选项）
    double_brackets_multi_match = re.search(r'\[\[([A-E](?:\s+[A-E])*)\]\]', text)
    if double_brackets_multi_match:
        letters = re.findall(r'[A-E]', double_brackets_multi_match.group(1))
        return ' '.join(letters)
    
    # 匹配 "Assistant: X" 格式
    assistant_match = re.search(r'Assistant:\s*([A-E])', text)
    if assistant_match:
        return assistant_match.group(1)
    
    
    # 匹配 "The Answer is X" 格式
    answer_is_match = re.search(r'The Answer is\s*([A-E])', text)
    if answer_is_match:
        return answer_is_match.group(1)
    
    # 如果以上方法都没有找到答案，则寻找方括号中的答案 [A], [B] 等
    brackets_matches = re.findall(r'\[([A-E])\]', text)
    if brackets_matches:
        seen = set()
        result = [x for x in brackets_matches if not (x in seen or seen.add(x))]
        return ' '.join(result)
    
    # 如果没有找到任何匹配项，返回0
    return "0"


def calculate_free_form_metrics(pred_answer, gold_answer, allow_partial_match=False):
    """计算free-form生成问题的相关指标：参考TimeBench的标准，我们可以计算（1）token-level的exact match，（2）token-level的F1 score
    
    Args:
        pred_answer (str): 预测答案
        gold_answer (str): 真实答案
        allow_partial_match (bool, optional): 是否允许部分匹配。当为True时，即使预测答案只包含真实答案的一部分，也可能获得较高的分数。默认为False。
    
    Returns:
        dict: 包含评估指标的字典
            - exact_match: 完全匹配（0或1）
            - f1_score: F1分数（0到1之间的浮点数）
            - precision: 精确率
            - recall: 召回率
    
    例如：
    如果你要计算指标来衡量 pred_answer 的正确性，可以使用基于词语重叠的 F1 分数（F1 Score）作为主要指标。这种方法适用于评估文本答案的正确性，尤其是当答案可能是短文本且语义相近但措辞不同时。以下是具体的计算步骤：
计算方法
对于每一对问题中的标准答案（gold answer）和预测答案（pred answer），按照以下步骤计算 F1 分数，然后取所有问题的平均值作为总体指标：
分词与规范化  
将 gold answer 和 pred answer 分词，通常按空格分割，并将所有词转换为小写形式，去除标点符号（可选，取决于具体需求）。这可以减少因大小写或标点差异导致的不匹配。
例如：
Gold Answer: "India Brown took a photo of a feather and shells on a beach."
分词后：["india", "brown", "took", "a", "photo", "of", "a", "feather", "and", "shells", "on", "a", "beach"]
Pred Answer: "Between April 1st, 2020 and April 9th, 2020, India Brown participated in numerous significant artistic endeavours and immersive experiences..."
分词后：["between", "april", "1st", "2020", "and", "april", "9th", "2020", "india", "brown", "participated", "in", ...]
计算公共词数量  
找出 gold answer 和 pred answer 中的公共词（common tokens），并考虑每个词的最小频率。例如，如果 "and" 在 gold answer 中出现 1 次，在 pred answer 中出现 3 次，则公共词 "and" 的计数为 1。
计算公式：  
c = \sum_{t} \min(\text{freq}_{gold}(t), \text{freq}_{pred}(t))
，其中 ( t ) 是两组词中的所有唯一词。
计算精确率（Precision）和召回率（Recall）  
精确率：
\text{precision} = \frac{c}{\text{词数}_{pred}}
  
即公共词数量除以 pred answer 中的总词数。
召回率：
\text{recall} = \frac{c}{\text{词数}_{gold}}
  
即公共词数量除以 gold answer 中的总词数。
例如，对于第一个问题：
公共词："india", "brown", "and", "of"
c = 4
gold 词数 = 13，pred 词数 ≈ 50（假设完整答案有 50 个词）
\text{precision} = \frac{4}{50} = 0.08
\text{recall} = \frac{4}{13} \approx 0.307
计算 F1 分数  
F1 分数是精确率和召回率的调和平均数：
F1 = \frac{2 \times \text{precision} \times \text{recall}}{\text{precision} + \text{recall}}
（若 
\text{precision} + \text{recall} > 0
，否则 F1 = 0）
继续上面的例子：
F1 = \frac{2 \times 0.08 \times 0.307}{0.08 + 0.307} \approx \frac{0.04912}{0.387} \approx 0.127
处理特殊情况（可选）  
当 gold answer 为 "There is no answer" 时，检查 pred answer 是否也表示"无答案"（如 "None" 或短于 2 个词）。如果语义上匹配，可以直接赋值为 F1 = 1，否则按上述步骤计算。
例如：
Gold Answer: "There is no answer" → ["there", "is", "no", "answer"]
    """
    import re
    import string
    from collections import Counter
    if isinstance(gold_answer, list):
        gold_answer = ' '.join(gold_answer)
    # 处理特殊情况：当gold_answer为"There is no answer"时
    if gold_answer.lower().strip() == "there is no answer.".strip() or gold_answer.lower().strip() == "there is no answer":
        # 检查pred_answer是否也表示"无答案"
        pred_lower = pred_answer.lower().strip()
        no_answer_phrases = ["none", "no answer", "there is no answer", "no", "nothing", "not available", "unknown"]
        if any(phrase in pred_lower for phrase in no_answer_phrases):
            return {
                "exact_match": 1.0,
                "f1_score": 1.0,
                "precision": 1.0,
                "recall": 1.0
            }
        else:
            return {
                "exact_match": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }
    
    # 分词与规范化
    def normalize_and_tokenize(text):
        # 去除前后空格
        text = text.strip()
        # 转换为小写
        text = text.lower()
        # 去除标点符号
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        # 分词
        tokens = text.split()
        return tokens
    
    # 对gold_answer和pred_answer进行分词和规范化
    gold_tokens = normalize_and_tokenize(gold_answer)
    pred_tokens = normalize_and_tokenize(pred_answer)
    
    # 计算exact match
    exact_match = 1.0 if gold_answer.lower().strip() == pred_answer.lower().strip() else 0.0
    
    # 如果任一答案为空，则F1为0
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return {
            "exact_match": exact_match,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }
    
    # 计算词频
    gold_counter = Counter(gold_tokens)
    pred_counter = Counter(pred_tokens)
    
    # 计算公共词数量
    common_tokens = gold_counter & pred_counter
    common_count = sum(common_tokens.values())
    
    # 计算精确率和召回率
    precision = common_count / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = common_count / len(gold_tokens) if len(gold_tokens) > 0 else 0.0
    
    # 计算F1分数
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "exact_match": exact_match,
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall
    }

def calculate_multi_choice_metrics(pred_answer, gold_answer):
    """
    计算多选题的准确率，支持两种计算方式：完全匹配和子集匹配
    
    Args:
        pred_answer (str): 预测答案，可能包含选项字母（如"A B C"或"A,B,C"等）
        gold_answer (str): 真实答案，包含选项字母（如"A B C"）
    
    Returns:
        dict: 包含多种准确率的字典
            - exact_match: 预测答案和真实答案完全匹配的准确率（0或1）
            - subset_match: 预测答案是真实答案的子集的准确率（0或1）
            - precision: 预测答案中正确选项数量 / 预测答案中选项总数
            - recall: 预测答案中正确选项数量 / 真实答案中选项总数
            - TP: 真正例 - 预测为正确且实际正确的选项数量
            - FP: 假正例 - 预测为正确但实际错误的选项数量
            - FN: 假负例 - 预测为错误但实际正确的选项数量
            - pair_level_f1: 基于TP、FP、FN计算的F1分数
    
    Examples:
    1.1 预测的答案是正确的
    {
        "Question": "Which of the following are time expressions mentioned in the context? (Note: There may be one or more correct options.)\nA. May 15, 1994\nB. November 12, 2006\nC. 2002\nD. August 20, 2005\nE. None of the above.",
        "Gold Answer": "C",
        "Pred Answer": "C"
    }
    1.2 预测的答案顺序和真实答案的顺序不一样，但是预测的答案是正确的
    {
        "Question": "Which of the following are time expressions mentioned in the context? (Note: There may be one or more correct options.)\nA. 1990\nB. 1956\nC. June 12, 1974\nD. 1948\nE. None of the above.",
        "Gold Answer": "A B",
        "Pred Answer": "B A"    
    }
    1.3 预测的答案是正确的，但是几个字母之间没有空格，直接挨在一起了
    {
        "Question": "Which of the following are time expressions mentioned in the context? (Note: There may be one or more correct options.)\nA. 2007\nB. 2009\nC. March, 2010\nD. 2011\nE. None of the above.",
        "Gold Answer": "A B D",
        "Pred Answer": "ABD"
    }
    1.4 预测的答案是正确的，但是可能存在","隔开了几个字母，例如
    {
        "Question": "Which of the following are time expressions mentioned in the context? (Note: There may be one or more correct options.)\nA. May 10, 2017\nB. 2017\nC. 2013\nD. 2010s\nE. None of the above.",
        "Gold Answer": "B C D",
        "Pred Answer": "B,C,D"
    }
    1.5 预测的答案是正确的，不过"Pred Answer"中除了选项以外，后面还跟着一些解释性的文字，这些文字是预测的依据，应该被忽略，例如
    {
        "Question": "Which of the following are time expressions mentioned in the context? (Note: There may be one or more correct options.)\nA. May 10, 2017\nB. 2017\nC. 2013\nD. 2010s\nE. None of the above.",
        "Gold Answer": "B C D",
        "Pred Answer": " B,C,D\n\nContext: One thing many people don't realize about working out every day is how much strain this can place upon your body over long periods of time. Overtraining syndrome occurs due to either excessive training volume or intensity; if neither applies then perhaps this person simply needs to take some extra rest days between workouts! \n Question: What could cause someone to suffer from Osteoarthropathy?\nSteamy summer nights mean lots of heatwaves during summer months in India usually last around two"
    }
    """
    # 处理预测答案，提取选项
    def extract_options(answer):
        # 如果答案包含换行符，只取第一行（忽略解释性文字）
        if '\n' in answer:
            answer = answer.split('\n')[0]
        
        # 去除前后空格
        answer = answer.strip()
        
        # 将答案中的逗号和句点替换为空格
        answer = answer.replace(',', ' ').replace('.', ' ')
        
        # 处理没有空格的情况（如"ABD"）
        if ' ' not in answer and len(answer) > 1:
            answer = ' '.join(list(answer))
        
        # 分割答案并转换为大写字母集合
        options = set(opt.upper() for opt in answer.split() if opt)
        
        return options
    
    def pred_option_has_one_not_gold_option(pred_options, gold_options):
        for pred_option in pred_options:
            if pred_option not in gold_options:
                return True
        return False
    
    # 提取预测答案和真实答案中的选项
    pred_options = extract_options(pred_answer)
    gold_options = extract_options(gold_answer)
    
    # 计算完全匹配准确率
    exact_match = 1 if pred_options == gold_options else 0
    
    # 计算子集匹配准确率（预测答案是真实答案的子集）
    subset_match = 1 if pred_options.issubset(gold_options) and pred_options else 0
    
    # 计算混淆矩阵
    # TP: 真正例 - 预测为正确且实际正确的选项数量
    TP = len(pred_options.intersection(gold_options))
    
    # FP: 假正例 - 预测为正确但实际错误的选项数量
    FP = len(pred_options - gold_options)
    
    # FN: 假负例 - 预测为错误但实际正确的选项数量
    FN = len(gold_options - pred_options)
    
    # 计算预测准确率和召回率
    precision = TP / len(pred_options) if pred_options else 0
    recall = TP / len(gold_options) if gold_options else 0
    
    # 计算pair-level F1分数
    pair_level_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "exact_match": exact_match,
        "subset_match": subset_match,
        "precision": precision,
        "recall": recall,
        "TP": TP,  # 真正例
        "FP": FP,  # 假正例
        "FN": FN,  # 假负例
        "pair_level_f1": pair_level_f1  if not pred_option_has_one_not_gold_option(pred_options, gold_options) else 0# 基于TP、FP、FN计算的F1分数
    }
    

######################################特殊处理代码######################################################

def calc_wiki_computation(pred_answer, gold_answer):
    """
    计算wikidata的L1_3指标
    
    Args:
        pred_answer (str): 预测答案
        gold_answer (str or list): 真实答案，可能是字符串或字符串列表
    
    Returns:
        dict: 包含评估指标的字典，与calculate_free_form_metrics返回格式相同
    """
    # 处理gold_answer为列表的情况
    if isinstance(gold_answer, list):
        # 初始化最大分数
        max_metrics = {
            "exact_match": 0.0,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }
        
        # 对每个可能的答案计算指标，取F1分数最高的结果
        for answer in gold_answer:
            metrics = calculate_free_form_metrics(pred_answer, answer, allow_partial_match=True)
            if metrics["f1_score"] > max_metrics["f1_score"]:
                max_metrics = metrics
        
        return max_metrics
    
    # 处理gold_answer为字符串的情况
    elif isinstance(gold_answer, str):
        return calculate_free_form_metrics(pred_answer, gold_answer, allow_partial_match=True)
    
    # 处理其他类型的情况
    else:
        raise ValueError(f"gold_answer的类型为{type(gold_answer)}，不支持的类型")

def get_metric_compute_func(dataset_name, task):
    if "wiki" in dataset_name.lower():
        if task in ["Extract", "Order_Compare", "Duration_Compare"]:
            return calculate_multi_choice_metrics
        elif task in ["Localization", "Explicit_Reasoning", "Order_Reasoning", "Relative_Reasoning", "Co_temporality", 
                      "Timeline", "Counterfactual"]:
            return calculate_free_form_metrics
        elif task in ["Computation"]:
            return calc_wiki_computation
    elif "news" in dataset_name.lower():
        if task in ["Extract", "Order_Compare", "Duration_Compare", "Explicit_Reasoning", "Order_Reasoning", "Relative_Reasoning", "Co_temporality", "Counterfactual"]:
            return calculate_multi_choice_metrics
        elif task in ["Localization", "Computation", "Timeline"]:
            return calculate_free_form_metrics
    elif "dialog" in dataset_name.lower():
        if task in ["Extract", "Order_Compare", "Duration_Compare", "Explicit_Reasoning", "Order_Reasoning", "Relative_Reasoning", "Co_temporality", "Counterfactual"]:
            return calculate_multi_choice_metrics
        elif task in ["Localization", "Computation", "Timeline"]:
            return calculate_free_form_metrics
    else:
        raise ValueError(f"dataset_name: {dataset_name} 和 task: {task} 不支持")


def main():
    """
    主函数，用于计算所有模型在所有数据集上的评估指标
    """
    models = list(os.listdir("/home/weishaohang/workspace/Omni-Temp/results_time_lite"))
    for model in models:
        for setting in os.listdir(f"/home/weishaohang/workspace/Omni-Temp/results_time_lite/{model}"):
            # 处理不同的设置
            if setting == "RAG":
                # 特殊处理RAG设置，增加一层检索器路径
                for retriever in os.listdir(f"/home/weishaohang/workspace/Omni-Temp/results_time_lite/{model}/{setting}"):
                    for datasource in os.listdir(f"/home/weishaohang/workspace/Omni-Temp/results_time_lite/{model}/{setting}/{retriever}"):
                        dir_path = f"/home/weishaohang/workspace/Omni-Temp/results_time_lite/{model}/{setting}/{retriever}/{datasource}"
                        output_dir_path = f"/home/weishaohang/workspace/Omni-Temp/metrics_time_lite/{model}/{setting}/{retriever}/{datasource}"
                        # 加入判断，如果output_dir_path不存在，则递归创建
                        if not os.path.exists(output_dir_path):
                            os.makedirs(output_dir_path, exist_ok=True)
                        
                        process_files(dir_path, output_dir_path, datasource)
            else:
                # 原有的处理逻辑
                for datasource in os.listdir(f"/home/weishaohang/workspace/Omni-Temp/results_time_lite/{model}/{setting}"):
                    dir_path = f"/home/weishaohang/workspace/Omni-Temp/results_time_lite/{model}/{setting}/{datasource}"
                    output_dir_path = f"/home/weishaohang/workspace/Omni-Temp/metrics_time_lite/{model}/{setting}/{datasource}"
                    # 加入判断，如果output_dir_path不存在，则递归创建
                    if not os.path.exists(output_dir_path):
                        os.makedirs(output_dir_path, exist_ok=True)
                    
                    process_files(dir_path, output_dir_path, datasource)


def compute_score_single_case(d, metrics_func):
    """
    输入：dataset中的一个case
    输出：一个task-dataset_name pair的score
    """
    task = d["Task"]
    dataset_name = d["Dataset Name"]
    
    # 从response中提取pred_answer的逻辑 d["Response"]
    response = d["Response"]
    extracted_pred_answer = response
    
    # 先处理reasoning model的特殊标记
    if "<think>" in extracted_pred_answer:
        extracted_pred_answer = response.split("<think>")[1]
    if "</think>" in extracted_pred_answer:
        extracted_pred_answer = response.split("</think>")[1]
    if "<answer>" in extracted_pred_answer:
        extracted_pred_answer = response.split("<answer>")[1]
    if "</answer>" in extracted_pred_answer:
        extracted_pred_answer = response.split("</answer>")[0]
        
    # 再根据问题类型提取pred_answer
    if metrics_func == calculate_multi_choice_metrics:
        extracted_pred_answer = extract_options_from_pred_answer(extracted_pred_answer)
    else:
        extracted_pred_answer = extracted_pred_answer.split("answer:")[-1].split("Answer:")[-1].split("ANSWER:")[-1].split("**Answer:**")[-1].split("**Final Answer**")[-1].split("Final Answer")[-1].split("答案：")[-1].split("\n\n")[-1]
        if "boxed{" in extracted_pred_answer:
            extracted_pred_answer = extracted_pred_answer.split("boxed{")[1].split("}")[0]
    
    # 再根据问题类型提取pred_answer
    if metrics_func == calculate_multi_choice_metrics:
        extracted_pred_answer = extract_options_from_pred_answer(extracted_pred_answer)
    else:
        extracted_pred_answer = extracted_pred_answer.split("answer:")[-1].split("Answer:")[-1].split("ANSWER:")[-1].split("**Answer:**")[-1].split("**Final Answer**")[-1].split("Final Answer")[-1].split("答案：")[-1].split("\n\n")[-1]
        if "boxed{" in extracted_pred_answer:
            extracted_pred_answer = extracted_pred_answer.split("boxed{")[1].split("}")[0]
    
    # 计算metrics
    metrics_dict = metrics_func(extracted_pred_answer, d["Gold Answer"])
    
    return metrics_dict, extracted_pred_answer


def compute_score(dataset, args):
    """
    输入：dataset全集
    输出：每一个task-dataset_name pair的name
    """
    task = args.task
    dataset_name = args.dataset_name
    metrics_func = get_metric_compute_func(dataset_name, task)
    for idx, d in enumerate(dataset):
        metrics_dict, extracted_pred_answer = compute_score_single_case(d, metrics_func)
        d["Extracted Pred Answer"] = extracted_pred_answer
        d["Case Metrics"] = metrics_dict
    
    # 保存metrics中间结果
    if 'news' not in dataset_name.lower():
        output_path = os.path.join(args.result_dir, f"{args.model_path.split('/')[-1]}_{args.dataset_name}_{args.task}_case_metrics.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
    else:
        output_path = os.path.join(args.result_dir, f"{args.model_path.split('/')[-1]}_{args.dataset_name}_{args.task}_{args.retriever}_case_metrics.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    # 计算平均指标
    overall_metrics = {}
    
    if metrics_func == calculate_multi_choice_metrics:
        # 对于多选题，计算平均exact_match, micro-F1, macro-F1, 平均pair-level-F1
        
        # 平均exact_match
        overall_metrics["avg_exact_match"] = sum(m["exact_match"] for m in all_metrics) / len(all_metrics) if all_metrics else 0
        
        # 平均pair-level-F1
        overall_metrics["avg_pair_level_f1"] = sum(m["pair_level_f1"] for m in all_metrics) / len(all_metrics) if all_metrics else 0
        
        # 计算micro-F1（先汇总所有TP, FP, FN，再计算F1）
        total_tp = sum(m["TP"] for m in all_metrics)
        total_fp = sum(m["FP"] for m in all_metrics)
        total_fn = sum(m["FN"] for m in all_metrics)
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        overall_metrics["micro_precision"] = micro_precision
        overall_metrics["micro_recall"] = micro_recall
        overall_metrics["micro_f1"] = micro_f1
        
        # 计算macro-F1（先计算每个问题的F1，再取平均）
        overall_metrics["macro_f1"] = sum(m["pair_level_f1"] for m in all_metrics) / len(all_metrics) if all_metrics else 0
        
    elif metrics_func == calculate_free_form_metrics or metrics_func == calc_wiki_computation:
        # 对于自由形式问题，计算平均exact_match和平均f1_score
        
        # 平均exact_match
        overall_metrics["avg_exact_match"] = sum(m["exact_match"] for m in all_metrics) / len(all_metrics) if all_metrics else 0
        
        # 平均f1_score
        overall_metrics["avg_f1_score"] = sum(m["f1_score"] for m in all_metrics) / len(all_metrics) if all_metrics else 0
        
        # 平均precision
        overall_metrics["avg_precision"] = sum(m["precision"] for m in all_metrics) / len(all_metrics) if all_metrics else 0
        
        # 平均recall
        overall_metrics["avg_recall"] = sum(m["recall"] for m in all_metrics) / len(all_metrics) if all_metrics else 0
    
    if "news" not in dataset_name.lower():
        overall_metrics_fp = os.path.join(args.result_dir, f"{args.model_path.split('/')[-1]}_{args.dataset_name}_{args.task}_overall_metrics.json")
    else:
        overall_metrics_fp = os.path.join(args.result_dir, f"{args.model_path.split('/')[-1]}_{args.dataset_name}_{args.task}_{args.retriever}_overall_metrics.json")
    
    with open(overall_metrics_fp, "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=4, ensure_ascii=False)

    print(f"Compute metrics for {dataset_name} {task} successfully")
    print(f"Overall metrics saved to {overall_metrics_fp}")
    print("--------------------------------")
    print()

if __name__ == "__main__":
    main()