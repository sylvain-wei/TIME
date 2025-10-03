<div align="center">

# ⏳ TIME


<div align="center" style="margin: 20px 0;">


[![Paper](https://img.shields.io/badge/📄-Paper-red?style=for-the-badge)](https://arxiv.org/abs/2505.12891)
[![Code](https://img.shields.io/badge/💻-Code-black?style=for-the-badge&logo=github)](https://github.com/sylvain-wei/TIME)
[![TIME Dataset](https://img.shields.io/badge/🤗-TIME%20Dataset-yellow?style=for-the-badge)](https://huggingface.co/datasets/SylvainWei/TIME)
[![TIME-Lite](https://img.shields.io/badge/⚡-TIME--Lite-blue?style=for-the-badge)](https://huggingface.co/datasets/SylvainWei/TIME-Lite)


</div>




<h2>[NeurIPS'25 Spotlight] TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios</h2>

<div align="center" style="margin: 20px 0;">
  <img src="assets/Peking_University_logo.svg" alt="Peking University" height="60" style="margin: 0 40px;"/>
  <img src="assets/Noah_s_ark_lab_logo.png" alt="Huawei Noah's Ark Lab" height="45" style="margin: 0 40px;"/>
</div>

</div>

> 🎉🎉 **Congratulations!** This paper has been accepted as **<span style="color: #dc3545; font-weight: bold;">NeurIPS 2025 Spotlight 🌟🔥</span>** at D&B track.
> 
**🌟 If you found this work helpful, please consider giving us a ⭐ on GitHub!**

[![GitHub stars](https://img.shields.io/github/stars/sylvain-wei/TIME?style=social&label=Star)](https://github.com/sylvain-wei/TIME)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/datasets/SylvainWei/TIME)

</div>


## 📋 Project Information

<!-- <img src="assets/logo.png" alt="TIME Logo" width="200"/> -->

> **Authors**: Shaohang Wei, Wei Li, Feifan Song, Wen Luo, Tianyi Zhuang, Haochen Tan, Zhijiang Guo, Houfeng Wang  
**Affiliation**: Peking University, Huawei Noah's Ark Lab  
**Contact**: [shaohang@stu.pku.edu.cn](mailto:shaohang@stu.pku.edu.cn)

## 📖 Abstract

<div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #953b3f; margin: 20px 0;">

Temporal reasoning is pivotal for Large Language Models (LLMs) to comprehend the real world. However, existing works neglect the real-world challenges for temporal reasoning: 

- **Intensive temporal information**
- **Fast-changing event dynamics** 
- **Complex temporal dependencies in social interactions**

To bridge this gap, we propose a multi-level benchmark **TIME**, designed for temporal reasoning in real-world scenarios. 

**TIME** consists of `38,522` QA pairs, covering 3 levels with 11 fine-grained sub-tasks. This benchmark encompasses 3 sub-datasets reflecting different real-world challenges: **TIME-Wiki**, **TIME-News**, and **TIME-Dial**. 

We conduct extensive experiments on reasoning models and non-reasoning models, and conducted an in-depth analysis of temporal reasoning performance across diverse real-world scenarios and tasks, and summarized the impact of test-time scaling on temporal reasoning capabilities. Additionally, we release **TIME-Lite**, a human-annotated subset to foster future research and standardized evaluation in temporal reasoning.

</div>

<div align="center" style="margin: 30px 0;">
  <img src="assets/dataset_overview.png" alt="TIME Dataset Overview" width="100%" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
</div>




## 🚀 Get Started

### 📥 Step 1: Install Dependencies

```bash
# Install git-lfs
pip install git-lfs
```

### 📊 Step 2: Download Dataset

We provide two datasets. Choose according to your needs:

<div style="background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 15px 0;">

**⚠️ Option 1: Complete TIME Dataset** *(Large dataset - may be too large for quick evaluation)*

```bash
# Navigate to the working directory and download the benchmark dataset TIME
chmod +x scripts/download_data_time.sh

# Download the data
./scripts/download_data_time.sh
```

</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">

**✅ Option 2: TIME-Lite Dataset** *(Recommended - High-quality subset)*

```bash
# Navigate to the working directory and download the benchmark dataset TIME-Lite
chmod +x scripts/download_data_time_lite.sh

# Download the data
./scripts/download_data_time_lite.sh
```

</div>



### 🔧 Step 3: Install Evaluation Dependencies

```bash
pip install -r evaluation/requirements.txt
```

### ▶️ Step 4: Run Evaluation

**Option A: Evaluate TIME dataset**
```
./scripts/eval_time.sh
```

**Option B: Evaluate TIME-Lite dataset** *(Recommended)*
```
./scripts/eval_timelite.sh
```

## 🧠 Construction Pipeline

<div align="center" style="margin: 30px 0;">
  <img src="assets/dataset_pipeline.png" alt="TIME Construction Pipeline" width="100%" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
</div>


## 📊 Data Quantity

<div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #953b3f; margin: 20px 0;">

**📈 Dataset Statistics:**

- **TIME**: `38,522` QA pairs (Complete benchmark)
- **TIME-Lite**: `943` QA pairs (High-quality subset)

</div>

Here is a detailed breakdown of the dataset statistics:

| Dataset          | All Tasks | Ext. | Loc. | Comp. | D.C. | O.C. | E.R. | O.R. | R.R. | C.T. | T.L. | C.F. |
|------------------|-----------|------|------|-------|------|------|------|------|------|------|------|------|
| **TIME** | **38522** | 1480 | 3546 | 3376  | 3401 | 3549 | 3537 | 3538 | 3537 | 3513 | 5508 | 3537 |
| TIME-Wiki        | 13848     | 1261 | 1299 | 1126  | 1151 | 1299 | 1287 | 1288 | 1287 | 1263 | 1300 | 1287 |
| TIME-News        | 19958     | 0    | 1800 | 1800  | 1800 | 1800 | 1800 | 1800 | 1800 | 1800 | 3758 | 1800 |
| TIME-Dial        | 4716      | 219  | 447  | 450   | 450  | 450  | 450  | 450  | 450  | 450  | 450  | 450  |
| **TIME-Lite** | **943** | 60   | 90   | 78    | 86   | 90   | 90   | 90   | 90   | 90   | 89   | 90   |
| TIME-Lite-Wiki   | 322       | 30   | 30   | 24    | 28   | 30   | 30   | 30   | 30   | 30   | 30   | 30   |
| TIME-Lite-News   | 299       | 0    | 30   | 30    | 30   | 30   | 30   | 30   | 30   | 30   | 29   | 30   |
| TIME-Lite-Dial   | 322       | 30   | 30   | 24    | 28   | 30   | 30   | 30   | 30   | 30   | 30   | 30   |

*Task abbreviations: Ext. (Extract), Loc. (Localization), Comp. (Computation), D.C. (Duration Compare), O.C. (Order Compare); E.R. (Explicit Reasoning), O.R. (Order Reasoning), R.R. (Relative Reasoning); C.T. (Co-temporality), T.L. (Timeline), C.F. (Counterfactual).*

## 💪🏻 Evaluation Results

### 📊 TIME-Lite Results Radar Charts

Here are the detailed evaluation results for the TIME-Lite dataset on different sub-datasets:

<div style="display: flex; flex-wrap: wrap; justify-content: space-around; margin: 30px 0; gap: 20px;">

<div style="flex: 1; min-width: 300px; text-align: center; background-color: #f8f9fa; padding: 20px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">

#### 🗄️ TIME-Lite-Wiki
<img src="assets/radar_time_lite_wiki.png" alt="TIME-Lite-Wiki Results" style="max-width: 100%; height: auto; border-radius: 10px;"/>

</div>

<div style="flex: 1; min-width: 300px; text-align: center; background-color: #f8f9fa; padding: 20px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">

#### 📰 TIME-Lite-News
<img src="assets/radar_time_lite_news.png" alt="TIME-Lite-News Results" style="max-width: 100%; height: auto; border-radius: 10px;"/>

</div>

<div style="flex: 1; min-width: 300px; text-align: center; background-color: #f8f9fa; padding: 20px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">

#### 💬 TIME-Lite-Dial
<img src="assets/radar_time_lite_dial.png" alt="TIME-Lite-Dial Results" style="max-width: 100%; height: auto; border-radius: 10px;"/>

</div>

</div>



## 💬 Citation

<div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #953b3f; margin: 20px 0;">

If you find our work interesting and meaningful, welcome to star this repo, give an upvote to our HF repo [TIME](https://huggingface.co/datasets/SylvainWei/TIME) and cite our paper as follows.

</div>

<div style="background-color: #2d3748; color: #e2e8f0; padding: 20px; border-radius: 10px; font-family: 'Courier New', monospace; font-size: 14px; line-height: 1.6; margin: 20px 0;">

```bibtex
@article{wei2025time,
  title={TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios},
  author={Wei, Shaohang and Li, Wei and Song, Feifan and Luo, Wen and Zhuang, Tianyi and Tan, Haochen and Guo, Zhijiang and Wang, Houfeng},
  journal={arXiv preprint arXiv:2505.12891},
  year={2025}
}
```

</div>

---

<div align="center" style="margin: 30px 0; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">



