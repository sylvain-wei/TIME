#! /bin/bash

# https 从 https://huggingface.co/datasets/SylvainWei/TIME 下载benchmark数据集
git lfs install
git clone https://huggingface.co/datasets/SylvainWei/TIME

# 解压数据集
tar -xzvf TIME.tar.gz

# 删除压缩包
rm TIME.tar.gz