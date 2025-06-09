#! /bin/bash

# download benchmark via https from https://huggingface.co/datasets/SylvainWei/TIME-Lite
git lfs install
git clone https://huggingface.co/datasets/SylvainWei/TIME-Lite

# unzip the dataset
tar -xzvf TIME-Lite.tar.gz

# remove the zip file
rm TIME-Lite.tar.gz