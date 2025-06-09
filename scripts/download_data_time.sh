#! /bin/bash

# download benchmark dataset via https from https://huggingface.co/datasets/SylvainWei/TIME
git lfs install
git clone https://huggingface.co/datasets/SylvainWei/TIME

# tar the dataset
tar -xzvf TIME.tar.gz

# delete the zip
rm TIME.tar.gz
