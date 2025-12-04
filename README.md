# MyGO
This is the repository for paper "Modality-incomplete Fake News Video Detection via Prompt-assisted Modality Disentangling Model", accepted by ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM) in 2025. 

## 1. Setup
- Use 'requirements.txt' to install the necessary dependencies for python environments. We recommend using Python 3.8 or later.
- Download the required data to dir ~/MyGO/dataset

## 3. Dataset
The proposed FakeSV+ dataset is an extension of FakeSV dataset. Please sign the agreement in this [GitHub Repository](https://github.com/ICTMCG/FakeSV) to get full access to the FakeSV dataset first, then use the ~/MyGO/src/dataset_mask.py to generate the FakeSV+ dataset.

## 2. Usage
To run MyGO, please follow the steps below:
```shell
cd ~/MyGO/src
# Change directory to the source code folder
python dataset_mask.py --mask_ratio 0.3 --mask_type GCNet-v2 --dataset FakeSV --min_modality 1 --preserve_ratio 0 --masker_fp masker.json
# Masking the dataset
python main.py 
```

## 3. Note
- The code is implemented on a server with Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz, 256GB of RAM, an NVIDIA Tesla V100 PCIe 32GB and CentOS 7.  

## 4. Citation
If you find this repository useful in your research, please consider citing:
```script
To be determined
```
