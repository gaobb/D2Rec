# D2Rec

> Official PyTorch Implementation of [Dual-Masked and Discriminative Reconstruction for Unified Vision Anomaly Detection](https://ieeexplore.ieee.org/document/11503645/), IEEE TIP.

## Updates
- Support training and evaluation for large-scale Real-IAD-Variety
- Add the results of MVTec, VisA, BTAD, Medical and Real-IAD-Variety


## Introduction 
D2Rec is a simple, effective, general and robust unified (multi-class) vision anomaly detection framework that integrates unsupervised dual-masked reconstruction and a self-supervised discriminator, achieving competitive performance on both industrial and medical anomaly detection benchmarks.

## D2Rec Framework
![overview](images/MetaUAS_Framework.jpg)

## Main Results

Evaluation on MVTec, VisA, BTAD, Medical and Real-IAD-Variety datasets with 224x224 input resolution.
| datasets | I-AUROC | P-AUROC | I-AUPR | P-AUPR |
| :------: | :-----: | :-----: | :----: | :----: | 
|  MVTec   |   98.9    |  99.6  |   98.9    |  74.3  |
|   VisA   |   95.4    |  96.3  |   99.0    |  48.5  |
|   BTAD   |   96.2    |  96.6  |   97.6    |  61.3  |
|  Medical |   88.6    |  88.5  |   98.0    |  60.6  |
| Real-IAD-Variety |   88.1    |  97.7  |   93.8    |  46.5  |

Please see more detailed results in [results](results)




## 1. Environments

Create a new conda environment and install required packages.

```
conda create -n d2rec python=3.8.12
conda activate d2rec
pip install -r requirements.txt
```

## 2. Prepare Datasets
Download MVTec, VisA, BTAD, Medical and Real-IAD-Variety datasets from the official websites and unzip them to `./data/`.

You can freely use the provided 'meta.json' files in './data'. You can also use the scripts in `./gen_meta_json/` to generate `meta.json` for each dataset with the following command:
```
python3 ./gen_meta_json/mvtec.py
python3 ./gen_meta_json/visa.py 
python3 ./gen_meta_json/btad.py
python3 ./gen_meta_json/medical.py 
python3 ./gen_meta_json/real-iad-variety.py
```

## 3. Training
using unified (i.e., multi-class) vision anomaly setting
```
image_size=224
for dataset in mvtec visa btad medical Real-IAD-Variety
do 
CUDA_VISIBLE_DEVICES=0 python3 main.py  \
   --data_path   "./datasets/"$dataset \
   --dataset $dataset \
   --image_size ${image_size} \
   --batch_size 16 \
   --dual_mask \
   --mask_head
done
```

### 4. Evaluation
```
image_size=224
for dataset in mvtec visa btad medical  Real-IAD-Variety
do 
CUDA_VISIBLE_DEVICES=0 python3 main.py  \
   -e \
   --data_path   "./datasets/"$dataset \
   --dataset $dataset \
   --save_path  "./checkpoints/" \
   --image_size ${image_size} \
   --batch_size 16 \
   --dual_mask \
   --mask_head
done
```


## Citing
If you find this code useful in your research, please consider citing us:

```
@article{gao2026d2rec,
  title  = {Dual-Masked and Discriminative Reconstruction for Unified Vision Anomaly Detection},
  author = {Gao, Bin-Bin},
  booktitle = {IEEE Transactions on Image Processing},
  pages = {4701-4712},
  year = {2026}
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=gaobb/D2Rec&type=Timeline)](https://www.star-history.com/#gaobb/MetaUAS&Timeline)
