# Semi-Supervised Domain Generalization with Evolving Intermediate Domain

[paper](https://www.sciencedirect.com/science/article/pii/S0031320324000311)

## Requirements

- numpy==1.19.2
- Pillow==8.1.0
- PyYAML==5.3.1
- scipy==1.9.3
- scikit_learn==1.1.1
- six==1.15.0
- torch==1.7.1
- torchvision==0.8.2
- Ubuntu==18.04
- Python==3.8.5

## Installation

Local install

```
git clone https://github.com/MetaVisionLab/SSDG.git
cd SSDG
pip install -r requirements.txt
```

Or using docker image ```pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel```.

## Data Preparation

You can download the dataset [PACS](https://csip.fzu.edu.cn/files/datasets/SSDG/pacs.zip), [Digits-DG](https://csip.fzu.edu.cn/files/datasets/SSDG/digits_dg.zip), [Office-Home-DG](https://csip.fzu.edu.cn/files/datasets/SSDG/office_home_dg.zip), [web_pacs](https://csip.fzu.edu.cn/files/datasets/SSDG/web_pacs.zip) and [web_office](https://csip.fzu.edu.cn/files/datasets/SSDG/web_office.zip) to the folder SSDG/data and unzip it, which include three folders representing three datasets in our paper.

File structure:

```
SSDG/data
|--digits_dg/
  |–– images/
  |–– splits/
|--office_home_dg/
  |–– images/
  |–– splits/
|--pacs/
  |–– images/
  |–– splits/
|--web_office/
|--web_pacs/
```

## Training

```
bash script/pacs.sh       #train pacs
bash script/digits.sh     #train digits_dg
bash script/officeHome.sh   #train office_home_dg
...
```

# Validation

We evaluate our method on all the SSDG tasks for each dataset and report the average accuracy.For each task(for example A2C),we report its average accuracy on last 5 epoch.

```
cd tools/
python parse_test_res_single.py log/UDAG_A2C.log --test-log
```

# Citation

Please cite our paper:

```
@article{lin2024semi,
  title={Semi-supervised domain generalization with evolving intermediate domain},
  author={Lin, Luojun and Xie, Han and Sun, Zhishu and Chen, Weijie and Liu, Wenxi and Yu, Yuanlong and Zhang, Lei},
  journal={Pattern Recognition},
  volume={149},
  pages={110280},
  year={2024},
  publisher={Elsevier}
}
```
# Contact us

For any questions, please feel free to contact  [Han Xie](mailto:han_xie@foxmail.com) or Dr. [Luojun Lin](mailto:linluojun2009@126.com).

# Copyright

This code is free to the academic community for research purpose only. For commercial purpose usage, please contact Dr. [Luojun Lin](mailto:linluojun2009@126.com).
