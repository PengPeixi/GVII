# GVII
This repository is based on and inspired by @Tuong Do's [work](https://github.com/aioz-ai/MICCAI21_MMQ). We sincerely thank for their sharing of the codes.

### Prerequisites

PYTHON 3.6

CUDA 9.2

Please install dependence package by run following command:
```
pip install -r requirements.txt
```

### Preprocessing

##### PathVQA dataset for VQA task

All data should be downloaded via [link](https://vision.aioz.io/f/e0554683595c4e1d9a08/?dl=1). The downloaded file should be extracted to `data_PathVQA/` directory.

### Training

Train MMQ + MEVF model with Bilinear Attention Network on PathVQA dataset.
```
$ sh run_vqa_PathVQA.sh
```

### Pretrained models and Testing

For our SOTA model on PathVQA dataset `saved_models/MMQ_BAN_MEVF_pathVQA`. The trained `MMQ_BAN_MEVF_pathVQA` model can be tested in PathVQA test set via: 
```
$ sh run_test_PathVQA.sh
```

The result json file can be found in the directory `results/`.
