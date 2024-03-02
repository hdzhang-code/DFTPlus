# HWU64 Datasets

This dataset is from paper 'Few-Shot Intent Detection via Contrastive Pre-Training and Fine-Tuning':

Zhang, Jianguo, et al. "Few-Shot Intent Detection via Contrastive Pre-Training and Fine-Tuning." arXiv preprint arXiv:2109.06349 (2021).
https://arxiv.org/abs/2109.06349

The original dataset can be found here:

https://github.com/jianguoz/Few-Shot-Intent-Detection

This repository contains original data and reformatted data.

HWU64 dataset is weird, I cannot find a well-formatted dataset at Internet. There are several sources and none of them is strictly consistent with the dataset statistics reported in the original paper. This version is uploaded with a published paper. Although the data is also not consistent with statistics reported even in the paper that uses the dataset, at least it is used by a published paper.

## Reformatted format
```
dataset.json
|____________ datasetname
 |___________ domain1
   |_________ data1
     |_________ utterance
     |_________ label set
       |_________ label1
       |_________ label2
   |_________ data2
     |_________ utterance
     |_________ label set
       |_________ label1
  |__________ domain2
...
```
```
showDataset.py: show dataset information.
convert.py: convert the original dataset into new format.
generateClsParition.py: generate class parition file.
other script tools.
```
```
clsPartition1.json: 1st class partition
clsPartition2.json: 2nd class partition
...
```

## Display the dataset
```shell
python convert.py
python showDataset.py
```
