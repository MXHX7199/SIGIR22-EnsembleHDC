<div style="text-align:center">
<img src="sth\L3E-HD_logo.png" alt="L3E-HD_logo" width="700"/>
<h2>L3E-HD: A Framework Enabling Efficient Ensemble in High-Dimensional Space for Language Tasks</h2>
</div>
pytorch implementation of L3E-HD for Language Tasks.


## overview

This repository provides a implementation of the framework for language tasks.

- Code

  - `hdc.py`
    - implementation of HDC framework, using ngram for encoding and hamming distance for similarity calculation.
  - `adaboost.py`
    - implementation of adaboost framework, using HdC framework as the classifier.
  - `main.py`
    - provides different datasets and paraameters for user to choose.
- dataset

  - `language`
    - Language classification task.
  - `SST-2`
    - The Stanford Sentiment Treebank from [GLUE](https://gluebenchmark.com/tasks), for sentiment classification task.
  - `ag_news_csv`
    - News articles classification task.
  - `spam.csv`
    - Text spam classification task.
  - `Youtube-all`
    - Youtube comment spam classification task.

## Run model

### Example

```shell
python main.py --task-id 5 --classifiers 4 --boost-lr 1.0 --dim 2000 --ngram 4 --retrain-rounds 0 --hdc-lr 0.0005
```

### Parameters

- `--task-id`
  - from 1 to 5, indicating different tasks
- `--classifiers`
  - the number of classifiers for the boost framework
- `--boost-lr`
  - learning rate of boost framework
- `--dim`
  - dimension of HDC framework
- `--ngram`
  - value of n for ngram encoding method in HDC framework
- `--retrain-rounds`
  - iterations of retraining in HDC framework
- `--hdc-lr`
  - learning rate of HDC framework

## Citation

We now have a [paper](#), titled "L3E-HD: A Framework Enabling Efficient Ensemble in High-Dimensional Space for Language Tasks", which is published in SIGIR-2022.
```bibtex
@inproceedings{liu2021L3EHD,
 title={L3E-HD: A Framework Enabling Efficient Ensemble in High-Dimensional Space for Language Tasks},
 author={Liu, Fangxin and Li, Haomin and Jiang, Li},
 booktitle={Proceedings of the International ACM Sigir Conference on Research and Development in Information Retrieval (SIGIR)},
 year={2022}
}
```