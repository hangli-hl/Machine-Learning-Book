# Code

This directory contains code implementations of the algorithms in the book *Machine Learning Methods*. This is the first batch of code to be released, and more will be added gradually. The code implementations are closely integrated with the book's content, making it convenient for readers to further study the algorithms and conduct experiments.

> **Disclaimer:** This code repository was completed by the author with the assistance of AI. The author devoted considerable effort to algorithm implementation design, standardization, and readability optimization.

## Contents

- [Data](./Data/)
- [Supervised Learning](./Supervised_Learning/)
- [Unsupervised Learning](./Unsupervised_Learning/)
- [Deep Learning](./Deep_Learning/)

## Algorithms

### Supervised Learning

- Linear Regression
- Perceptron
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Trees
- Logistic Regression
- Support Vector Machines (SVM)
- Boosting Methods

### Unsupervised Learning

- K-Means Clustering
- EM Algorithm
- Markov Chain Monte Carlo Methods (MCMC)

### Deep Learning

- Feedforward Neural Networks (FFN)
- Convolutional Neural Networks (CNN)
- Vision Transformer (ViT)
- Generative Adversarial Networks (GAN)
- Variational Autoencoders (VAE)
- Diffusion Models
- Flow Matching Models

## Getting Started

In this repository, the **repository root** is the directory that contains this README and `pyproject.toml`.

**1. Install this repository in editable mode (once per environment)** (Python 3.11+). Run one of the following in the repository root:

```bash
pip install -e .            # traditional machine learning: Supervised_Learning/, Unsupervised_Learning/; include numpy, scikit-learn, etc
pip install -e ".[deep]"    # deep learning: Deep_Learning/; include torch, torchvision
```

**2. Run any script directly**, from the repository root or from inside its own directory, for example:

```bash
python Supervised_Learning/gradient_boosting_binary.py
```

Results are saved to `./plot` (and `./gen` for generative models) in the directory from which the command is launched.

## Data

All algorithms get their data from the shared `Data/` package through `Data/data_loaders.py`, so every method is evaluated on identical, reproducible datasets (fixed seed, same train/test split, standardization with training statistics).

```python
from Data.data_loaders import load_binary_classification_data
```

See [Data/README.md](./Data/README.md) for the available datasets, loader functions and conventions.

---

# 代码

本目录收录了《机器学习方法》一书中算法的代码实现。本次发布为第一批代码，后续将持续补充。代码实现与书中内容紧密配合，便于读者深入学习相关算法并开展实验。

> **声明：** 本代码库由作者在 AI 辅助下完成。作者投入了大量精力进行算法实现设计、统一规范化以及易读性优化。

## 目录

- [数据](./Data/)
- [监督学习](./Supervised_Learning/)
- [无监督学习](./Unsupervised_Learning/)
- [深度学习](./Deep_Learning/)

## 算法

### 监督学习

- 线性回归
- 感知机
- K近邻（KNN）
- 朴素贝叶斯
- 决策树
- 逻辑斯谛回归
- 支持向量机（SVM）
- 提升方法

### 无监督学习

- K均值聚类
- EM算法
- 马尔可夫链蒙特卡罗法（MCMC）

### 深度学习

- 前馈神经网络（FFN）
- 卷积神经网络（CNN）
- 视觉Transformer（ViT）
- 生成对抗网络（GAN）
- 变分自编码器（VAE）
- 扩散模型
- 流匹配模型

## 快速开始

在本仓库中，**仓库根目录**指包含本 README 和`pyproject.toml`的目录。

**1. 以可编辑模式安装本仓库（每个环境只需一次）**（Python 3.11+）。在仓库根目录执行其中一条：

```bash
pip install -e .            # 传统机器学习：Supervised_Learning/、Unsupervised_Learning/，包含numpy、scikit-learn等
pip install -e ".[deep]"    # 深度学习：Deep_Learning/；包含torch、torchvision
```

**2. 直接运行任意脚本**，可以在仓库根目录，也可以进入脚本所在目录，例如：

```bash
python Supervised_Learning/gradient_boosting_binary.py
```

结果保存到启动命令所在目录的 `./plot`（生成模型还有 `./gen`）。

## 数据

所有算法都通过共用的 `Data/` 包（`Data/data_loaders.py`）获取数据，因此各方法使用相同且可复现的数据集（固定随机种子、相同的训练/测试划分、用训练集统计量标准化）。

```python
from Data.data_loaders import load_binary_classification_data
```

可用的数据集、加载函数和约定见 [Data/README.md](./Data/README.md)。