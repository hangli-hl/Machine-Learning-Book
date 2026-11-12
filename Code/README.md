# Code

This directory contains code implementations of the algorithms in the book *Machine Learning Methods*. This is the first batch of code to be released, and more will be added gradually. The code implementations are closely integrated with the book's content, making it convenient for readers to further study the algorithms and conduct experiments.

> **Disclaimer:** This code repository was completed by the author with the assistance of AI. The author devoted considerable effort to algorithm implementation design, standardization, and readability optimization.

## Contents

- [Supervised Learning](./Supervised_Learning/)
- [Unsupervised Learning](./Unsupervised_Learning/)
- [Deep Learning](./Deep_Learning/)
- [Data](./Data/): unified dataset loaders shared by all algorithms

## Algorithms

### Supervised Learning

- Linear Regression
- Perceptron
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Trees
- Logistic Regression
- Support Vector Machines (SVM)
- Boosting Methods (AdaBoost and GBDT)

### Unsupervised Learning

- Clustering Methods (k-Means)
- EM Algorithm
- Markov Chain Monte Carlo Methods (MCMC: Metropolis-Hastings Algorithm and Gibbs Sampling)

### Deep Learning

- Feedforward Neural Networks (FNN)
- Convolutional Neural Networks (CNN)
- Transformer
- Generative Adversarial Networks (GAN)
- Variational Autoencoders (VAE)
- Diffusion Models
- Flow Matching Models

## Getting Started

**1. Install the dependencies.**

```bash
pip install numpy scikit-learn torch torchvision
```

**2. Install this repository in editable mode (once per environment).** Run the following in the repository root, where `pyproject.toml` is located:

```bash
python -m pip install -e .
```

**3. Run any script directly**, for example:

```bash
python Supervised_Learning/gradient_boosting_binary.py
```

## Data

All algorithms obtain their data through a single entry point, `Data/data_loaders.py`:

```python
from Data.data_loaders import load_binary_classification_data

train_x, train_y, test_x, test_y = load_binary_classification_data("moons", signed_label=True)
```

| Loader | Available datasets |
|---|---|
| `load_binary_classification_data` | `gaussians`, `jigsaw`, `circle`, `moons`, `iris` |
| `load_multiclass_classification_data` | `classification`, `blobs`, `iris` |
| `load_regression_data` | `plane`, `diabetes` |
| `load_clustering_data` | `gaussians`, `iris` |
| `load_gaussian_mixture_1d_data` | 1D Gaussian mixture (generative models) |
| `load_mnist_classification_data` / `load_mnist_generation_data` | MNIST |

Conventions:

- Instances have shape `(num_instances, num_features)` and are standardized with **training** statistics.
- Binary labels are `{0, 1}`, or `{-1, 1}` with `signed_label=True`. Multiclass and cluster labels are `{0, ..., k-1}`.
- Splits are fixed by random seeds, so results are reproducible.

Repository-defined synthetic datasets are generated in `Data/data_creators.py`. Algorithm code should only use `data_loaders.py` and should not import `data_creators.py`.

---

# 代码

本目录收录了《机器学习方法》一书中算法的代码实现。本次发布为第一批代码，后续将持续补充。代码实现与书中内容紧密配合，便于读者深入学习相关算法并开展实验。

> **声明:** 本代码库由作者在AI辅助下完成。作者投入了大量精力进行算法实现设计、统一规范化以及易读性优化。

## 目录

- [监督学习](./Supervised_Learning/)
- [无监督学习](./Unsupervised_Learning/)
- [深度学习](./Deep_Learning/)
- [数据](./Data/)：所有算法共用的统一数据加载模块

## 算法

### 监督学习

- 线性回归
- 感知机
- K近邻
- 朴素贝叶斯
- 决策树
- 逻辑斯谛回归
- 支持向量机
- 提升方法(AdaBoost 与 GBDT)

### 无监督学习

- 聚类方法(k 均值聚类)
- EM 算法
- 马尔可夫链蒙特卡罗法(MCMC:Metropolis-Hastings 算法与吉布斯抽样)

### 深度学习

- 前馈神经网络
- 卷积神经网络
- Transformer
- 生成对抗网络
- 变分自编码器
- 扩散模型
- 流匹配模型

## 快速开始

**1. 安装依赖。**

```bash
pip install numpy scikit-learn torch torchvision
```

**2. 以可编辑模式安装本仓库(每个环境只需一次)。** 在仓库根目录(`pyproject.toml` 所在目录)执行:

```bash
python -m pip install -e .
```

**3. 直接运行任意脚本**,例如:

```bash
python Supervised_Learning/gradient_boosting_binary.py
```

## 数据

所有算法都通过唯一入口 `Data/data_loaders.py` 获取数据:

```python
from Data.data_loaders import load_binary_classification_data

train_x, train_y, test_x, test_y = load_binary_classification_data("moons", signed_label=True)
```

| 加载函数 | 可用数据集 |
|---|---|
| `load_binary_classification_data` | `gaussians`、`jigsaw`、`circle`、`moons`、`iris` |
| `load_multiclass_classification_data` | `classification`、`blobs`、`iris` |
| `load_regression_data` | `plane`、`diabetes` |
| `load_clustering_data` | `gaussians`、`iris` |
| `load_gaussian_mixture_1d_data` | 一维高斯混合分布(生成模型) |
| `load_mnist_classification_data` / `load_mnist_generation_data` | MNIST |

约定:

- 样本形状为 `(样本数, 特征数)`,并使用**训练集**的均值和标准差做标准化。
- 二分类标签为 `{0, 1}`,设置 `signed_label=True` 时为 `{-1, 1}`。多分类和聚类标签为 `{0, ..., k-1}`。
- 数据划分由固定随机种子决定,结果可复现。

本仓库自定义的合成数据集在 `Data/data_creators.py` 中生成。算法代码只应使用 `data_loaders.py`,不要直接导入 `data_creators.py`。