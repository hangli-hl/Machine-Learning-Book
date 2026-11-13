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

In this repository, the **repository root** is the directory that contains this README, `pyproject.toml` and the `Data/` package.

**1. Install the dependencies** (Python 3.11+).

```bash
pip install numpy scipy scikit-learn matplotlib torch torchvision
```

`torch` and `torchvision` are needed only by the scripts in `Deep_Learning/` that use PyTorch, and by the MNIST and 1D Gaussian mixture loaders. The scripts in `Supervised_Learning/` and `Unsupervised_Learning/`, and the from-scratch networks `ffn_binary.py` and `ffn_multiclass.py`, run without them. The exact requirements of each directory are listed in its own README.

**2. Install this repository in editable mode (once per environment).** Run the following in the repository root:

```bash
python -m pip install -e .
```

What this does:

- `-e` (editable) registers the repository in the current Python environment as a link to the source tree instead of copying files. Therefore `from Data.data_loaders import ...` works from any working directory.
- Edits to the source files take effect immediately, with no reinstall.
- It is needed only once per environment (virtualenv, conda env, ...); repeat it when you create a new one.
- Without it, the scripts fail with `ModuleNotFoundError: No module named 'Data'`.

**3. Run any script directly**, from the repository root or from inside its own directory, for example:

```bash
python Supervised_Learning/gradient_boosting_binary.py
```

Figures are saved to `./plot` (and `./gen` for generative models) relative to the directory from which the command is launched.

## Data

All algorithms get their data from the shared `Data/` package through `Data/data_loaders.py`, so every method is evaluated on identical, reproducible datasets (fixed seed, same train/test split, standardization with training statistics). Algorithm code imports only from `data_loaders`:

```python
from Data.data_loaders import load_binary_classification_data
```

See [Data/README.md](./Data/README.md) for the available datasets, loader functions and conventions.

---

# 代码

本目录收录了《机器学习方法》一书中算法的代码实现。本次发布为第一批代码，后续将持续补充。代码实现与书中内容紧密配合，便于读者深入学习相关算法并开展实验。

> **声明：** 本代码库由作者在 AI 辅助下完成。作者投入了大量精力进行算法实现设计、统一规范化以及易读性优化。

## 目录

- [监督学习](./Supervised_Learning/)
- [无监督学习](./Unsupervised_Learning/)
- [深度学习](./Deep_Learning/)
- [数据](./Data/)：所有算法共用的统一数据加载模块

## 算法

### 监督学习

- 线性回归
- 感知机
- K 近邻
- 朴素贝叶斯
- 决策树
- 逻辑斯谛回归
- 支持向量机
- 提升方法（AdaBoost 与 GBDT）

### 无监督学习

- 聚类方法（k 均值聚类）
- EM 算法
- 马尔可夫链蒙特卡罗法（MCMC：Metropolis-Hastings 算法与吉布斯抽样）

### 深度学习

- 前馈神经网络
- 卷积神经网络
- Transformer
- 生成对抗网络
- 变分自编码器
- 扩散模型
- 流匹配模型

## 快速开始

在本仓库中，**仓库根目录**指包含本 README、`pyproject.toml` 和 `Data/` 包的目录。

**1. 安装依赖**（Python 3.11+）。

```bash
pip install numpy scipy scikit-learn matplotlib torch torchvision
```

只有 `Deep_Learning/` 中基于 PyTorch 的脚本，以及 MNIST 和一维高斯混合分布的加载函数需要 `torch` 和 `torchvision`。`Supervised_Learning/` 和 `Unsupervised_Learning/` 中的脚本，以及从零实现的 `ffn_binary.py` 和 `ffn_multiclass.py`，不需要它们。各目录的具体依赖见该目录自己的 README。

**2. 以可编辑模式安装本仓库（每个环境只需一次）。** 在仓库根目录执行：

```bash
python -m pip install -e .
```

这一步的作用：

- `-e`（editable）把本仓库以链接的方式注册到当前 Python 环境，而不是复制文件，因此无论在哪个工作目录，`from Data.data_loaders import ...` 都能工作。
- 修改源码后立即生效，无需重新安装。
- 每个环境（virtualenv、conda 环境等）只需执行一次，新建环境后要重新执行。
- 不执行这一步，脚本会报 `ModuleNotFoundError: No module named 'Data'`。

**3. 直接运行任意脚本**，可以在仓库根目录，也可以进入脚本所在目录，例如：

```bash
python Supervised_Learning/gradient_boosting_binary.py
```

图像保存到相对于启动命令所在目录的 `./plot`（生成模型还有 `./gen`）。

## 数据

所有算法都通过共用的 `Data/` 包（`Data/data_loaders.py`）获取数据，因此各方法使用相同且可复现的数据集（固定随机种子、相同的训练/测试划分、用训练集统计量标准化）。算法代码只从 `data_loaders` 导入：

```python
from Data.data_loaders import load_binary_classification_data
```

可用的数据集、加载函数和约定见 [Data/README.md](./Data/README.md)。