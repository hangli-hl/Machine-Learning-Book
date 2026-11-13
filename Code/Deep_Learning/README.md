# Deep Learning Methods

This directory contains code implementations of the deep learning algorithms presented in *Machine Learning Methods*. It is still under construction, and more implementations will be added.

The two feedforward networks `ffn_binary.py` and `ffn_multiclass.py` are implemented from scratch with NumPy; all other models are built with PyTorch. Datasets are obtained through the shared `Data` package at the repository root (`Data/data_loaders.py`). Each model is a standalone script with a command-line interface, so one command runs a full train / visualize / evaluate cycle.

## Contents

### Feedforward Neural Networks (FNN)

- `ffn_binary.py` - Feedforward network with one sigmoid hidden layer for 2D binary classification, implemented from scratch in NumPy with explicit backpropagation.
- `ffn_multiclass.py` - Feedforward network with two ReLU hidden layers and a softmax output for multiclass classification, implemented from scratch in NumPy.
- `ffn.py` - PyTorch feedforward network with four ReLU hidden layers for MNIST digit classification, trained with Adam and cross-entropy loss.

### Convolutional Neural Networks (CNN)

- `cnn.py` - PyTorch convolutional network with two convolution-pooling blocks and two fully connected layers for MNIST digit classification.

### Transformer

- `vit.py` - Vision Transformer with convolutional patch embedding, a learnable CLS token, learnable positional embeddings and four pre-norm encoder blocks for MNIST classification.

### Generative Adversarial Networks (GAN)

- `gan_ffn.py` - GAN with a feedforward generator and discriminator for MNIST image generation, trained with alternating binary cross-entropy losses.
- `gan_cnn.py` - DCGAN-style GAN with a transposed-convolutional generator and a convolutional discriminator for MNIST image generation.

### Variational Autoencoders (VAE)

- `vae_ffn.py` - VAE with feedforward encoder and decoder, the reparameterization trick, a reconstruction-plus-KL objective and a t-SNE plot of the latent space.
- `vae_cnn.py` - VAE with a convolutional encoder and a transposed-convolutional decoder, otherwise following the same training and visualization pipeline.

### Diffusion Models

- `ddpm_1d.py` - Denoising diffusion probabilistic model on 1D Gaussian mixture data, visualizing the forward noise-adding and reverse denoising trajectories.
- `ddpm_unet.py` - Denoising diffusion probabilistic model with a UNet noise-prediction network for MNIST image generation, including a generation-progress grid.
- `smld_1d.py` - Score matching with Langevin dynamics on 1D Gaussian mixture data, comparing the learned score with the analytic score of the data.
- `smld_unet.py` - Score matching with Langevin dynamics using a UNet score network and annealed Langevin predictor-corrector sampling for MNIST image generation.

### Flow Matching Models

- `fm_1d.py` - Flow matching on 1D Gaussian mixture data with optimal-transport conditional paths, forward and inverse ODE integration, and a plot of the learned velocity field.
- `fm_unet.py` - Flow matching with a UNet velocity-prediction network and selectable ODE solvers (Euler, midpoint, Heun, RK4) for MNIST image generation.

### Shared Modules

- `unet.py` - U-shaped networks shared by the diffusion and flow matching scripts: a 2D UNet for MNIST and a 1D UNet for toy data, both conditioned by sinusoidal embeddings of a scalar time or noise level.
- `utility_functions.py` - Shared utilities: NumPy activation and loss functions, accuracy evaluation, collection of misclassified instances, VAE latent extraction, and device selection.
- `plot_functions.py` - Shared plotting: training curves, decision boundaries, misclassified images, VAE latent space, 1D trajectories, score and velocity fields, generated image grids, and generation progress.

The files under Shared Modules are libraries imported by the model scripts; all the others are entry points.

Datasets are not created in this directory: they come from `Data/data_loaders.py` (see [Datasets](#datasets)).

---

## Requirements

Python 3.11+ with `numpy`, `scikit-learn`, `matplotlib`, `torch` and `torchvision`.

```bash
pip install numpy scikit-learn matplotlib torch torchvision
```

`ffn_binary.py` and `ffn_multiclass.py` need only `numpy`, `scikit-learn` and `matplotlib`, because the `Data` package imports PyTorch only inside the MNIST and 1D Gaussian mixture loaders.

### Install the repository (once per environment)

All scripts import their data from the `Data` package at the repository root. From the repository root (where `pyproject.toml` is located), run:

```bash
python -m pip install -e .
```

Without it, the scripts fail with `ModuleNotFoundError: No module named 'Data'`. See [Getting Started](../README.md#getting-started) in the root README for details.

### Running the scripts

Run a script directly, either from the repository root or from inside this directory:

```bash
# from the repository root
python Deep_Learning/ddpm_unet.py --num-epochs 1

# or from inside this directory
cd Deep_Learning
python ddpm_unet.py --num-epochs 1
```

The sibling modules (`unet`, `utility_functions`, `plot_functions`) are imported by module name, which works because Python puts the script's own directory on the import path. Figures are saved to `./plot` and `./gen` relative to the directory from which the command is launched.

The PyTorch scripts select the device automatically through `utility_functions.set_device`, preferring CUDA, then Apple MPS, then CPU. MNIST is downloaded automatically on first use.

---

## Quick start

```bash
# From-scratch feedforward networks on small 2D / low-dimensional data
python ffn_binary.py
python ffn_multiclass.py

# Classifiers on MNIST
python ffn.py
python cnn.py
python vit.py

# Generative models on MNIST
python vae_ffn.py
python vae_cnn.py
python gan_ffn.py
python gan_cnn.py
python ddpm_unet.py
python smld_unet.py
python fm_unet.py

# Generative models on 1D Gaussian mixture data
python ddpm_1d.py
python smld_1d.py
python fm_1d.py
```

Each script loads a dataset, trains while printing progress, saves its figures, and either reports accuracy (classifiers) or writes generated samples (generative models).

Use `-h` to see all options with their defaults:

```bash
python fm_unet.py -h
```

The MNIST generative models are the expensive ones; pass a small `--num-epochs` for a quick smoke test:

```bash
python ddpm_unet.py --num-epochs 1
```

---

## Datasets

All data are loaded through `Data/data_loaders.py`; the dataset names, sizes and conventions are listed in [Data/README.md](../Data/README.md). Splits are fixed by `DATA_SEED` inside the `Data` package.

| Scripts | Loader | Data | Default |
| --- | --- | --- | --- |
| `ffn_binary.py` | `load_binary_classification_data` | 2D binary datasets, selected with `--data` | `jigsaw` |
| `ffn_multiclass.py` | `load_multiclass_classification_data` | Multiclass datasets, selected with `--data` | `blobs` |
| `ffn.py`, `cnn.py`, `vit.py` | `load_mnist_classification_data` | MNIST, pixels in `[0, 1]` | - |
| `vae_ffn.py`, `vae_cnn.py` | `load_mnist_generation_data` | MNIST, pixels in `[0, 1]` to match the sigmoid decoder | - |
| `gan_ffn.py`, `gan_cnn.py`, `ddpm_unet.py`, `smld_unet.py`, `fm_unet.py` | `load_mnist_generation_data` with `tanh_range=True` | MNIST, pixels in `[-1, 1]` to match the tanh output and the samplers | - |
| `ddpm_1d.py`, `smld_1d.py`, `fm_1d.py` | `load_gaussian_mixture_1d_data` | 1D Gaussian mixture (30% at `-5`, 70% at `+5`), with analytic density and score for reference curves | 2000 samples |

The 2D datasets are the only ones whose model can be drawn as a decision boundary, so `ffn_binary.py` and `ffn_multiclass.py` skip that plot for `classification` and `iris`.

---

## Options

Shared by the three MNIST classifiers (`ffn.py`, `cnn.py`, `vit.py`):

| Option | Default | Meaning |
| --- | --- | --- |
| `--num-epochs` | `10` | Number of training epochs |
| `--learning-rate` | `0.001` | Learning rate for Adam |
| `--batch-size` | `128` | Batch size for training and testing |

From-scratch feedforward networks:

| Option | Script | Default | Meaning |
| --- | --- | --- | --- |
| `--data` | both | `jigsaw` (`ffn_binary.py`), `blobs` (`ffn_multiclass.py`) | Dataset name |
| `--hidden-size` | `ffn_binary.py` | `4` | Units in the single hidden layer |
| `--hidden1-size`, `--hidden2-size` | `ffn_multiclass.py` | `6`, `6` | Units in the two hidden layers |
| `--num-epochs` | both | `10000` | Number of training epochs |
| `--learning-rate` | both | `0.1` (`ffn_binary.py`), `0.01` (`ffn_multiclass.py`) | Step size for gradient descent |
| `--batch-size` | both | `None` | Mini-batch size for SGD; full-batch GD when omitted |

MNIST generative models:

| Option | Scripts | Default | Meaning |
| --- | --- | --- | --- |
| `--latent-size` | VAEs, GANs | `16` (VAEs), `64` (GANs) | Size of the latent vector |
| `--num-epochs` | VAEs, GANs, UNet models | `50` (VAEs, GANs), `30` (UNet models) | Number of training epochs |
| `--learning-rate` | all | `0.001` (VAEs), `0.0002` (GANs), `0.0001` (UNet models) | Learning rate for Adam |
| `--batch-size` | all | `128` | Batch size for training |
| `--base-channels` | UNet models | `64` | Channels at the full resolution of the UNet |

Diffusion and flow matching specifics:

| Option | Scripts | UNet default | 1D default | Meaning |
| --- | --- | --- | --- | --- |
| `--num-timesteps` | DDPM | `1000` | `1000` | Number of diffusion timesteps |
| `--beta-start`, `--beta-end` | DDPM | `0.0001`, `0.02` | `0.0001`, `0.02` | Endpoints of the linear beta schedule |
| `--num-noise-levels` | SMLD | `200` | `10` | Levels in the geometric sigma schedule |
| `--sigma-min`, `--sigma-max` | SMLD | `0.005`, `15.0` | `0.01`, `10.0` | Endpoints of the sigma schedule |
| `--num-langevin-steps` | SMLD | `1` | `10` | Langevin corrector steps per noise level |
| `--target-snr` | SMLD | `0.16` | `0.10` | Target signal-to-noise ratio of the corrector step |
| `--num-steps` | FM | `50` | `100` | ODE integration steps for sampling |
| `--solver` | FM | `heun` | `rk4` | `euler` \| `midpoint` \| `heun` \| `rk4` |
| `--sigma-min` | FM | `0.0` | `0.0` | Minimum noise level of the OT conditional path |

1D toy scripts additionally take `--num-samples` (`2000`), `--num-iterations` (`2000` for DDPM, `5000` for SMLD and FM), `--batch-size` (`256`), `--hidden-size` (`128`) and `--num-gen-samples` (`3000`); they train on freshly drawn mini-batches rather than full passes over the dataset.

Examples:

```bash
python ffn_binary.py --data moons --hidden-size 8 --batch-size 32
python vit.py --num-epochs 3 --batch-size 256
python vae_cnn.py --latent-size 8 --num-epochs 20
python smld_unet.py --num-epochs 5 --num-langevin-steps 2
python fm_1d.py --solver euler --num-steps 20
```

---

## Output

Figures are written to two directories, both created automatically, with re-runs overwriting previous files:

- `./plot` - analysis plots: training curves, decision boundaries, misclassified images, VAE latent spaces, 1D trajectories, learned score and velocity fields
- `./gen` - generative results: per-epoch sample grids, generation-progress grids, 1D generated-sample histograms

File names are derived from the model name, for example `./plot/cnn_training_curve.png`, `./plot/vit_misclassified.png`, `./plot/vae_ffn_latent_space.png`, `./gen/gan_cnn_epoch_50.png`, `./gen/ddpm_unet_prog.png`, `./gen/fm_1d_gen.png`.

- **Training curve** - loss against training iterations, on a log y axis by default. The GAN scripts draw the discriminator and generator losses in one figure.
- **Misclassified images** - the first 20 misclassified test digits with their true and predicted labels.
- **Latent space** - VAE latent means of up to 5000 training images projected to 2D by t-SNE and coloured by digit.
- **Sample grid** - a 6x6 grid decoded from the same fixed latent vectors (VAE, GAN) or freshly sampled noise (DDPM, SMLD, FM) after every epoch, so the saved files show how training progresses. Only the last epoch is displayed on screen.
- **Generation progress** - one row per sample, columns being snapshots from noise to the final image, with the columns concentrated near the end of the trajectory where structure appears.
- **1D plots** - forward and reverse (or conditional-path, generative-ODE and inverse-ODE) trajectories, the learned score against the analytic score, the velocity field over `(x, t)`, and the generated distribution overlaid on the true density.

On a headless machine, set a non-interactive backend so that `plt.show()` does not block:

```bash
# Linux / macOS
MPLBACKEND=Agg python ddpm_unet.py --num-epochs 1

# Windows (PowerShell)
$env:MPLBACKEND = "Agg"; python ddpm_unet.py --num-epochs 1
```

---

## Evaluation

The classifiers print the accuracy on both the training and the test set and then display their first misclassified test instances. The from-scratch networks evaluate with `utility_functions.evaluate_accuracy`, the PyTorch ones with `evaluate_model_accuracy`.

The generative models have no single accuracy number, and are judged from the training curves, the saved sample grids, the generation-progress grids and, for the 1D models, the overlap between the generated histogram and the analytic density.

Every script fixes `MODEL_SEED = 64` after the data is loaded. The data itself is generated and split with `DATA_SEED = 32` inside the `Data` package, so repeated runs on the same device are reproducible.