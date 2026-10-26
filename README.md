# Comparative Study of Generative Models on Fashion-MNIST

A comparative deep learning study of six generative modeling approaches using the Fashion-MNIST dataset: Vanilla Autoencoder, Sparse Autoencoder, Variational Autoencoder (VAE), Vanilla GAN, Deep Convolutional GAN (DCGAN), and Denoising Diffusion Probabilistic Model (DDPM).

The project evaluates reconstruction quality, image generation capability, computational cost, and the limitations of using a discriminative MLP classifier for image generation.

## Overview

Generative models learn patterns in a dataset to reconstruct existing samples or generate new synthetic data. Different architectures make different trade-offs between reconstruction fidelity, sample diversity, training stability, and inference efficiency.

This project compares the following models under a common Fashion-MNIST setting:

| Model | Main Approach | Primary Objective |
|---|---|---|
| Vanilla AE | Deterministic encoder-decoder | Image reconstruction |
| Sparse AE | Autoencoder with L1 latent regularization | Reconstruction with sparse latent representations |
| VAE | Probabilistic latent-variable model | Reconstruction and latent-space regularization |
| Vanilla GAN | Generator-discriminator framework | Synthetic image generation |
| DCGAN | Convolutional GAN | Image generation with spatial feature learning |
| DDPM | Iterative denoising diffusion | Image generation through reverse diffusion |

The project also investigates whether a standard MLP classifier can serve as an image generator and evaluates a frozen MLP latent decoder as an additional baseline.

## Dataset

**Fashion-MNIST** is a grayscale image dataset containing 10 clothing categories.

- Image resolution: `28 × 28`
- Channels: `1` (grayscale)
- Training images: 60,000
- Test images: 10,000
- Number of classes: 10

The reconstruction evaluation shown in the experiment uses 7,000 test images. FID evaluation uses 2,048 real samples and 2,048 generated samples.

## Models and Methods

### 1. Vanilla Autoencoder

Maps an input image to a latent representation and reconstructs it through a decoder. Training minimizes pixel-level reconstruction error.

### 2. Sparse Autoencoder

Adds an L1 penalty to latent activations to encourage sparse representations:

\[
L = L_{\mathrm{reconstruction}}+\lambda\operatorname{mean}(|z|)
\]

The recorded configuration uses `latent_dim = 64` and `λ = 0.001`.

### 3. Variational Autoencoder (VAE)

Learns a probabilistic latent representation using a mean vector and log-variance. The objective combines reconstruction loss with KL-divergence regularization toward a standard normal prior.

### 4. Vanilla GAN

Uses a generator to synthesize images from random noise and a discriminator to distinguish real images from generated samples.

### 5. Deep Convolutional GAN (DCGAN)

Uses convolutional and transposed-convolutional layers to learn spatial image features for generation.

### 6. Denoising Diffusion Probabilistic Model (DDPM)

Learns to reverse a gradual noise-adding process. Generation starts from noise and iteratively denoises the sample, resulting in a more computationally expensive sampling process.

## Experimental Setup

The recorded experiments use the following settings where applicable:

- Framework: PyTorch
- Dataset: Fashion-MNIST
- Latent dimension: 64 for the recorded AE/VAE configurations
- Batch size: 128 for the recorded training configuration
- Autoencoder, VAE, GAN, DCGAN, and DDPM training: 20 epochs
- MLP classifier: 30 epochs
- Frozen MLP decoder: 20 epochs
- Inference latency benchmark: batch size 64
- FID benchmark: 2,048 real and 2,048 generated samples

Experiments were run on CUDA. Actual training and inference times depend on the GPU, runtime environment, and implementation settings.

## Evaluation Metrics

| Metric | Purpose | Interpretation |
|---|---|---|
| MSE | Pixel-level reconstruction error | Lower is better |
| PSNR | Reconstruction fidelity in decibels | Higher is better |
| SSIM | Structural similarity between images | Higher is generally better |
| FID | Feature-distribution distance between real and generated images | Lower is generally better |
| Training time per epoch | Computational training cost | Lower is faster |
| Inference latency per batch of 64 | Sampling or inference cost | Lower is faster |

MSE, PSNR, and SSIM are used for reconstruction evaluation. FID is reported for the evaluated generative models. Reconstruction metrics are not directly interchangeable with FID because they measure different aspects of model performance.

## Selected Results

### Reconstruction Quality

The following results are from the final reconstruction-metrics output, evaluated on 7,000 test images.

| Model | Test MSE | Test SSIM |
|---|---:|---:|
| Vanilla AE | 0.007044 | 0.852541 |
| Sparse AE | 0.006994 | 0.851860 |
| VAE | 0.015807 | 0.734566 |

The Sparse AE achieved the lowest MSE among these three models, while the Vanilla AE achieved the highest SSIM. The VAE produced lower reconstruction scores in this experiment, reflecting a trade-off associated with its probabilistic latent representation and training objective.

### Image Generation Quality

Recorded FID results:

| Model | FID |
|---|---:|
| DCGAN | 36.8303 |
| VAE | 97.1361 |
| Vanilla GAN | 167.5108 |
| DDPM | 187.7433 |

**Important:** These FID values reflect the specific implementation and evaluation pipeline used in this experiment. They should not be interpreted as universal rankings of these model families. The DDPM result, in particular, should be discussed alongside its sampling configuration and visual outputs.

### Computational Cost

Recorded average training time per epoch and inference latency:

| Model | Mean Training Time / Epoch | Mean Inference Latency / Batch 64 |
|---|---:|---:|
| Vanilla AE | 7.79 s | 0.602 ms |
| Sparse AE | 7.85 s | 0.789 ms |
| VAE | 8.09 s | 0.394 ms |
| Vanilla GAN | 8.65 s | 0.534 ms |
| DCGAN | 12.75 s | 6.336 ms |
| DDPM | 14.47 s | 518.109 ms |

The DDPM implementation has substantially higher inference latency because it performs iterative denoising. The reported latency is specific to the implemented sampling procedure and benchmark environment.

## Visualizations

The notebook produces visualizations for qualitative and quantitative analysis, including:

- **Reconstruction comparison:** original images and autoencoder reconstructions.
- **Random generative samples:** images sampled from the generative models.
- **MLP generation limitations:** comparison of direct weight-transpose projection, frozen-decoder reconstruction, and generative sampling.
- **Latent/noise interpolation:** transitions between selected latent representations or noise vectors.
- **Training-loss curves:** training and validation losses across epochs.
- **DDPM denoising trajectory:** intermediate outputs during reverse diffusion.
- **MLP classifier evaluation:** loss, validation accuracy, and classification metrics.

Figures are saved in the `figures/` directory when the corresponding notebook cells complete successfully.

## MLP Classifier as a Generative Baseline

The project evaluates the limitations of using a discriminative MLP classifier for image generation.

A classifier is optimized to predict class labels \(P(y\mid x)\), rather than to model the image distribution \(P(x)\). Transposing classifier weights does not generally invert the learned transformation, and nonlinear activations can discard information needed for image reconstruction.

The frozen MLP decoder experiment provides an additional comparison with a learned reconstruction mapping. Its performance is evaluated using MSE, PSNR, and SSIM.

## Repository Structure

A suggested repository layout is:

```text
.
├── Generative_Models_Take_Home.ipynb
├── README.md
├── figures/
│   ├── figure1_reconstruction.png
│   ├── figure2_random_sampling.png
│   ├── figure3_mlp_generation_limitations.png
│   ├── figure4_latent_interpolation.png
│   ├── figure5_training_loss_curves.png
│   ├── ddpm_denoising_trajectory.png
│   └── mlp_classifier_training.png
├── metrics/
│   ├── reconstruction_metrics_final.csv
│   ├── inference_latency.csv
│   ├── fid_metrics.csv
│   └── final_comparison_table.csv
└── report/
    └── comparative_study_report.pdf
```

The layout above is a suggested organization; adjust filenames and directories to match the files actually committed to the repository.

## How to Run

1. Clone or download this repository.
2. Open `Generative_Models_Take_Home.ipynb` in Google Colab or Jupyter Notebook.
3. Select a GPU runtime if available.
4. Install the dependencies used by the notebook.
5. Configure the dataset, checkpoint, output, figure, and metrics paths.
6. Run the notebook cells in order.
7. Inspect the generated images, CSV metrics, and training logs.

Example setup for Google Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
```

The notebook may require path adjustments if it is run outside the original Google Drive directory.

## Dependencies

The experiment uses Python and PyTorch, with additional libraries for data processing, visualization, and evaluation.

Typical dependencies include:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-image scikit-learn torchmetrics torch-fidelity
```

Install compatible versions for your runtime, especially for PyTorch, CUDA, and metric-evaluation packages.

## Key Findings

1. **Autoencoders are effective reconstruction baselines.** The Vanilla AE and Sparse AE achieved the strongest reconstruction results among the three models with reported reconstruction metrics.
2. **Reconstruction quality and generative quality are different objectives.** MSE, PSNR, and SSIM evaluate reconstruction, while FID evaluates feature-distribution differences between real and generated samples.
3. **DCGAN achieved the lowest recorded FID** among the four models evaluated with FID in this experiment.
4. **DDPM inference was substantially slower** than the other evaluated models because generation requires repeated denoising steps.
5. **A discriminative classifier is not inherently a generative model.** Reusing or transposing its weights does not provide a principled model of the image distribution.

## Limitations

- Results depend on the architecture, hyperparameters, training duration, random seed, and hardware.
- The reported FID values use a limited sample count and an InceptionV3/ImageNet feature extractor; interpretation should account for this evaluation setup.
- SSIM results depend on correct output extraction, image range, and metric implementation. Reconstruction outputs should be checked to ensure that every model is evaluated consistently.
- The experiments do not establish that one model family is universally superior.
- Timing measurements are environment-dependent and should be treated as benchmark results for the recorded runtime.

## Conclusion

This study compares deterministic reconstruction, probabilistic latent modeling, adversarial generation, and diffusion-based generation on Fashion-MNIST. The results illustrate the trade-offs between reconstruction fidelity, generated-image quality, and computational cost.

The most suitable model depends on the application: autoencoders are useful for reconstruction, GANs can offer efficient sampling with appropriate training, and diffusion models provide an alternative generation approach with a higher iterative inference cost.

## Reproducibility

For reproducibility, retain the notebook, metric CSV files, figures, model configuration, and relevant environment details. Report the exact settings used for each benchmark and avoid comparing measurements collected under different runtime conditions without qualification.

---

**Author:** *Christina Angela Jodana*  
**Course:** Deep Learning  
**Project:** Comparative Study of Generative Models on Fashion-MNIST
