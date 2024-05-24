# Diffusion Generative Model

## Overview

The diffusion generative model is a probabilistic model used for image generation, leveraging diffusion processes. Given an initial image  $`x_0 `$, the model generates samples $` x_t `$ by iteratively applying a diffusion process over time $` t `$. A survey article in this topic can be found in [Diffusion Models for Generative Artificial Intelligence: An Introduction for Applied Mathematicians
](https://arxiv.org/abs/2312.14977).

## Generated Images
The folowing two images show how synthetic images can evolve at time $` t = 0 `$:
![initial noise](https://github.com/satwik-math/DDPM/blob/main/asset/generated_image.png)

starting with a complete random noise, produced at time $` t = 1000 `$:

![final noise](https://github.com/satwik-math/DDPM/blob/main/asset/noise.png)



## Mathematical Formulation

### Diffusion Process

At each time step $` t `$, the diffusion process adds noise to the image according to:

```math
x_{t+1} = x_t + \sqrt{\beta_t} \cdot \epsilon
```

where:
- $` x_t `$ is the image at time $` t `$,
- $` \beta_t `$ controls the magnitude of the noise at time $` t `$,
- $` \epsilon `$ is drawn from a standard Gaussian distribution.

### Reverse Diffusion

After a certain number of time steps, the process reverses, gradually removing noise from the image:

```math
x_{t-1} = x_t - \sqrt{\beta_t} \cdot \epsilon
```

### Training Objective

During training, the model learns to predict the parameters $` \beta_t `$ of the diffusion process to generate high-quality images resembling the training data. This is achieved by minimizing the following loss function:

```math
\mathcal{L} = \sum_{t=1}^{T} \mathbb{E}_{x_t \sim p_{\text{data}}(x_t)} \left[ -\log p_{\phi}(x_t | x_0) \right]
```

where $` p_{\phi}(x_t | x_0) `$ is the conditional probability of generating image $` x_t `$ given the initial image $` x_0 `$. Here, $` p_{data} `$ denotes the true image data distribution and model will learn $` p_{\phi} `$, where $` \phi `$ denotes model parameters. 

## Applications

- **Image Generation**: The diffusion generative model is effective in generating high-quality images with realistic textures and details.
- **Image Editing**: It can be used for tasks such as denoising, inpainting, and style transfer.
- **Anomaly Detection**: The model can detect anomalies or unusual patterns in images, useful for applications like medical imaging and surveillance.
