

---
<center>
<img src="https://supportvectors.ai/logo-poster-transparent.png" width=400px style="opacity:0.8">
</center>



# Introduction

# Autoencoder Project Summary

This project explores different implementations of autoencoders across multiple notebooks, demonstrating their versatility in image processing and representation learning.

## Main Components
(in order of notebooks)

1. **Vanilla Autoencoder (MNIST dataset)**
   - Implements a basic autoencoder on the MNIST handwritten digits dataset
   - Demonstrates reconstruction, latent space sampling, and image generation
   - Uses Gaussian Mixture Models to model the latent distribution
   - Visualizes interpolations between digit representations

2. **ResNet50-based Autoencoder (Tree dataset)**
   - Advanced autoencoder built on ResNet50 architecture
   - Works with a custom dataset of Oak and Weeping Willow trees
   - Includes detailed preprocessing pipeline with data augmentation
   - Showcases model evaluation and loss tracking
   - Visualizes reconstructed tree images

3. **General Image Reconstruction**
   - Applies a pre-trained tree autoencoder to random images
   - Demonstrates transfer capability of the trained model
   - Provides visualization of original vs. reconstructed images
   - Shows interpolation between different image pairs

4. **Denoising Autoencoder**
   - Uses a ResNet50-based architecture trained for noise removal
   - Processes images with artificially induced noise
   - Demonstrates the model's ability to reconstruct clean images from noisy input
   - Visualizes the original noisy images alongside denoised reconstructions

5. **Masked Autoencoder**
   - Implements a ResNet50-based masked autoencoder
   - Features a custom masking function that randomly removes image patches
   - Shows how the model reconstructs missing information in images
   - Visualizes the masked input alongside complete reconstructions

The project demonstrates multiple autoencoder variants (vanilla, denoising, and masked) across different architectures, showcasing their applications in image reconstruction, generation, denoising, and inpainting.

# Steps to execute this project:

1. Open the auto_encoders project in your IDE.
2. Enter the auto_encoders folder in your terminal, and run uv sync
3. Activate your environment using source .venv/bin/activate
4. Create and update the .env file.
5. Download the trees dataset provided on the course portal if you have not already.
6. Download any 5-6 random images and save them in another folder (we have used images of our team members for this class).
7. In the `config.yaml` file of your project, update the following:
    - `data` directory paths for MNIST and trees dataset. (If you have downloaded the MNIST dataset in the previous lab sessions already, you can use the same directory)
    - `results` path for the trees dataset.
    - `data` path for the folder with the random images.
8. Ctrl+S to save changes to the `config.yaml` file.

