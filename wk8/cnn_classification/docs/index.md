

---
<center>
<img src="images/logo-poster.png" width=400px style="opacity:0.8">
</center>



# Tree classification with CNNs

This notebook aims to classify images of Willow and Oak trees using a Convolutional Neural Network (CNN). 
The dataset consists of labeled images, and the model is trained to distinguish between the two tree types based on visual features.

### Process

1. Build a CNN model to classify Willow and Oak tree images.
2. Apply data augmentation and preprocessing for better model generalization.
3. Train the model using PyTorch and evaluate performance on a validation set.
4. Visualize sample images and model predictions.

Model Overview

1. Convolutional Neural Network (CNN): Sequential model with convolutional layers, batch normalization, ReLU activation, and max pooling for feature extraction.
2. Fully Connected Layers: Convert extracted features into a classification output.
3. Binary Classification.

# Steps to execute this project

1. Open the cnn_classification project in your IDE.
2. Enter the cnn_classification folder in your terminal, and run `uv sync`
3. Activate your environment using `source .venv/bin/activate`
4. Create and update the .env file.
5. Download the "trees" dataset provided, unzip and store it in your desired directory.
6. In the `config.yaml` file of your project, update the `data` folder path for the dataset accordingly.
7. Create an empty folder, say "trees" for your model to be saved, preferably within your results directory.
8. In the `config.yaml` file of your project, update the `results` folder path to the newly created folder accordingly.
9. Ctrl+S to save changes to the `config.yaml` file.
10. Download any random image from the internet of an oak tree and willow tree. This will be used while running the code.
11. To run **Tree classification**: Run the jupyter notebook file at `docs/notebooks/CNN_trees_classification.ipynb`