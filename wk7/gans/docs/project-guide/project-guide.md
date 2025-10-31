# Project Guide

## Objective
### **1. Train DCGAN Models (Unconditional Generation)**

We’ll begin with the standard DCGAN setup, where the generator creates images **without being told what to generate**—just from random noise.

#### 🔹 Task 1: Generate Handwritten Digits  
You will train a DCGAN on the MNIST dataset to generate realistic images of handwritten digits (0–9). The model will learn the overall style and distribution of digits and produce new samples that look like they came from the original dataset.

#### 🔹 Task 2: Generate Tree Images  
Next, you'll move on to a custom dataset of tree images. The DCGAN will learn to generate new, realistic-looking trees based on the training data.

---

### **2. Train Conditional DCGAN Models (Conditional Generation)**

Now, we’ll take things further with **Conditional GANs**, where the generator is **guided with labels**—so it learns to produce images corresponding to a specific class.

#### 🔹 Task 1: Generate Specific Digits with Cues  
Instead of random digits, you’ll train a model that can generate a specific digit when you provide it the label (e.g., “generate a 5”).

#### 🔹 Task 2: Generate Specific Types of Trees  
Given a class label like "Oak" or "Weeping Willow", the model will generate an image of that type of tree. This is useful when you want **conditional and class-specific image synthesis**.

---

### **3. Explore Control GANs (Influencing the Output)**

Finally, you'll experiment with **Control GANs**, which are designed to provide **even more control** over the generation process. These models allow you to **nudge the output** of the generator toward a desired class—**even without explicit labels**.

You'll see how fine-tuning the input vectors can steer the generator to favor certain types of images, giving you creative power over what’s generated.

---


## Setup

1. Create .env file by copying contents from .env.example
2. Download the `trees dataset` from the [Lab Session Data Hub](https://supportvectors.io/course/view.php?id=32&section=3)
3. Update the `config.yaml` file

```yaml
mnist-classification:
  data: /Users/chandarl/data/mnist # path to where the MNIST dataset  will get downloaded
  results: /Users/chandarl/results/mnist # path to results

tree-classification:
  data: /Users/chandarl/data/trees # path to the folder containing the Oak and Weeping Willow folders
  results: /Users/chandarl/results/trees # path to results

```

## Execution
### DCGAN 
For the 1st task, set the `current task` in `config.yaml` as `mnist-classification`.

1. Run the `src/svlearn_gan/scripts/dcgan_main.py` to train the dcgan model on `mnist-classification`
2. Then run `docs/notebooks/mnist-gan-eval.ipynb`

For the 2nd task, set the `current task` in `config.yaml` as `tree-classification`.

1. Run the `src/svlearn_gan/scripts/dcgan_main.py` to train the dcgan model on `tree-classification`
2. Then run `docs/notebooks/trees-gan-eval.ipynb`


### Conditional DCGAN 
For the 1st task, set the `current task` in `config.yaml` as `mnist-classification`.

1. Run the `src/svlearn_gan/scripts/conditional_dcgan_main.py` to train the dcgan model on `mnist-classification`
2. Then run `docs/notebooks/mnist-gan-eval.ipynb`

For the 2nd task, set the `current task` in `config.yaml` as `tree-classification`.

1. Run the `src/svlearn_gan/scripts/conditional_dcgan_main.py` to train the dcgan model on `tree-classification`
2. Then run `docs/notebooks/trees-gan-eval.ipynb`

### Control GAN
For the 1st task, set the `current task` in `config.yaml` as `mnist-classification`.

1. Run the `src/svlearn_gan/scripts/classifier_main.py`
2. Then run the `docs/notebooks/mnist-control-gan-eval.ipynb`

For the 2nd task, set the `current task` in `config.yaml` as `tree-classification`.

1. Run the `src/svlearn_gan/scripts/classifier_main.py` to train the dcgan model on `tree-classification`
2. Then run `docs/notebooks/trees-control-gan-eval.ipynb`