# MNIST CNN with Activation Functions Analysis

This project explores the role of **activation functions** in deep learning and demonstrates how to build and train a **Convolutional Neural Network (CNN)** for the **MNIST handwritten digit classification** task. The project includes:
- Implementation and visualization of activation functions (Sigmoid, Tanh, ReLU, Leaky ReLU, Swish, Mish).
- Building and training a CNN model using TensorFlow/Keras.
- Testing different activation functions and analyzing their impact on model performance.
- Visualizing learned filters and feature maps to understand what the CNN is learning.

---

## Project Overview

The goal of this project is to:
1. Understand the importance of **activation functions** in neural networks.
2. Build a **CNN model** to classify handwritten digits from the **MNIST dataset**.
3. Analyze the impact of different activation functions on model performance.
4. Visualize the **filters** and **feature maps** learned by the CNN.

The project is divided into two phases:
1. **Phase 1**: Understanding and implementing activation functions.
2. **Phase 2**: Building and testing a CNN model on the MNIST dataset.

---

## Key Concepts

### 1. Activation Functions
Activation functions introduce **non-linearity** into neural networks, enabling them to learn complex patterns. Key activation functions explored in this project include:
- **Sigmoid**: Smooth S-shaped curve, prone to vanishing gradients.
- **Tanh**: Zero-centered version of Sigmoid, also prone to vanishing gradients.
- **ReLU**: Simple and fast, but can cause the "dying ReLU" problem.
- **Leaky ReLU**: Addresses the dying ReLU problem by allowing small gradients for negative inputs.
- **Swish**: Smooth and non-monotonic, often outperforms ReLU.
- **Mish**: Combines the benefits of ReLU and Swish.

### 2. Convolutional Neural Networks (CNNs)
CNNs are specialized neural networks for processing grid-like data (e.g., images). They consist of:
- **Conv2D Layers**: Extract spatial features from input images.
- **MaxPooling2D Layers**: Downsample feature maps to reduce computational complexity.
- **Flatten Layer**: Converts 2D feature maps into a 1D vector.
- **Dense Layers**: Perform classification based on extracted features.

### 3. Vanishing Gradient Problem
The **vanishing gradient problem** occurs when gradients become very small during backpropagation, slowing down or stopping learning. This is common with activation functions like Sigmoid and Tanh.

### 4. Dying ReLU Problem
The **dying ReLU problem** occurs when ReLU neurons output zero for all inputs, effectively becoming "dead" and not contributing to learning.

---

## Results

### 1. CNN Performance
- The CNN model achieved a **test accuracy of 98.97%** on the MNIST dataset.
- Training accuracy: **99.82%**
- Validation accuracy: **98.91%**

### 2. Activation Function Analysis
- **ReLU** performed well, but **Leaky ReLU** was tested as an alternative to address the dying ReLU problem.
- Visualizations of activation functions and their derivatives were created to understand their behavior.

### 3. Visualizations
- **Activation Functions**: Plots of Sigmoid, Tanh, ReLU, Leaky ReLU, Swish, and Mish, along with their derivatives.
- **Filters**: Visualized the filters learned by the first Conv2D layer.
- **Feature Maps**: Visualized the feature maps for a test image to understand how the CNN processes input data.

---

## Visualizations

### 1. Activation Functions and Derivatives
![Activation Functions](https://github.com/richapatel93/Deep-learning/blob/main/sigmoid%20and%20tanh%20active%20and%20drived%20function.png)
![Activation Functions](https://github.com/richapatel93/Deep-learning/blob/main/ReLU%20and%20RelU%20derivative%20.png)
### 2. Training and Validation Accuracy
![Accuracy Plot](https://github.com/richapatel93/Deep-learning/blob/main/Traing%20and%20testing%20accrancy%20with%20Leaky%20ReLu.png)
![Accuracy Plot]

### 3. Filters Learned by the First Conv2D Layer
![Filters](https://github.com/richapatel93/Deep-learning/blob/main/CNN%20first%20filter.png)

### 4. Feature Maps for a Test Image
![Feature Maps](https://github.com/richapatel93/Deep-learning/blob/main/Feature%20map%20for%20test%20result.png)

---

