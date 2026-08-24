# Flower Recognition Model 🌸🤖

## Overview
This project implements a Convolutional Neural Network (CNN) image classification model using TensorFlow and Keras. The model is trained to recognize and classify images of flowers into five distinct categories: **Daisy, Dandelion, Rose, Sunflower, and Tulip**.

## Technologies Used
*   **Frameworks:** TensorFlow, Keras
*   **Languages:** Python
*   **Libraries:** NumPy, Matplotlib, OS
*   **Environment:** Google Colab / Jupyter Notebook

## Model Architecture
The model is built using a sequential CNN architecture designed for robust image feature extraction:
1.  **Data Augmentation Layer:** Implements random horizontal flips, rotations, and zooming to prevent overfitting and improve model generalization.
2.  **Rescaling Layer:** Normalizes pixel values (1./255) for optimal neural network performance.
3.  **Convolutional & Pooling Layers:** Three sets of `Conv2D` (16, 32, and 64 filters) paired with `MaxPooling2D` to extract spatial hierarchies and patterns from images.
4.  **Dropout Layer:** Set at `0.2` to further reduce overfitting during training.
5.  **Dense Layers:** A fully connected layer with 128 neurons (ReLU activation) leading to the final output layer of 5 classes.

## Dataset & Training
*   **Dataset Size:** 4,317 images split into 80% training (3,454 images) and 20% validation (863 images).
*   **Image Size:** Standardized to 180x180 pixels.
*   **Compilation:** Optimized using `adam` and `SparseCategoricalCrossentropy` loss function.
*   **Performance:** Achieved approximately **78% training accuracy** and **72.6% validation accuracy** over 15 epochs.

## Usage & Prediction
The model includes a custom Python function `classify_images(image_path)` that loads a new image, processes it, and predicts the flower class along with a confidence score.

```python
# Example Usage:
outcome = classify_images('Sample/rose.jpg')
print(outcome)
# Output: 'The Image belongs to rose with a score of 93.28'
