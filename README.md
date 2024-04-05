# TNSDC-Generative-AI
---

# Age and Gender Detection from Facial Images

This project aims to predict age and gender from facial images using Convolutional Neural Networks (CNNs). It leverages deep learning techniques to accurately classify age and gender based on visual cues extracted from facial images.

## Dataset

The dataset used for training consists of facial images labeled with age and gender information. The images are preprocessed and resized to 128x128 grayscale images to facilitate model training.The dataset is https://www.kaggle.com/code/eward96/age-and-gender-prediction-on-utkface

## Model Architecture

The CNN model architecture comprises convolutional layers followed by max-pooling layers for feature extraction. The extracted features are then flattened and fed into fully connected layers for classification. The model is designed to predict gender using a binary classification approach and age using a regression approach.

## Training and Evaluation

The model is trained using the Adam optimizer with binary cross-entropy loss for gender prediction and mean absolute error (MAE) loss for age prediction. The training process is monitored using accuracy metrics for gender prediction and loss metrics for both gender and age prediction. 

## Results

The model achieves exceptional accuracy and reliability in predicting age and gender from facial images, as demonstrated through rigorous testing and validation. Visualizations are provided to illustrate the model's performance in terms of accuracy and loss over the training epochs.

## Usage

To use this code:

1. Ensure you have the required dependencies installed (`tensorflow`, `keras`, `numpy`, `matplotlib`, `seaborn`, `pillow`).
2. Update the `BASE_DIR` variable to point to the directory containing your dataset.
3. Run the provided code cells to load the data, preprocess images, define and train the model, and visualize the results.
