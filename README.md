This repository provides a PyTorch implementation for classifying handwritten digits from the MNIST dataset using a multilayer feedforward neural network. It covers the complete workflow:

Data loading and preprocessing

Model architecture and training procedure

Validation and model saving

Inference, metrics, and visualizations

Model Architecture
The core model is defined in model-1.py, consisting of:

Four fully-connected (nn.Linear) layers with ReLU activations

Progressive layer sizes: 282 → 256 → 128 → 64 → 10 (output classes)

Dropout layers for regularization (rate=0.2) after each activation except the output

Data Preparation
The training and inference scripts expect CSV files: each row contains an image and its label. The custom MNISTDataset class loads and preprocesses the data, normalizing the pixel values to , and returns data as tensors. One-hot encoding is used for labels during training.

Training Pipeline
The model is trained using Adam optimizer and cross-entropy loss for up to 20 epochs (modifiable). Training includes:

Train/validation split (default 80/20)

Progress monitoring: plots for loss and accuracy over epochs

Model checkpointing (mnistmodel.pth) for reusability

Inference and Evaluation
The inference script restores the trained model and performs predictions on test data:

Calculates confusion matrix and macro F1 score (with manual implementation)

Visualizes the confusion matrix using Seaborn

Highlights incorrect predictions with sample visualizations

Usage
Training: Run train-1.py after placing the CSV dataset in the expected location. Outputs include saved model weights and training/validation plots.

Inference: Use inferences.py to generate evaluation metrics and visualizations for test data. Adjust CSV file paths as needed.

Model Definition: Check or modify architecture in model-1.py.
