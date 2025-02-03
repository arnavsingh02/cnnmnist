import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns
from model import MyNeuralNet  # Ensure this imports your model class
from train import MNISTDataset  # Ensure this imports your dataset class

# Load the dataset for inference from Colab's sample data
test_ds = MNISTDataset('/content/sample_data/mnist_test.csv')
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

# Load the trained model (if needed again)
model = MyNeuralNet()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# Run Inference on Test Data
all_preds = []
all_labels = []

with torch.no_grad():
    for xs, ys in test_dl:
        yhats = model(xs).argmax(dim=1)
        all_preds.append(yhats)
        all_labels.append(ys.argmax(dim=1))

all_preds_tensor = torch.cat(all_preds)
all_labels_tensor = torch.cat(all_labels)

# Function to compute confusion matrix manually
def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[t.item(), p.item()] += 1
    return cm

# Function to plot confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm.numpy(), annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

# Calculate confusion matrix and F1 score manually
confusion_mat = compute_confusion_matrix(all_labels_tensor, all_preds_tensor)
plot_confusion_matrix(confusion_mat)

# Manual calculation of F1 Score (Macro F1 Score)
def calculate_f1_score(y_true, y_pred):
    tp = (y_true * y_pred).sum().float()  
    fp = ((1 - y_true) * y_pred).sum().float()  
    fn = (y_true * (1 - y_pred)).sum().float()  
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return f1_score

# Calculate F1 Score for each class and average it (Macro F1 Score)
f1_scores = []
for i in range(10):
    binary_labels = (all_labels_tensor == i).float()
    binary_preds = (all_preds_tensor == i).float()
    f1_scores.append(calculate_f1_score(binary_labels, binary_preds))

macro_f1_score = sum(f1_scores) / len(f1_scores)
print(f'F1 Score (Macro): {macro_f1_score:.4f}')

# Incorrect Predictions Visualization
incorrect_indices = (all_preds_tensor != all_labels_tensor).nonzero(as_tuple=True)[0]
fig, ax = plt.subplots(5, 5, figsize=(10, 10))
for i in range(25):
    idx = incorrect_indices[i].item()
    ax[i//5, i%5].imshow(test_ds.x[idx].reshape(28, 28), cmap='gray')
    ax[i//5, i%5].set_title(f'True: {all_labels_tensor[idx].item()}, Pred: {all_preds_tensor[idx].item()}')
    ax[i//5, i%5].axis('off')
plt.tight_layout()
plt.show()
