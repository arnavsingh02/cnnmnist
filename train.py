#train.py
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
import torch.nn.functional as F  # Import functional module
from torch import nn  # Import nn module for neural network functionalities
from model import MyNeuralNet  # Import MyNeuralNet class

# Dataset Class
class MNISTDataset(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv(filepath).values
        self.x = torch.tensor(data[:, 1:].astype(np.float32) / 255.0)
        self.y = torch.tensor(data[:, 0].astype(int))
        self.y_one_hot = F.one_hot(self.y, num_classes=10).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, ix):
        return self.x[ix], self.y_one_hot[ix]

# Training Function
def train_model(train_dl, val_dl, model, n_epochs=20):  # Added val_dl
    opt = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(n_epochs):
        model.train()  # Set model to training mode
        total_correct_train = 0
        total_samples_train = 0
        epoch_loss_train = 0

        for x, y in train_dl:
            opt.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y.argmax(dim=1))
            loss.backward()
            opt.step()

            epoch_loss_train += loss.item()
            total_correct_train += (y_pred.argmax(dim=1) == y.argmax(dim=1)).sum().item()
            total_samples_train += y.size(0)

        train_accuracy = total_correct_train / total_samples_train
        train_losses.append(epoch_loss_train / len(train_dl))
        train_accuracies.append(train_accuracy)

        # Validation Loop
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for validation
            total_correct_val = 0
            total_samples_val = 0
            epoch_loss_val = 0

            for x, y in val_dl:
                y_pred = model(x)
                loss = loss_fn(y_pred, y.argmax(dim=1))

                epoch_loss_val += loss.item()
                total_correct_val += (y_pred.argmax(dim=1) == y.argmax(dim=1)).sum().item()
                total_samples_val += y.size(0)

        val_accuracy = total_correct_val / total_samples_val
        val_losses.append(epoch_loss_val / len(val_dl))
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Train Accuracy={train_accuracy*100:.2f}%, Val Loss={val_losses[-1]:.4f}, Val Accuracy={val_accuracy*100:.2f}%')

    return model, train_losses, val_losses, train_accuracies, val_accuracies

# Load training data from Colab's sample data
full_dataset = MNISTDataset('/content/sample_data/mnist_train_small.csv')

# Split the dataset into training and validation sets (e.g., 80% train, 20% validation)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)  # No need to shuffle validation data

# Initialize and train the model
model = MyNeuralNet()  # Now this should work correctly
trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(train_dl, val_dl, model)

# Save the model state dict (optional if you want to save it for later use)
torch.save(trained_model.state_dict(), 'mnist_model.pth')

# Visualizations: Loss vs Epoch Graph
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Visualizations: Accuracy vs Epoch Graph
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()
