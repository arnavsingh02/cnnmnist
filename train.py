import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
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
def train_model(dl, model, n_epochs=20):
    opt = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    
    for epoch in range(n_epochs):
        total_correct = 0
        total_samples = 0
        epoch_loss = 0
        
        for x, y in dl:
            opt.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y.argmax(dim=1))
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            total_correct += (y_pred.argmax(dim=1) == y.argmax(dim=1)).sum().item()
            total_samples += y.size(0)

        accuracy = total_correct / total_samples
        losses.append(epoch_loss / len(dl))
        accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}: Loss={losses[-1]:.4f}, Accuracy={accuracy*100:.2f}%')

    return model, losses, accuracies

# Load training data from Colab's sample data
train_ds = MNISTDataset('/content/sample_data/mnist_train_small.csv')
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

# Initialize and train the model
model = MyNeuralNet()  # Now this should work correctly
trained_model, losses, accuracies = train_model(train_dl, model)

# Save the model state dict (optional if you want to save it for later use)
torch.save(trained_model.state_dict(), 'mnist_model.pth')

# Visualizations: Loss vs Epoch Graph
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()

# Visualizations: Accuracy vs Epoch Graph
plt.figure(figsize=(10, 5))
plt.plot(accuracies)
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
