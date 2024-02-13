import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import inspect
import sys
import matplotlib.pyplot as plt
# Method to find the root directory (assuming .git is in the root)
def find_root_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):  # To avoid infinite loop
        if ".git" in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None  # Or raise an error if the root is not found
root = find_root_dir()
sys.path.append(root)
from src.bo_cvae import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
input_size = 100  # Example input size for table data
labels_length = 10  # Example number of classes
H = 50
H2 = 12
latent_dim = 3
batch_size = 64
learning_rate = 1e-3
epochs = 500

# Synthetic data
x = torch.randn(batch_size, input_size)  # Random input data
x = torch.sigmoid(x)  # Normalize to range [0, 1]
labels = torch.randint(0, labels_length, (batch_size,))  # Random class labels
labels_one_hot = nn.functional.one_hot(labels, labels_length).float()


# Instantiate the model
model = CVAE(input_size, labels_length, H, H2, latent_dim)

# Loss function and optimizer
reconstruction_loss = nn.BCELoss(reduction='sum')
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_loss(recon_x, x.view(-1, input_size))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    recon_x, mu, logvar = model(x, labels_one_hot)
    loss = loss_function(recon_x, x, mu, logvar)
    
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    print(f'Epoch {epoch}, Loss: {loss.item()}')

# Plotting the loss
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.title('Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()