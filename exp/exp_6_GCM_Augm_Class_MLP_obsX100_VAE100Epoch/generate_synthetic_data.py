import os
import json
import sys
import inspect
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from sklearn.metrics import classification_report, f1_score, multilabel_confusion_matrix
from scipy.io import arff

# Get the root directory of your project (the directory containing 'src' and 'plugins')
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# Set the device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Configure Polars
pl.Config.set_tbl_rows(-1)


# VAE configurations and helper functions ------------------------------------------
# Creating Variational Autoencoders

def normalize_data(data):
    min_val = torch.min(data)
    max_val = torch.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

class VAE(nn.Module):

    def __init__(self, input_dim=16063, hidden_dim=400, latent_dim=200, device=device):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, log_var

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat, mean, log_var

model = VAE(input_dim=16063).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

def trainVAE(model, optimizer, epochs, device, train_loader, x_dim=16063):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(train_loader):
            
            x = x.view(x.size(0), x_dim).to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
    return overall_loss

def generate_synthetic_data(model, num_samples, device):
    model.eval()
    with torch.no_grad():
        # Sample from a standard normal distribution with shape [num_samples, 2]
        z = torch.randn(num_samples, 2).to(device)  # Adjusted to match the decoder's input dimension
        # Generate synthetic data
        synthetic_data = model.decode(z)
    return synthetic_data.cpu()


# read from directory --------------------------------------------------------------
tra, trameta = arff.loadarff('/home/sebacastillo/ealab/data/GCM_Training.arff')
tst, tstmeta = arff.loadarff('/home/sebacastillo/ealab/data/GCM_Test.arff')
train = pl.from_numpy(tra).to_numpy()
test = pl.from_numpy(tst).to_numpy()

# Assuming 'data' is your numpy array
data = train

# Identify unique classes
unique_classes = np.unique(data[:, -1])

# Lists to store synthetic data and labels
synthetic_data_list = []
labels_list = []

EPOCH = 1000

# Iterate over each class with a progress bar
for cancer_type in tqdm(unique_classes, desc="Processing Cancer Types"):
    try:
        # Filter data for the current cancer type
        filtered_data = data[data[:, -1] == cancer_type]

        # Prepare the data for training (excluding the label column)
        training_data = np.delete(filtered_data, -1, axis=1).astype(np.float32)

        # Convert to a PyTorch tensor and normalize
        training_data = torch.from_numpy(training_data)
        train_norm = normalize_data(training_data)  # Assuming normalize_data is defined

        # Create train DataLoader
        batch_size = 4
        train_loader = DataLoader(dataset=train_norm, batch_size=batch_size, shuffle=True)

        # Initialize and train the VAE model
        model = VAE(input_dim=16063).to(device)
        trainVAE(model, optimizer, epochs=EPOCH, device=device, train_loader=train_loader)

        # Generate synthetic data
        num_samples = 100
        synthetic_data = generate_synthetic_data(model, num_samples, device)

        # Add the synthetic data and labels to the lists
        synthetic_data_list.append(synthetic_data)
        labels_list.extend([cancer_type] * num_samples)

    except Exception as e:
        print(f"Error processing {cancer_type}: {e}")

# Saving the synthetic data list
np.save(currentdir + '/synthetic_data_list.npy', np.array(synthetic_data_list, dtype=object))
# Saving the labels list
np.save(currentdir + '/labels_list.npy', np.array(labels_list, dtype=object))


