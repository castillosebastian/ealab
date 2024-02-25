import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class VAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) Class

    This class defines a Variational Autoencoder, a type of generative model that learns
    latent representations of input data. 

    Attributes:
        linear1, lin_bn1: First encoding layer and its batch normalization. Transforms the input data 
                        (with dimension D_in) to a hidden representation (with dimension H), followed 
                        by batch normalization for stability and performance improvement.
        linear2, lin_bn2: Second encoding layer and its batch normalization. Further transforms the data 
                        from dimension H to H2, followed by batch normalization.
        linear3, lin_bn3: Third encoding layer and its batch normalization. Processes the data at dimension 
                        H2, enabling the model to learn more complex abstractions at this level.
        fc1, bn1: Layer and batch normalization to transform the final encoder output to the latent space 
                representation (dimension: latent_dim). This represents the learned parameters of the 
                latent distribution.
        fc21, fc22: Two linear layers to output the mean (mu) and log-variance (logvar) of the latent 
                    distribution. These are used for the reparameterization trick.
        fc3, fc_bn3: Initial decoding layer and its batch normalization. Transforms latent representations 
                    back to dimension H2.
        fc4, fc_bn4: Further decoding layer and its batch normalization to start reconstructing the original 
                    input from the latent representation.
        linear4, lin_bn4: Additional decoding layer to process the reconstructed data at dimension H2.
        linear5, lin_bn5: Decoding layer to upscale the dimension from H2 back to H.
        linear6, lin_bn6: Final decoding layer to transform the dimension from H back to the original input 
                        dimension (D_in), aiming to reconstruct the original input data.

    Methods:
        encode(x): Processes the input data x through the encoding layers to produce the latent 
                distribution parameters (mean and log-variance).
        reparameterize(mu, logvar): Applies the reparameterization trick to sample from the latent 
                                    distribution using its parameters (mean and log-variance). Essential 
                                    for backpropagation through random sampling.
        decode(z): Transforms the sampled latent representations back to the data space, attempting to 
                reconstruct the input data from its latent representation.
        forward(x): Defines the forward pass of the VAE. Encodes the input, reparameterizes to sample 
                    from the latent space, and then decodes back to the data space.

    The VAE's encoder compresses the input data into a lower-dimensional latent space, and the decoder 
    attempts to reconstruct the input data from this compressed representation. This architecture is 
    useful for learning efficient data representations and generating new data that's similar to the 
    training data.
    """

    def __init__(self, D_in, H=50, H2=12, latent_dim=3):
        super(VAutoencoder, self).__init__()

        # Encoder
        self.linear1 = nn.Linear(D_in, H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2 = nn.Linear(H, H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3 = nn.Linear(H2, H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)

        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)

        # Decoder
        self.linear4 = nn.Linear(H2, H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5 = nn.Linear(H2, H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6 = nn.Linear(H, D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)

        self.relu = nn.ReLU()

    def encode(self, x):
        print('-'*10)
        print(f"Input to encode: {x.shape}")
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        print(f"Output of linear1: {lin1.shape}")
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        print(f"Output of linear2: {lin2.shape}")
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))
        print(f"Output of linear3: {lin3.shape}")

        fc1 = F.relu(self.bn1(self.fc1(lin3)))
        print(f"Output of fc1: {fc1.shape}")

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        print(f"Outputs of fc21 and fc22: {r1.shape}, {r2.shape}")

        return r1, r2

    def reparameterize(self, mu, logvar):
        print(f"Input to reparameterize: mu {mu.shape}, logvar {logvar.shape}")
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        print('-'*10)
        print(f"Input to decode: {z.shape}")
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        print(f"Output of fc3: {fc3.shape}")
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))
        print(f"Output of fc4: {fc4.shape}")

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        print(f"Output of linear4: {lin4.shape}")
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        print(f"Output of linear5: {lin5.shape}")
        output = self.lin_bn6(self.linear6(lin5))
        print(f"Output of linear6: {output.shape}")
        return output

    def forward(self, x):
        print('-'*50)
        print(f"Input to VAE: {x.shape}")
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



# Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, _, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Create datasets and data loaders
train_dataset = TensorDataset(X_train_tensor, torch.zeros(len(X_train_tensor)))
test_dataset = TensorDataset(X_test_tensor, torch.zeros(len(X_test_tensor)))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Initialize the VAE
vae = VAutoencoder(D_in=X.shape[1])

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Function to evaluate on test data
def evaluate(vae, data_loader):
    vae.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in data_loader:
            reconstructed, _, _ = vae(data)
            loss = criterion(reconstructed, data)
            total_loss += loss.item()
    return total_loss / len(data_loader)

# Training and evaluation
num_epochs = 50
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    for data, _ in train_loader:
        optimizer.zero_grad()
        reconstructed, mu, logvar = vae(data)
        loss = criterion(reconstructed, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    if epoch % 10 == 0:
        test_loss = evaluate(vae, test_loader)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Plotting
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(10, num_epochs + 1, 10), test_losses, label='Test Loss', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

print(f'Iris X shape: {X.shape}')
print(f'Iris y shape {y.shape}')

plt.show()