# Import statements
import json
import pandas as pd
import optuna
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sdmetrics import load_demo
from sdmetrics.reports.single_table import QualityReport
import os
import inspect
import sys
# Your root directory
root = '/home/sebacastillo/ealab/'
# Add the root directory to the system path
sys.path.append(root)
from src.bo_vae import VAutoencoder, customLoss, DataBuilder, train, test, objective, create_metadata_dict, weights_init_uniform_rule 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Parameters----------------------------------------------------------------------------------
root = '/home/sebacastillo/ealab/'
data_path = root + 'data/wine.csv'
exp_dir = root + "exp/exp_13_BO_VAE_refactor/"
class_column = 'Wine'
# BO
n_trials = 1
param_ranges = {
    'hiden1': {'low': 100, 'high': 1000},
    'hiden2': {'low': 50, 'high': 500},
    'latent_dim': {'low': 5, 'high': 20},
    'lr': {'low': 1e-5, 'high': 1e-3},
    'epochs': {'low': 800, 'high': 4000}
}
n_samples = 200
# Evaluate
show_quality_figs = True

# Run-------------------------------------------------------------------------------------------
# Load and process data
print(f'Starting data access')
df_base = pd.read_csv(data_path, sep=',')
cols = df_base.columns
D_in = df_base.shape[1]
traindata_set = DataBuilder(data_path, train=True)
testdata_set = DataBuilder(data_path, train=False)
trainloader = DataLoader(dataset=traindata_set, batch_size=1024)
testloader = DataLoader(dataset=testdata_set, batch_size=1024)
# Optimization phase
print(f'Starting optimization')
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, trainloader, testloader, param_ranges=param_ranges,device=device), n_trials=n_trials)
print(f'Optimal hyperparameters: {study.best_params}')

# Generation phase------------------------------------------------------------------------------
print(f'Starting generation')
best_params = study.best_params
model = VAutoencoder(D_in, best_params['hiden1'], best_params['hiden2'], best_params['latent_dim']).to(device)
model.apply(weights_init_uniform_rule)
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
loss_mse = customLoss()

for epoch in range(1, best_params['epochs'] + 1):
    train(epoch, model, optimizer, loss_mse, trainloader, device)

torch.save(model, exp_dir + 'vautoencoder_complete.pth')

with torch.no_grad():
    mus, logvars = [], []
    for data in testloader:
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
        mus.append(mu)
        logvars.append(logvar)
    mu = torch.cat(mus, dim=0)
    logvar = torch.cat(logvars, dim=0)

# Calculate sigma: a concise way to calculate the standard deviation σ from log-variance
sigma = torch.exp(logvar / 2)
# Sample z from q
q = torch.distributions.Normal(mu.mean(dim=0), sigma.mean(dim=0))
z = q.rsample(sample_shape=torch.Size([n_samples]))
# Decode z to generate fake data
with torch.no_grad():
    pred = model.decode(z).cpu().numpy()
# Inverse transform to get data in original scale
scaler = trainloader.dataset.standardizer
fake_data = scaler.inverse_transform(pred)

# Create a DataFrame for the fake data
df_fake = pd.DataFrame(fake_data, columns=cols)
df_fake.to_csv( exp_dir + 'syndf.csv', sep=',')

# Evaluation phase--------------------------------------------------------------------------
df_base_str = str(df_base.dtypes)
metadata_dict = create_metadata_dict(df_base_str, class_column)

# Save the dictionary to a JSON file
file_path = exp_dir + 'metadata.json'
with open(file_path, 'w') as file:
    json.dump(metadata_dict, file, indent=4)

with open(file_path, 'r') as file:
    metadata_dict = json.load(file)

my_report = QualityReport()

my_report.generate(df_base, df_fake, metadata_dict)

print(my_report.get_details(property_name='Column Shapes'))
print(my_report.get_details(property_name='Column Pair Trends'))

fig_pair_trends = my_report.get_visualization(property_name='Column Pair Trends')
fig_pair_trends.write_image(exp_dir+ "pair_trends.pdf")

fig_shapes = my_report.get_visualization(property_name='Column Shapes')
fig_shapes.write_image(exp_dir+ "shapes.pdf")

if show_quality_figs:
    fig_pair_trends.show()
    fig_shapes.show()

my_report.save(filepath= exp_dir + 'demo_data_quality_report.pkl')