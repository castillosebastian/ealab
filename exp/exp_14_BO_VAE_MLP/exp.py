# Import statements
import numpy as np
import json
import pandas as pd
import optuna
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
exp_dir = root + "exp/exp_14_BO_VAE_MLP/"
class_column = 'Wine'
# BO
n_trials = 10
param_ranges = {
    'hiden1': {'low': 100, 'high': 1000},
    'hiden2': {'low': 50, 'high': 500},
    'latent_dim': {'low': 5, 'high': 20},
    'lr': {'low': 1e-5, 'high': 1e-3},
    'epochs': {'low': 800, 'high': 4000}
}
n_samples = 1300
# Evaluate
show_quality_figs = True
# Clasify
max_iter = 500

# Run-------------------------------------------------------------------------------------------
# Load and process data
print('-'*100)
print(f'Starting data access')
df_base = pd.read_csv(data_path, sep=',')
cols = df_base.columns
D_in = df_base.shape[1]
traindata_set = DataBuilder(data_path, train=True)
testdata_set = DataBuilder(data_path, train=False)
print(traindata_set.x)
trainloader = DataLoader(dataset=traindata_set, batch_size=1024)
testloader = DataLoader(dataset=testdata_set, batch_size=1024)
# Optimization phase
print(f'Starting optimization')
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, trainloader, testloader, param_ranges=param_ranges,device=device), n_trials=n_trials)
print(f'Optimal hyperparameters: {study.best_params}')
# Save to a JSON file
best_params = study.best_params
with open(exp_dir + 'best_params.json', 'w') as f:
    json.dump(best_params, f, indent=4)
print("Best parameters saved to 'best_params.json'")

# Generation phase------------------------------------------------------------------------------
print('-'*100)
print(f'Starting generation')
model = VAutoencoder(D_in, best_params['hiden1'], best_params['hiden2'], best_params['latent_dim']).to(device)
model.apply(weights_init_uniform_rule)
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
loss_mse = customLoss()

for epoch in range(1, best_params['epochs'] + 1):
    train(epoch, model, optimizer, loss_mse, trainloader, device)

torch.save(model, exp_dir + 'vautoencoder_complete.pth')
#model = torch.load(exp_dir + 'vautoencoder_complete.pth')
#model.eval()  # Set it to evaluation mode if you're doing inference

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
# Because the generation phase output target class as float
# you need to convert the target class from float to integer 
# and then from value > 0, coerce to 1 as class goes from 1 to 3.
df_fake = pd.DataFrame(fake_data, columns=cols)
df_fake[class_column] = np.round(df_fake[class_column]).astype(int)
df_fake[class_column] = np.where(df_fake[class_column]<1, 1, df_fake[class_column])
df_fake.to_csv( exp_dir + 'syndf.csv', sep=',')

# Evaluation phase--------------------------------------------------------------------------
print('-'*100)
print(f'Starting evaluation')
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

fig_pair_trends = my_report.get_visualization(property_name='Column Pair Trends')
fig_pair_trends.write_image(exp_dir+ "pair_trends.pdf")

fig_shapes = my_report.get_visualization(property_name='Column Shapes')
fig_shapes.write_image(exp_dir+ "shapes.pdf")

if show_quality_figs:
    fig_pair_trends.show()
    fig_shapes.show()

my_report.save(filepath= exp_dir + 'demo_data_quality_report.pkl')

# Clasification phase--------------------------------------------------------------------------
print('-'*100)
print(f'Starting Clasification')
X = df_base.drop(columns=[class_column])  # replace `class_column` with the name of your target column
y = df_base[class_column]

# Split the original data into train and test sets
X_train_orig, X_test, y_train_orig, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Prepare the synthetic data (assuming it has the same features as the original data)
X_train_fake = df_fake.drop(columns=[class_column])
y_train_fake = df_fake[class_column]

def create_mlp_model():
    # Define your MLP model
    # Adjust the parameters according to your needs
    model = MLPClassifier(hidden_layer_sizes=(500, 200, 100), max_iter=max_iter, alpha=0.001,
                          solver='adam', verbose=10, random_state=42)
    return model


# Create a pipeline with a scaler and the MLP classifier
pipeline_orig = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', create_mlp_model())
])

pipeline_fake = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', create_mlp_model())
])


# Train the model on the original data
pipeline_orig.fit(X_train_orig, y_train_orig)

# Train the model on the synthetic data
pipeline_fake.fit(X_train_fake, y_train_fake)

# Evaluate on the test set
y_pred_orig = pipeline_orig.predict(X_test)
y_pred_fake = pipeline_fake.predict(X_test)

# Generate classification reports
report_orig = classification_report(y_test, y_pred_orig)
report_fake = classification_report(y_test, y_pred_fake)
accuracy_orig = accuracy_score(y_test, y_pred_orig)
accuracy_fake = accuracy_score(y_test, y_pred_fake)
report_orig += "\nAccuracy: {:.4f}".format(accuracy_orig)
report_fake += "\nAccuracy: {:.4f}".format(accuracy_fake)

# Compare the results
print("Original Data Model Performance:")
print(report_orig)
print(accuracy_orig)

print("\nSynthetic Data Model Performance:")
print(report_fake)
print(accuracy_fake)

# Define file paths for saving the reports
report_orig_path = exp_dir + 'classification_report_original.txt'
report_fake_path = exp_dir + 'classification_report_synthetic.txt'

# Save the reports to text files
with open(report_orig_path, 'w') as f:
    f.write(report_orig)

with open(report_fake_path, 'w') as f:
    f.write(report_fake)

print("Classification reports saved.")


