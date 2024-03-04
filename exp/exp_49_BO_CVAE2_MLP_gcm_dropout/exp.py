# Import statements
import numpy as np
import json
import pandas as pd
import optuna
import matplotlib.pyplot as plt
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

# Parameters----------------------------------------------------------------------------------
exp_dir = root + "/exp/exp_49_BO_CVAE2_MLP_gcm_dropout/"
dataset_name = 'gcm'
class_column = 'class'
num_classes = 14
# BO
n_trials = 10
param_ranges = {
    'hiden1': {'low': 200, 'high': 500},
    'hiden2': {'low': 100, 'high': 200},    
    'latent_dim': {'low': 5, 'high': 80},
    'lr': {'low': 1e-5, 'high': 1e-3},
    'epochs': {'low': 100, 'high': 5000},
    'dropout_rate':  {'low': 0.05, 'high': 0.5}

}
n_samples = 3000
# Evaluate
evaluate = False
show_quality_figs = False
# Final Test Clasification with MLP
max_iter = 500

# Run-------------------------------------------------------------------------------------------
# Load and process data
print('-'*100)
print(f'Starting data access')
train_df, test_df, scaler, df_base, class_mapping = load_and_standardize_data_thesis(root, dataset_name, class_column)
print(f'Data set dimensions: {df_base.shape}')
print(f'class maping: {class_mapping}')
cols = df_base.columns
D_in = train_df.shape[1]
traindata_set = DataBuilder(root, dataset_name, class_column, num_classes, train=True)
testdata_set = DataBuilder(root, dataset_name, class_column, num_classes, train=False)
print(f'Train data after scale and encode class: {traindata_set.data}')
trainloader = DataLoader(dataset=traindata_set, batch_size=1024)
testloader = DataLoader(dataset=testdata_set, batch_size=1024)
# Optimization phase
print(f'Starting optimization')
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, trainloader, testloader, 
                                       param_ranges=param_ranges,
                                       device=device,
                                       num_classes=num_classes), n_trials=n_trials)
print(f'Optimal hyperparameters: {study.best_params}')
# Save to a JSON file
best_params = study.best_params
with open(exp_dir + 'best_params.json', 'w') as f:
    json.dump(best_params, f, indent=4)
print("Best parameters saved to 'best_params.json'")

# Plot optimization history
opt_history = optuna.visualization.plot_optimization_history(study)
opt_history.show()
opt_history.write_image(exp_dir + "opt_history.png")
# Plot hyperparameter importance
param_importance = optuna.visualization.plot_param_importances(study)
param_importance.show()
param_importance.write_image(exp_dir + "param_importance.png")
# Plot slice
slice_plot = optuna.visualization.plot_slice(study)
slice_plot.show()
slice_plot.write_image(exp_dir + "slice_plot.png")
# Plot contour of hyperparameters
contour_plot = optuna.visualization.plot_contour(study, params=['hiden1', 'hiden2', 'latent_dim', 'lr', 'epochs', 'dropout_rate'])
contour_plot.show()
contour_plot.write_image(exp_dir + "contour_plot.png")

# Generation phase------------------------------------------------------------------------------
print('-'*100)
print(f'Starting generation')
model = CVAE(input_size= trainloader.dataset.data.shape[1],
             labels_length=num_classes, 
             H=best_params['hiden1'], 
             H2=best_params['hiden2'],             
             latent_dim=best_params['latent_dim'],
             dropout_rate= best_params['dropout_rate'], 
             ).float().to(device)
model.apply(weights_init_uniform_rule)
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
loss_mse = customLoss()
 # Training and validation process
best_test_loss = float('inf')
epochs_no_improve = 0
patience = 50  # Number of epochs to wait for improvement before stopping
epochs = max_iter
for epoch in range(1, epochs + 1):
    train(epoch, model, optimizer, loss_mse, trainloader, device)
    test_loss = test(epoch, model, loss_mse, testloader, device)

    # Check if test loss improved
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        epochs_no_improve = 0  # Reset counter
    else:
        epochs_no_improve += 1

    # Early stopping check
    if epochs_no_improve == patience:
        print(f"Early stopping triggered at epoch {epoch}: test loss has not improved for {patience} consecutive epochs.")
        break

torch.save(model, exp_dir + 'vautoencoder_complete.pth')
#model = torch.load(exp_dir + 'vautoencoder_complete.pth')
#model.eval()  # Set it to evaluation mode if you're doing inference

with torch.no_grad():
    mus, logvars = [], []
    for data, labels in testloader:
        # Ensure data and labels are on the correct device
        data = data.to(device)
        labels = labels.to(device)

        # Get the reconstructed batch, mu, and logvar from the model
        recon_batch, mu, logvar = model(data, labels)
        
        mus.append(mu)
        logvars.append(logvar)

    # Concatenate all mu and logvar values
    mu = torch.cat(mus, dim=0)
    logvar = torch.cat(logvars, dim=0)


# Calculate sigma: a concise way to calculate the standard deviation σ from log-variance
sigma = torch.exp(logvar / 2)
# Sample z from q
q = torch.distributions.Normal(mu.mean(dim=0), sigma.mean(dim=0))
# samples to generate
n_samples_per_label = int(n_samples/num_classes)  # Number of samples you want to generate per label
labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]  
# Initialize an empty list to hold the generated data
generated_data = []

for label in labels:
    # Create a tensor of the specific label, repeated n_samples_per_label times
    specific_labels = torch.ones(n_samples_per_label, dtype=torch.long) * label
    # One-hot encode the labels
    specific_labels_one_hot = torch.nn.functional.one_hot(specific_labels, num_classes=num_classes).float().to(device)
    # Sample z from the distribution
    z = q.rsample(sample_shape=torch.Size([n_samples_per_label]))
    # Decode z to generate fake data, conditioned on the specific labels
    with torch.no_grad():
        pred = model.decode(z, specific_labels_one_hot).cpu().numpy()        
        pred = scaler.inverse_transform(pred)
        pred= np.hstack([pred, specific_labels.numpy()[:, None]])                
        generated_data.append(pred)

# Concatenate all generated data
df_fake = np.concatenate(generated_data, axis=0)

# Create a DataFrame for the fake data
# Because the generation phase output target class as float
# you need to convert the target class from float to integer. 
# And when there is no 0 class, coerce 0 to 1 .
df_fake = pd.DataFrame(df_fake, columns=cols)
df_fake[class_column] = np.round(df_fake[class_column]).astype(int)
class_counts = df_fake['class'].value_counts()
print(f'class counts {class_counts}')      
df_fake.to_csv( exp_dir + 'syndf.csv', sep=',')

# Evaluation phase--------------------------------------------------------------------------
if evaluate:
    print('-'*100)
    print(f'Starting evaluation')
    df_base_str = str(df_base.dtypes)
    #metadata_dict = create_metadata_dict(df_base_str, class_column)
    metadata_dict = create_dictionary(class_column=class_column,cols=cols)

    # Save the dictionary to a JSON file
    file_path = exp_dir + 'metadata.json'
    with open(file_path, 'w') as file:
        json.dump(metadata_dict, file, indent=4)

    with open(file_path, 'r') as file:
        metadata_dict = json.load(file)

    my_report = QualityReport()
    my_report.generate(df_base, df_fake, metadata_dict)

    syn_score = my_report.get_score()
    syn_score_name = exp_dir + 'syn_score.txt'    
    with open(syn_score_name, 'w') as f:
        f.write(str(syn_score))
    syn_properties = my_report.get_properties()
    syn_properties_name = exp_dir + 'syn_properties.txt'
    with open(syn_properties_name, 'w') as f:
        f.write(str(syn_properties))

    if show_quality_figs:
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
print("End of the script")


