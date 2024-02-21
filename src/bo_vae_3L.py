
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sdmetrics import load_demo
from scipy.io import arff

def load_and_standardize_data(path):
    # read in from csv
    df = pd.read_csv(path, sep=',')
    # replace nan with -99
    df = df.fillna(-99)
    df = df.values.reshape(-1, df.shape[1]).astype('float32')
    # randomly split
    X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)
    # standardize values
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler, df


def load_and_standardize_data_thesis(root_dir, dataset_name, class_column):

    if dataset_name == 'leukemia':
        # File paths for leukemia dataset
        train_file_path = os.path.join(root_dir, 'data', 'leukemia_train_38x7129.arff')
        test_file_path = os.path.join(root_dir, 'data', 'leukemia_test_34x7129.arff')

        # Load the data
        tra, _ = arff.loadarff(train_file_path)
        tst, _ = arff.loadarff(test_file_path)

        # Convert to pandas DataFrame
        train_df = pd.DataFrame(tra)
        test_df = pd.DataFrame(tst)

        # Decode byte strings to strings (necessary for string data in arff files)
        train_df = train_df.apply(lambda x: x.decode() if isinstance(x, bytes) else x)
        test_df = test_df.apply(lambda x: x.decode() if isinstance(x, bytes) else x)

        # Initialize label encoder
        label_encoder = LabelEncoder()

        # Fit label encoder and return encoded labels
        train_df[class_column] = label_encoder.fit_transform(train_df[class_column])
        test_df[class_column] = label_encoder.transform(test_df[class_column])

        # Create a mapping dictionary for class labels
        class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        # Combine the train and test dataframes
        df = pd.concat([train_df, test_df], ignore_index=True)

        # Standardize only the feature columns (assuming last column is label)
        feature_columns = df.columns[df.columns != class_column]
        #feature_columns = df.columns
        scaler = StandardScaler()
        train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
        test_df[feature_columns] = scaler.transform(test_df[feature_columns])

        train_df = train_df.to_numpy()
        test_df = test_df.to_numpy()

        return train_df, test_df, scaler, df, class_mapping
    
    elif dataset_name == 'madelon':
        # File paths for leukemia dataset
        train_file_path = os.path.join(root_dir, 'data', 'madelon.trn.arff')
        test_file_path = os.path.join(root_dir, 'data', 'madelon.tst.arff')

        # Load the data
        tra, _ = arff.loadarff(train_file_path)
        tst, _ = arff.loadarff(test_file_path)

        # Convert to pandas DataFrame
        train_df = pd.DataFrame(tra)
        test_df = pd.DataFrame(tst)

        # Decode byte strings to strings (necessary for string data in arff files)
        train_df = train_df.apply(lambda x: x.decode() if isinstance(x, bytes) else x)
        test_df = test_df.apply(lambda x: x.decode() if isinstance(x, bytes) else x)

        # Initialize label encoder
        label_encoder = LabelEncoder()

        # Fit label encoder and return encoded labels
        train_df[class_column] = label_encoder.fit_transform(train_df[class_column])
        test_df[class_column] = label_encoder.transform(test_df[class_column])

        # Create a mapping dictionary for class labels
        class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        # Combine the train and test dataframes
        df = pd.concat([train_df, test_df], ignore_index=True)

        # Standardize only the feature columns (assuming last column is label)
        feature_columns = df.columns[df.columns != class_column]
        #feature_columns = df.columns
        scaler = StandardScaler()
        train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
        test_df[feature_columns] = scaler.transform(test_df[feature_columns])

        train_df = train_df.to_numpy()
        test_df = test_df.to_numpy()

        return train_df, test_df, scaler, df, class_mapping
    elif dataset_name == 'gisette':
        # File paths for leukemia dataset
        train_file_path = os.path.join(root_dir, 'data', 'gisette_train.arff')
        test_file_path = os.path.join(root_dir, 'data', 'gisette_test.arff')

        # Load the data
        tra, _ = arff.loadarff(train_file_path)
        tst, _ = arff.loadarff(test_file_path)

        # Convert to pandas DataFrame
        train_df = pd.DataFrame(tra)
        test_df = pd.DataFrame(tst)

        # Decode byte strings to strings (necessary for string data in arff files)
        train_df = train_df.apply(lambda x: x.decode() if isinstance(x, bytes) else x)
        test_df = test_df.apply(lambda x: x.decode() if isinstance(x, bytes) else x)

        # Initialize label encoder
        label_encoder = LabelEncoder()

        # Fit label encoder and return encoded labels
        train_df[class_column] = label_encoder.fit_transform(train_df[class_column])
        test_df[class_column] = label_encoder.transform(test_df[class_column])

        # Create a mapping dictionary for class labels
        class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        # Combine the train and test dataframes
        df = pd.concat([train_df, test_df], ignore_index=True)

        # Standardize only the feature columns (assuming last column is label)
        feature_columns = df.columns[df.columns != class_column]
        #feature_columns = df.columns
        scaler = StandardScaler()
        train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
        test_df[feature_columns] = scaler.transform(test_df[feature_columns])

        train_df = train_df.to_numpy()
        test_df = test_df.to_numpy()

        return train_df, test_df, scaler, df, class_mapping
    
    elif dataset_name == 'gcm':
        # File paths for leukemia dataset
        train_file_path = os.path.join(root_dir, 'data', 'GCM_Training.arff')
        test_file_path = os.path.join(root_dir, 'data', 'GCM_Test.arff')

        # Load the data
        tra, _ = arff.loadarff(train_file_path)
        tst, _ = arff.loadarff(test_file_path)

        # Convert to pandas DataFrame
        train_df = pd.DataFrame(tra)
        test_df = pd.DataFrame(tst)

        # Decode byte strings to strings (necessary for string data in arff files)
        train_df = train_df.apply(lambda x: x.decode() if isinstance(x, bytes) else x)
        test_df = test_df.apply(lambda x: x.decode() if isinstance(x, bytes) else x)

        # Initialize label encoder
        label_encoder = LabelEncoder()

        # Fit label encoder and return encoded labels
        train_df[class_column] = label_encoder.fit_transform(train_df[class_column])
        test_df[class_column] = label_encoder.transform(test_df[class_column])

        # Create a mapping dictionary for class labels
        class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        # Combine the train and test dataframes
        df = pd.concat([train_df, test_df], ignore_index=True)

        # Standardize only the feature columns (assuming last column is label)
        feature_columns = df.columns[df.columns != class_column]
        #feature_columns = df.columns
        scaler = StandardScaler()
        train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
        test_df[feature_columns] = scaler.transform(test_df[feature_columns])

        train_df = train_df.to_numpy()
        test_df = test_df.to_numpy()

        return train_df, test_df, scaler, df, class_mapping

    else:
        pass


class DataBuilder(Dataset):
    def __init__(self, root, datasetname, classcolumn,  train=True):
        self.X_train, self.X_test, self.standardizer, _, _ = load_and_standardize_data_thesis(root_dir=root, dataset_name=datasetname, class_column=classcolumn)
        if train:
            self.x = torch.from_numpy(self.X_train)
            self.len=self.x.shape[0]
        else:
            self.x = torch.from_numpy(self.X_test)
            self.len=self.x.shape[0]
        del self.X_train
        del self.X_test
    def __getitem__(self,index):
        return self.x[index]
    def __len__(self):
        return self.len

class VAutoencoder(nn.Module):
    def __init__(self,D_in,H=50,H2=12,H3=20,latent_dim=3):

        #Encoder
        super(VAutoencoder,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2 = nn.Linear(H,H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3 = nn.Linear(H2, H3)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H3)
        self.linear4 = nn.Linear(H3,H3)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H3)

        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(H3, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim) # For mu
        self.fc22 = nn.Linear(latent_dim, latent_dim) # For logvar

        # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H3)
        self.fc_bn4 = nn.BatchNorm1d(H3)

        # Decoder        
        self.linear5=nn.Linear(H3,H3)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H3)
        self.linear6=nn.Linear(H3,H2)
        self.lin_bn6 = nn.BatchNorm1d(num_features=H2)
        self.linear7=nn.Linear(H2,H)
        self.lin_bn7 = nn.BatchNorm1d(num_features=H)
        self.linear8=nn.Linear(H,D_in)
        self.lin_bn8 = nn.BatchNorm1d(num_features=D_in)

        self.relu = nn.ReLU()

    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))
        lin4 = self.relu(self.lin_bn4(self.linear4(lin3)))

        fc1 = F.relu(self.bn1(self.fc1(lin4)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn5(self.linear5(fc4)))
        lin5 = self.relu(self.lin_bn6(self.linear6(lin4)))
        lin6 = self.relu(self.lin_bn7(self.linear7(lin5)))
        return self.lin_bn8(self.linear8(lin6))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def train(epoch, model, optimizer, loss_mse, trainloader, device):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(trainloader):
        data = data.float().to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_mse(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    if epoch % 200 == 0:
        print('====> Epoch: {} Average training loss: {:.4f}'.format(
            epoch, train_loss / len(trainloader.dataset)))    
    return train_loss / len(trainloader.dataset)

def test(epoch, model, loss_mse, testloader, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            data = data.float().to(device)
            recon_batch, mu, logvar = model(data)
            loss = loss_mse(recon_batch, data, mu, logvar)
            test_loss += loss.item()

    average_test_loss = test_loss / len(testloader.dataset)
    if epoch % 200 == 0:
        print('===============> Epoch: {} Average test loss: {:.4f}'.format(epoch, average_test_loss))
    return average_test_loss

def objective(trial,trainloader,testloader, param_ranges=None, device = 'cpu'):
    # Suggest values for the hyperparameters based on the dictionary
    hiden1 = trial.suggest_int('hiden1', param_ranges['hiden1']['low'], param_ranges['hiden1']['high'])
    hiden2 = trial.suggest_int('hiden2', param_ranges['hiden2']['low'], param_ranges['hiden2']['high'])
    hiden3 = trial.suggest_int('hiden3', param_ranges['hiden3']['low'], param_ranges['hiden3']['high'])
    latent_dim = trial.suggest_int('latent_dim', param_ranges['latent_dim']['low'], param_ranges['latent_dim']['high'])
    lr = trial.suggest_float('lr', param_ranges['lr']['low'], param_ranges['lr']['high'])
    epochs = trial.suggest_int('epochs', param_ranges['epochs']['low'], param_ranges['epochs']['high'])
    
    D_in = trainloader.dataset.x.shape[1]

    # Initialize model, optimizer, and loss function with suggested values
    model = VAutoencoder(D_in, hiden1, hiden2,hiden3, latent_dim).float().to(device)
    model.apply(weights_init_uniform_rule)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_mse = customLoss()

    # Training and validation process
    best_test_loss = float('inf')
    epochs_no_improve = 0
    patience = 10  # Number of epochs to wait for improvement before stopping

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


    # Return the final test loss
    return test_loss


def create_metadata_dict(dtype_str, primary_key):
    # Initialize the dictionary structure
    metadata = {
        "primary_key": primary_key,
        "columns": {}
    }
    # Split the string by newlines and iterate over the lines
    for line in dtype_str.split('\n'):
        # Ignore the 'dtype: object' line
        if line.strip() == 'dtype: object':
            continue

        # Split each line by spaces and filter out empty strings
        parts = [part for part in line.split(' ') if part]

        # Check if there are two parts (column name and data type)
        if len(parts) == 2:
            column_name, dtype = parts

            # Determine the sdtype based on dtype
            if dtype in ['int64', 'float64']:
                sdtype = 'numerical'
            else:
                sdtype = 'unknown'  # Placeholder for other data types

            # Populate the metadata dictionary
            metadata['columns'][column_name] = {"sdtype": sdtype}

    return metadata

def create_dictionary(class_column, cols):
    template = {
        "primary_key": class_column,
        "columns": {}
    }
    for col in cols:
        #if col != class_column:
        template["columns"][col] = {"sdtype": "numerical"}

    return template
