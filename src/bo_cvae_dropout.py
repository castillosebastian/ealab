
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
    
    elif  dataset_name == "madelon":        
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

        # Map encoded labels from '-1', '1' to '0', '1'
        #train_df[class_column] = train_df[class_column].apply(lambda x: 0 if x == -1 else 1)
        #test_df[class_column] = test_df[class_column].apply(lambda x: 0 if x == -1 else 1)

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
    
    elif  dataset_name == "gisette":        
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
    
    elif  dataset_name == "gcm":        
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
    def __init__(self, root, datasetname, classcolumn, num_classes, train=True):
        self.X_train, self.X_test, self.standardizer, _, _ = load_and_standardize_data_thesis(root_dir=root, dataset_name=datasetname, class_column=classcolumn)
        
        if train:
            self.data = torch.from_numpy(self.X_train[:, :-1]).float()  # All columns except last, converted to float
            self.labels = torch.from_numpy(self.X_train[:, -1]).long()  # Last column, converted to long
        else:
            self.data = torch.from_numpy(self.X_test[:, :-1]).float()
            self.labels = torch.from_numpy(self.X_test[:, -1]).long()

        # One-hot encode labels
        self.labels = torch.nn.functional.one_hot(self.labels, num_classes=num_classes).float()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

class CVAE(nn.Module):
    def __init__(self, input_size, labels_length, H=50, H2=12, latent_dim=3, dropout_rate=0.5):
        super(CVAE, self).__init__()
        
        # Adjusted sizes to include label information
        input_size_with_label = input_size + labels_length
        adjusted_hidden_size = H2 + labels_length

        # Encoder
        self.fc1 = nn.Linear(input_size_with_label, H)
        self.fc_bn1 = nn.BatchNorm1d(num_features=H)
        self.fc2 = nn.Linear(H, H2)
        self.fc_bn2 = nn.BatchNorm1d(num_features=H2)
        self.fc2_repeat = nn.Linear(H2, H2)
        self.fc_bn2_repeat = nn.BatchNorm1d(num_features=H2)
        
        # Dropout layers after each activation
        self.dropout = nn.Dropout(dropout_rate)
        
        # Latent vectors mu and sigma
        self.fc21 = nn.Linear(H2, latent_dim)
        self.fc22 = nn.Linear(H2, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim + labels_length, adjusted_hidden_size)
        self.fc_bn3 = nn.BatchNorm1d(num_features=adjusted_hidden_size)
        self.fc3_repeat = nn.Linear(adjusted_hidden_size, adjusted_hidden_size)
        self.fc_bn3_repeat = nn.BatchNorm1d(num_features=adjusted_hidden_size)
        self.fc4 = nn.Linear(adjusted_hidden_size, H)
        self.fc_bn4 = nn.BatchNorm1d(num_features=H)
        self.fc5 = nn.Linear(H, input_size)

        self.relu = nn.ReLU()

    def encode(self, x, labels):
        combined = torch.cat((x, labels), 1)
        x = self.dropout(self.relu(self.fc_bn1(self.fc1(combined))))
        x = self.dropout(self.relu(self.fc_bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.fc_bn2_repeat(self.fc2_repeat(x))))  # Applying dropout after repeated layer
        return self.fc21(x), self.fc22(x)

    def decode(self, z, labels):
        z = torch.cat((z, labels), 1)
        z = self.dropout(self.relu(self.fc_bn3(self.fc3(z))))
        z = self.dropout(self.relu(self.fc_bn3_repeat(self.fc3_repeat(z))))  # Applying dropout after repeated layer
        z = self.dropout(self.relu(self.fc_bn4(self.fc4(z))))
        return torch.sigmoid(self.fc5(z))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

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
    for batch_idx, (data, labels) in enumerate(trainloader):
        data = data.float().to(device)
        labels = labels.long().to(device)  # Ensure labels are long type for embedding layer

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, labels)  # Pass labels to the model
        loss = loss_mse(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    if epoch % 200 == 0:
        print(f'====> Epoch: {epoch} Average training loss: {train_loss / len(trainloader.dataset):.4f}')
    
    return train_loss / len(trainloader.dataset)


def test(epoch, model, loss_mse, testloader, device):
    if len(testloader) == 0:
        print("Testloader is empty.")
        return float('inf')  # or appropriate error value

    model.eval()  # Set the model to evaluation mode
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(testloader):
            data = data.float().to(device)
            labels = labels.long().to(device)  # Ensure labels are long type

            recon_batch, mu, logvar = model(data, labels)  # Pass labels to the model
            loss = loss_mse(recon_batch, data, mu, logvar)
            if loss is not None:
                test_loss += loss.item()
            else:
                print("Encountered None loss value.")
                return float('inf')  # or appropriate error value

    average_test_loss = test_loss / len(testloader.dataset)
    if epoch % 200 == 0:
        print(f'===============> Epoch: {epoch} Average test loss: {average_test_loss:.4f}')
    
    return average_test_loss



def objective(trial,trainloader,testloader, param_ranges=None, device = 'cpu', num_classes=None):

    try: 
        # Suggest values for the hyperparameters based on the dictionary
        hiden1 = trial.suggest_int('hiden1', param_ranges['hiden1']['low'], param_ranges['hiden1']['high'])
        hiden2 = trial.suggest_int('hiden2', param_ranges['hiden2']['low'], param_ranges['hiden2']['high'])
        latent_dim = trial.suggest_int('latent_dim', param_ranges['latent_dim']['low'], param_ranges['latent_dim']['high'])
        lr = trial.suggest_float('lr', param_ranges['lr']['low'], param_ranges['lr']['high'])
        epochs = trial.suggest_int('epochs', param_ranges['epochs']['low'], param_ranges['epochs']['high'])
        dropout_rate = trial.suggest_int('dropout_rate', param_ranges['dropout_rate']['low'], param_ranges['dropout_rate']['high'])          
        D_in = trainloader.dataset.data.shape[1]
        
        # Initialize model, optimizer, and loss function with suggested values
        model = CVAE(input_size=D_in, labels_length=num_classes, H=hiden1, H2=hiden2,latent_dim=latent_dim, dropout_rate=dropout_rate).float().to(device)
        model.apply(weights_init_uniform_rule)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_mse = customLoss()

        # Training and validation process
        best_test_loss = float('inf')
        epochs_no_improve = 0
        patience = 50  # Number of epochs to wait for improvement before stopping

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
    except Exception as e:
        print(f"Exception in trial: {e}")
        return float('inf')  # Return a high loss in case of failure

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

