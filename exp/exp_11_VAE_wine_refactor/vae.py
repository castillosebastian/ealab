import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sdmetrics import load_demo
from sdmetrics.reports.single_table import QualityReport
import json
import plotly

# Parameters----------------------------------------------------------------------
root = '/home/sebacastillo/ealab/'
DATA_PATH = root + 'data/wine.csv'
exp_dir = root + "exp/exp_11_VAE_wine_refactor/"
class_column = 'Wine'
# VAE
hiden1 = 500
hiden2 = 200
latent_dim = 10
lr = 1.5e-3
# Training
epochs = 1200
log_interval = 50
# Generation
n_samples = 200
# Quality reports
show_quality_figs = True

# Processsing-----------------------------------------------------------------------
df_base = pd.read_csv(DATA_PATH, sep=',')
cols = df_base.columns


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

class DataBuilder(Dataset):
    def __init__(self, path, train=True):
        self.X_train, self.X_test, self.standardizer, _ = load_and_standardize_data(DATA_PATH)
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

traindata_set=DataBuilder(DATA_PATH, train=True)
testdata_set=DataBuilder(DATA_PATH, train=False)
trainloader=DataLoader(dataset=traindata_set,batch_size=1024)
testloader=DataLoader(dataset=testdata_set,batch_size=1024)

print(f'Loaded datasets: train: {type(trainloader.dataset.x)}, test: {type(testloader.dataset.x)}')
print(f'Shapes of datasets: train: {trainloader.dataset.x.shape}, test: {testloader.dataset.x.shape}')

class Autoencoder(nn.Module):
    def __init__(self,D_in,H=50,H2=12,latent_dim=3):

        #Encoder
        super(Autoencoder,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2=nn.Linear(H,H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3=nn.Linear(H2,H2)
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
        self.linear4=nn.Linear(H2,H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5=nn.Linear(H2,H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6=nn.Linear(H,D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)

        self.relu = nn.ReLU()

    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

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

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))



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

# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

D_in = df_base.shape[1]
model = Autoencoder(D_in, hiden1, hiden2, latent_dim).to(device)
model.apply(weights_init_uniform_rule)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_mse = customLoss()

"""## Train Model"""
val_losses = []
train_losses = []
test_losses = []

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(trainloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_mse(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    if epoch % 200 == 0:
        print('====> Epoch: {} Average training loss: {:.4f}'.format(
            epoch, train_loss / len(trainloader.dataset)))
        train_losses.append(train_loss / len(trainloader.dataset))

def test(epoch):
    with torch.no_grad():
        test_loss = 0
        for batch_idx, data in enumerate(testloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_mse(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            if epoch % 200 == 0:
                print('===============> Epoch: {} Average test loss: {:.4f}'.format(
                    epoch, test_loss / len(testloader.dataset)))
            test_losses.append(test_loss / len(testloader.dataset))

for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)


with torch.no_grad():
    for batch_idx, data in enumerate(testloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

scaler = trainloader.dataset.standardizer
recon_row = scaler.inverse_transform(recon_batch[0].reshape(1, -1).cpu().numpy())
real_row = scaler.inverse_transform(testloader.dataset.x[0].reshape(1, -1).cpu().numpy())
df = pd.DataFrame(np.vstack((recon_row, real_row)), columns = cols)
print(f'Reconstructed row 1 vs Real row 2 data:\n {df}')

"""
Not to bad right (the first row is the reconstructed row, the second one the real row from the 
data)? However, what we want is to built this row not with the real input so to speak,
since right now we were giving the model the complete rows with their 14 columns, 
condensed it to 3 input parameters, just to blow it up again to the corresponding 14 columns.
What I want to do is to create these 14 rows by giving the model 3 latent factors as input. 
Let's have a look at these latent variables. 
"""
sigma = torch.exp(logvar/2)


"""Mu represents the mean for each of our latent factor values, logvar the log of the
 standard deviation. Each of these have a distribution by itself. We have 54 cases in 
 our test data, so we have 3x54 different mu and logvar. We can have a look at the 
 distribution of each of the 3 latent variables:"""

"""
All of the latent variables have a mean around zero, but the last latent factor has 
a wider standard deviation. So when we sample values from each of these latent variables,
the last value will vary much more then the other two. I assume a normal distribution 
for all the latent factors.
"""

# sample z from q
q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
z = q.rsample(sample_shape=torch.Size([n_samples]))

"""With these three latent factors we can now start and create fake data for our 
dataset and see how it looks like:"""

with torch.no_grad():
    pred = model.decode(z).cpu().numpy()

fake_data = scaler.inverse_transform(pred)
print(f'Creating synthetic data from Autoencoder {fake_data.shape}')

df_fake = pd.DataFrame(fake_data, columns = cols)
df_fake.to_csv( exp_dir + 'syndf.csv', sep=',')
df_fake[class_column] = np.round(df_fake[class_column]).astype(int)
df_fake[class_column] = np.where(df_fake[class_column]<1, 1, df_fake[class_column])


# print(f'For comparison the sythetic data:\n {df_fake.head(10)}')
# print(f'For comparison the real data:\n {df_base.sample(10)}')
print(f'For comparison by group the sythetic data: {df_fake.groupby(class_column).mean()}')
print(f'For comparison by group the real data: {df_base.groupby(class_column).mean()}')


# TODO
# Comparison and evaluation
# graphs
# https://github.com/lschmiddey/deep_tabular_augmentation/blob/main/Notebooks/DeepLearning_DataAugmentation_RF.ipynb

df_base_str = str(df_base.dtypes)

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


metadata_dict = create_metadata_dict(df_base_str, class_column)

# Save the dictionary to a JSON file
file_path = exp_dir + 'metadata.json'
with open(file_path, 'w') as file:
    json.dump(metadata_dict, file, indent=4)

with open(file_path, 'r') as file:
    metadata_dict = json.load(file)

my_report = QualityReport()

"""
Column Shapes
    Does the synthetic data capture the shape of each column?
    The shape of a column describes its overall distribution. The higher the score, 
    the more similar the distributions of real and synthetic data.
    This yields a separate score for every column. The final Column 
    Shapes score is the average of all columns.
Column Pair Trends
    Does the synthetic data capture trends between pairs of columns?
    The trend between two columns describes how they vary in relation 
    to each other, for example the correlation. The higher the score, 
    the more the trends are alike.

"""


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
# load it at any point in the future
#my_report = QualityReport.load(filepath='demo_data_quality_report.pkl')