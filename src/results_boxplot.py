import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
data = pd.read_csv('expga1/experiments_results.csv')

# Helper function to categorize experiments
def categorize_experiments(df, original_data_list, experiment_prefix, group_length):
    df = df[df['experiment_name'].str.contains(experiment_prefix)].copy()
    df.loc[:, 'experiment_group'] = df['experiment_name'].apply(lambda x: x[:group_length])
    df.loc[:, 'group'] = df['experiment_group'].apply(lambda x: 'original' if x in original_data_list else 'augmented')
    return df

# Leukemia
leukemia_original_data = ['leukemia_base_0001', 'leukemia_base_0012', 'leukemia_base_0013']
leukemia = categorize_experiments(data, leukemia_original_data, 'leukemia', 18)
leukemia = leukemia[~leukemia['experiment_group'].isin(['leukemia_base_0003', 'leukemia_base_0006', 'leukemia_base_0007', 'leukemia_base_0008'])]

# Gisette
gisette_original_data = ['gisette_base_0020', 'gisette_base_0022']
gisette = categorize_experiments(data, gisette_original_data, 'gisette', 17)
gisette_selected_experiments = ['gisette_base_0005', 'gisette_base_0020', 'gisette_base_0022']
gisette = gisette[gisette['experiment_group'].isin(gisette_selected_experiments)].copy()
gisette.loc[:, 'group'] = gisette['experiment_group'].apply(lambda x: 'original' if x in gisette_original_data else 'augmented')

# Madelon
madelon_original_data = ['mandelon_base_0017', 'mandelon_base_0023']
madelon = categorize_experiments(data, madelon_original_data, 'mandelon', 18)
madelon = madelon[~madelon['experiment_group'].isin(['mandelon_base_0004', 'mandelon_base_0021'])]
madelon.loc[:, 'group'] = madelon['experiment_group'].apply(lambda x: 'original' if x in madelon_original_data else 'augmented')

# GCM
gcm_original_data = ['gcm_base_0026', 'gcm_base_0037', 'gcm_base_0038', 'gcm_base_0039', 'gcm_base_0040']
gcm = categorize_experiments(data, gcm_original_data, 'gcm', 13)

# Plot configuration
custom_palette = {'original': 'gray', 'augmented': 'pink'}

# Generate the plot
plt.figure(figsize=(14, 10))

# Leukemia plot
plt.subplot(2, 2, 1)
sns.boxplot(x='group', y='pob_accuracy_avg', data=leukemia, palette=custom_palette)
sns.stripplot(x='group', y='pob_accuracy_avg', data=leukemia, color='grey', dodge=True)
plt.title('Leukemia Accuracy Average by Group')

# Gisette plot
plt.subplot(2, 2, 2)
sns.boxplot(x='group', y='pob_accuracy_avg', data=gisette, palette=custom_palette)
sns.stripplot(x='group', y='pob_accuracy_avg', data=gisette, color='grey', dodge=True)
plt.title('Gisette Accuracy Average by Group')

# Madelon plot
plt.subplot(2, 2, 3)
sns.boxplot(x='group', y='pob_accuracy_avg', data=madelon, palette=custom_palette)
sns.stripplot(x='group', y='pob_accuracy_avg', data=madelon, color='grey', dodge=True)
plt.title('Madelon Accuracy Average by Group')

# GCM plot
plt.subplot(2, 2, 4)
sns.boxplot(x='group', y='pob_accuracy_avg', data=gcm, palette=custom_palette)
sns.stripplot(x='group', y='pob_accuracy_avg', data=gcm, color='grey', dodge=True)
plt.title('GCM Accuracy Average by Group')

plt.tight_layout()
plt.show()
