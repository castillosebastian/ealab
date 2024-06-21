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
gisette_selected_experiments = ['gisette_base_0005', 'gisette_base_0020', 'gisette_base_0022']
gisette = categorize_experiments(data, gisette_original_data, 'gisette', 17)
gisette = gisette[gisette['experiment_group'].isin(gisette_selected_experiments)].copy()
gisette.loc[:, 'group'] = gisette['experiment_group'].apply(lambda x: 'original' if x in gisette_original_data else 'augmented')

# Madelon
madelon_original_data = ['mandelon_base_0017', 'mandelon_base_0023']
madelon = categorize_experiments(data, madelon_original_data, 'mandelon', 18)
madelon = madelon[~madelon['experiment_group'].isin(['mandelon_base_0004', 'mandelon_base_0021'])]
madelon.loc[:, 'group'] = madelon['experiment_group'].apply(lambda x: 'original' if x in madelon_original_data else 'augmented')

# Plot configuration
custom_palette = {'original': 'gray', 'augmented': 'pink'}

# Generate the plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Leukemia plot
sns.boxplot(ax=axes[0], x='group', y='pob_ngenes_avg', data=leukemia, palette=custom_palette)
sns.stripplot(ax=axes[0], x='group', y='pob_ngenes_avg', data=leukemia, color='darkgrey', dodge=True)
axes[0].set_xlabel('')
axes[0].set_title('Leukemia')

# Gisette plot
sns.boxplot(ax=axes[1], x='group', y='pob_ngenes_avg', data=gisette, palette=custom_palette)
sns.stripplot(ax=axes[1], x='group', y='pob_ngenes_avg', data=gisette, color='darkgrey', dodge=True)
axes[1].set_xlabel('')
axes[1].set_title('Gisette')

# Madelon plot
sns.boxplot(ax=axes[2], x='group', y='pob_ngenes_avg', data=madelon, palette=custom_palette)
sns.stripplot(ax=axes[2], x='group', y='pob_ngenes_avg', data=madelon, color='darkgrey', dodge=True)
axes[2].set_xlabel('')
axes[2].set_title('Madelon')

# Add legend for binary datasets
fig.legend(['Binary (Leukemia, Gisette, Madelon)'], loc='upper center', ncol=2, frameon=False)

# Add shared labels
fig.text(0.5, 0.04, 'Group', ha='center', va='center')
fig.text(0.06, 0.5, 'Pob Ngenes Avg', ha='center', va='center', rotation='vertical')

plt.suptitle('Gene Count Comparison of Original vs Augmented Data Across Datasets', y=1.02)
plt.tight_layout()
plt.savefig('expga1/boxplot_ngenes_results.png')
plt.show()
