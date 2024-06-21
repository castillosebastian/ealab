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

# Split the leukemia data into two based on pob_ngenes_avg
leukemia_below_100 = leukemia[leukemia['pob_ngenes_avg'] < 100]
leukemia_above_100 = leukemia[leukemia['pob_ngenes_avg'] >= 100]

# Plot configuration
custom_palette = {'original': 'gray', 'augmented': 'pink'}

# Generate the plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Leukemia plot for pob_ngenes_avg below 100
sns.boxplot(ax=axes[0], x='group', y='pob_ngenes_avg', data=leukemia_below_100, palette=custom_palette)
sns.stripplot(ax=axes[0], x='group', y='pob_ngenes_avg', data=leukemia_below_100, color='darkgrey', dodge=True)
axes[0].set_xlabel('')
axes[0].set_title('Leukemia (pob_ngenes_avg < 100)')

# Leukemia plot for pob_ngenes_avg above 100
sns.boxplot(ax=axes[1], x='group', y='pob_ngenes_avg', data=leukemia_above_100, palette=custom_palette)
sns.stripplot(ax=axes[1], x='group', y='pob_ngenes_avg', data=leukemia_above_100, color='darkgrey', dodge=True)
axes[1].set_xlabel('')
axes[1].set_title('Leukemia (pob_ngenes_avg >= 100)')

# Add shared labels
fig.text(0.5, 0.04, 'Group', ha='center', va='center')
fig.text(0.06, 0.5, 'Pob Ngenes Avg', ha='center', va='center', rotation='vertical')

plt.suptitle('Gene Count Comparison of Original vs Augmented Data for Leukemia', y=1.02)
plt.tight_layout()
plt.savefig('expga1/boxplot_ngenes_leukemia_results.png')
plt.show()
