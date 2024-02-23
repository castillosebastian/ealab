import pandas as pd
import plotly.express as px
import json

# Load JSON data into a DataFrame
# Load JSON data
with open('exp/exp_37_GS_VAE3_MLP_leukemia_latent_dim/grid_search_results.json', 'r') as file:
    data = json.load(file)

results_df = pd.DataFrame(data)

# Calculate the mean of 'test_loss'
# Aggregate over 'latent_dim' to calculate mean 'test_loss'
aggregated_df = results_df.groupby('latent_dim')['test_loss'].mean().reset_index()


# Create scatter plot
fig = px.scatter(aggregated_df, x='latent_dim', y='test_loss', 
                 labels={"latent_dim": "Latent Dimension", "test_loss": "Test Loss"},
                 title="Mean Test Loss vs Latent Dimension")

# Show the plot
fig.show()

# Save the plot to a PNG file
fig.write_image('exp/exp_37_GS_VAE3_MLP_leukemia_latent_dim/scatter_plot.png')