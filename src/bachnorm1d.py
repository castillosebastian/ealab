import torch
import torch.nn as nn

print('First example: each feature with linear progression')
# Sample input (batch size = 4, number of features = 3)
input = torch.tensor([[1.0, 2.0, 3.0], 
                      [4.0, 5.0, 6.0], 
                      [7.0, 8.0, 9.0], 
                      [10.0, 11.0, 12.0]])

# Batch normalization layer for 1D input with 3 features
batch_norm = nn.BatchNorm1d(num_features=3)

# Apply batch normalization
output = batch_norm(input)

print("Input:\n", input)
print("Output:\n", output)


print('Second example: not linear progression')
# More varied sample input
input = torch.tensor([[1.0, 20.0, 3.0], 
                      [14.0, 5.0, 16.0], 
                      [7.0, 8.0, 29.0], 
                      [10.0, 11.0, 12.0]])

# Batch normalization layer for 1D input with 3 features
batch_norm = nn.BatchNorm1d(num_features=3)

# Apply batch normalization
output = batch_norm(input)

print("Input:\n", input)
print("Output:\n", output)

