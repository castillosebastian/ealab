
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the data
tra, trameta = arff.loadarff('/home/sebacastillo/ealab/data/leukemia_train_38x7129.arff')
tst, tstmeta = arff.loadarff('/home/sebacastillo/ealab/data/leukemia_test_34x7129.arff')

# Convert to pandas DataFrame
train_df = pd.DataFrame(tra)
test_df = pd.DataFrame(tst)

# Decode byte strings to strings (necessary for string data in arff files)
train_df = train_df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)
test_df = test_df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)

# Assuming 'CLASS' is the column with class labels
class_column = 'CLASS'

# Initialize label encoder
label_encoder = LabelEncoder()

# Fit label encoder and return encoded labels
train_df[class_column] = label_encoder.fit_transform(train_df[class_column])
test_df[class_column] = label_encoder.transform(test_df[class_column])

# Create a mapping dictionary for class labels
class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

print("Training DataFrame:\n", train_df.head())
print("Test DataFrame:\n", test_df.head())
print("Class Mapping:\n", class_mapping)
