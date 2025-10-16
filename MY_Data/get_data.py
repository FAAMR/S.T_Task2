import pandas as pd
import os

# Public GitHub mirror of the Heart Failure Prediction dataset
url = 'https://raw.githubusercontent.com/dsrscientist/dataset1/master/heartdisease_data.csv'

# Load dataset directly from the URL
df = pd.read_csv(url)

# Ensure the data directory exists
os.makedirs('MyDaTA', exist_ok=True)

# Save locally inside MyDaTA folder
df.to_csv('MyDaTA/data_raw.csv', index=False)

print("Heart Failure dataset downloaded and saved to MyDaTA/data_raw.csv")

