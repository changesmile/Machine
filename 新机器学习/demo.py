import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("./data/creditcard.csv")
data1 = data['Amount'].values.reshape(-1, 1)
data['normAmount'] = StandardScaler().fit_transform(data1)
data = data.drop(['Time', 'Amount'], axis=1)
print(data)
X = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']

number_records_fraud = len(data[data.Class == 1])  # 492
fraud_indices = np.array(data[data.Class == 1].index)

normal_indices = data[data.Class == 0].index

# 从 normal_indices 中 取 number_records_fraud(492) 个 数据
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

# Under sample dataset
under_sample_data = data.iloc[under_sample_indices, :]

X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'Class']

# Showing ratio
print("Percentage of normal transactions: ",
      len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
print("Percentage of fraud transactions: ",
      len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))
