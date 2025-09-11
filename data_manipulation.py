import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import pairwise_distances
import seaborn as sns

# Getting data in form of a dictionary from a pickle file
with open('dataset.pickle', 'rb') as f:
    data = pickle.load(f)

# Loop thru each variable in dictionary and append it to its own row
rows = []

for userID, userData in data.items():
    for inputMode, userData in userData.items():
        for deviceType, userData in userData.items():
            for contentID, userData in userData.items():

                # Extract all needed data
                inputType = userData.get('input_type', str)
                inputContent = userData.get('input_content', str)
                sensorData = userData.get('data', {})

                ts = sensorData.get('ts', [])
                rawposX = sensorData.get('rawposX', [])
                rawposY = sensorData.get('rawposY', [])
                relposX = sensorData.get('relposX', [])
                relposY = sensorData.get('relposY', [])
                velX = sensorData.get('velX', [])
                velY = sensorData.get('velY', [])
                magX = sensorData.get('magX', [])
                magY = sensorData.get('magY', [])
                magZ = sensorData.get('magZ', [])
                orientation = sensorData.get('orientation', [])
                pressure = sensorData.get('pressure', [])
                size = sensorData.get('size', [])

                required_vectors = [rawposX, rawposY, relposX, relposY,
                                    velX, velY, magX, magY, magZ,
                                    orientation, pressure, size]
                if any(len(v) == 0 for v in required_vectors):
                    continue  # skip this sample entirely

                rawposX_mean = np.mean(rawposX)
                rawposY_mean = np.mean(rawposY)
                rawposX_std = np.std(rawposX)
                rawposY_std = np.std(rawposY)
                relposX_mean = np.mean(relposX)
                relposY_mean = np.mean(relposY)
                relposX_std = np.std(relposX)
                relposY_std = np.std(relposY)
                velX_mean = np.mean(velX)
                velY_mean = np.mean(velY)
                velX_std = np.std(velX)
                velY_std = np.std(velY)
                magX_mean = np.mean(magX)
                magY_mean = np.mean(magY)
                magZ_mean = np.mean(magZ)
                magX_std = np.std(magX)
                magY_std = np.std(magY)
                magZ_std = np.std(magZ)
                orientation_mean = np.mean(orientation)
                orientation_std = np.std(orientation)
                pressure_mean = np.mean(pressure)
                pressure_std = np.std(pressure)
                size_mean = np.mean(size)
                size_std = np.std(size)

                rows.append({'rawposX_mean': rawposX_mean, 
                             'rawposY_mean': rawposY_mean, 
                             'rawposX_std': rawposX_std, 
                             'rawposY_std': rawposY_std,
                             'relposX_mean': relposX_mean,
                             'relposY_mean': relposY_mean,
                             'relposX_std': relposX_std,
                             'relposY_std': relposY_std,
                             'velX_mean': velX_mean,
                             'velY_mean': velY_mean,
                             'velX_std': velX_std,
                             'velY_std': velY_std,
                             'magX_mean': magX_mean,
                             'magY_mean': magY_mean,
                             'magZ_mean': magZ_mean,
                             'magX_std': magX_std,
                             'magY_std': magY_std,
                             'magZ_std': magZ_std,
                             'orientation_mean': orientation_mean,
                             'orientation_std': orientation_std,
                             'pressure_mean': pressure_mean,
                             'pressure_std': pressure_std,
                             'size_mean': size_mean,
                             'size_std': size_std,
                             'user_id': userID,
                             })

# Convert rows to pandas dataframe
df = pd.DataFrame(rows)

# Check if there are missing values in the data set
df = df.dropna()
df_nulls = pd.DataFrame(df.isnull().sum().sort_values(ascending=False), columns=['Number of Missing Values'])
df_nulls['% Missing'] = df.isnull().sum().sort_values(ascending=False)/len(df)
print(df_nulls)

# Split data into training and testing set
X = df.drop('user_id', axis=1)
y = df[['user_id']]

scaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size = 0.8)


# EXTRA ANALYSIS: why user 9 might have perfect classification

# Intra-class variance (consistency)
user9_data = df[df['user_id'] == 9].drop('user_id', axis=1)
user9_variance = user9_data.var().mean()
others_data = df[df['user_id'] != 9].drop('user_id', axis=1)
others_variance = others_data.var().mean()

print(f"\n[Intra-class consistency]")
print(f"Average variance of user 9's features: {user9_variance:.4f}")
print(f"Average variance of all other users: {others_variance:.4f}")

# Inter-class separability (distance to others)
dist_matrix = pairwise_distances(user9_data, others_data)
avg_distance_to_others = dist_matrix.mean()
print(f"\n[Inter-class separability]")
print(f"Average distance between user 9's samples and others: {avg_distance_to_others:.4f}")

# Compare with another user for context (e.g., user 2)
sample_other_user = df[df['user_id'] == 2].drop('user_id', axis=1)
dist_matrix_others = pairwise_distances(sample_other_user, others_data)
print(f"Average distance between user 2's samples and others: {dist_matrix_others.mean():.4f}")
'''
# Visualization: compare feature distribution for a key feature
plt.figure(figsize=(12,6))
sns.kdeplot(df[df['user_id'] == 9]['pressure_mean'], label="User 9", fill=True)
sns.kdeplot(df[df['user_id'] == 2]['pressure_mean'], label="User 2", fill=True)
sns.kdeplot(df[df['user_id'] == 14]['pressure_mean'], label="User 14", fill=True)
plt.title("Pressure Mean Distribution for Users 9, 2, and 14")
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
sns.kdeplot(df[df['user_id'] == 9]['magX_mean'], label="User 9", fill=True)
sns.kdeplot(df[df['user_id'] == 2]['magX_mean'], label="User 2", fill=True)
sns.kdeplot(df[df['user_id'] == 14]['magX_mean'], label="User 14", fill=True)
plt.title("size Mean Distribution for Users 9, 2, and 14")
plt.legend()
plt.show()
'''