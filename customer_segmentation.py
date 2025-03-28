import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load data
df = pd.read_csv('../data/bank_transactions.csv')

# Basic check
print(df.shape)
print(df.head()) 
print(df.info())
print(df.isnull().sum())
print(df.describe())

df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'], errors='coerce')
# Calculate age based on birth year (assuming current year is 2025)
df['Age'] = 2025 - df['CustomerDOB'].dt.year
# Extract transaction hour from TransactionTime
df['TransactionHour'] = df['TransactionTime'] // 10000
# Categorize transaction time into time of day (e.g., Morning, Afternoon)
df['TimeOfDay'] = pd.cut(df['TransactionHour'], bins=[-1,11,17,23], labels=['Morning', 'Afternoon', 'Evening'])
# 시간대별 거래량 시각화
df['TransactionHour'] = df['TransactionTime'] // 10000  # 시간 추출
df['TimeSlot'] = pd.cut(df['TransactionHour'], bins=[-1, 5, 11, 17, 23], labels=['Night', 'Morning', 'Afternoon', 'Evening'])

df['TimeSlot'].value_counts().plot(kind='bar')
plt.title("Transaction Counts by Time Slot")
plt.xlabel("Time Slot")
plt.ylabel("Count")
plt.show()

# Extract hour from TransactionTime
df['TransactionHour'] = df['TransactionTime'] // 10000

# Define time bins and labels
time_bins = [-1, 5, 11, 17, 23]
time_labels = ['Night', 'Morning', 'Afternoon', 'Evening']

# Create a new column for time slot
df['TimeSlot'] = pd.cut(df['TransactionHour'], bins=time_bins, labels=time_labels)

# Check distribution of transactions by time slot
print(df['TimeSlot'].value_counts())

# Calculate age (assume current year is 2025)
df['Age'] = 2025 - df['CustomerDOB'].dt.year

# Define age bins and labels
age_bins = [0, 20, 30, 40, 50, 60, 100]
age_labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60+']

# Create age group column
df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

# Group by AgeGroup and calculate count & average transaction amount
age_summary = df.groupby('AgeGroup')['TransactionAmount (INR)'].agg(['count', 'mean']).reset_index()
print(age_summary)

# Plot transaction count by age group
plt.figure(figsize=(8,5))
plt.bar(age_summary['AgeGroup'], age_summary['count'])
plt.title('Transaction Count by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Transaction Count')
plt.show()

# Plot average transaction amount by age group
plt.figure(figsize=(8,5))
plt.bar(age_summary['AgeGroup'], age_summary['mean'], color='orange')
plt.title('Average Transaction Amount by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Avg. Transaction Amount (INR)')
plt.show()

# Group by Gender
gender_summary = df.groupby('CustGender')['TransactionAmount (INR)'].agg(['count', 'mean']).reset_index()
print(gender_summary)

# Plot
plt.figure(figsize=(8,5))
plt.bar(gender_summary['CustGender'], gender_summary['mean'], color='skyblue')
plt.title('Avg Transaction Amount by Gender')
plt.xlabel('Gender')
plt.ylabel('Avg. Transaction Amount (INR)')
plt.show()
# Top 10 locations by transaction count
location_top10 = df['CustLocation'].value_counts().nlargest(10)
print(location_top10)

# Plot
plt.figure(figsize=(10,5))
location_top10.plot(kind='bar', color='purple')
plt.title('Top 10 Transaction Locations')
plt.xlabel('Location')
plt.ylabel('Transaction Count')
plt.xticks(rotation=45)
plt.show()

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

# Drop rows with missing values in relevant columns
cluster_df = df[['Age', 'CustGender', 'CustAccountBalance', 'TransactionAmount (INR)']].dropna()

# Encode gender (F → 0, M → 1)
le = LabelEncoder()
cluster_df['CustGender'] = le.fit_transform(cluster_df['CustGender'])

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_df)

# Run KMeans clustering (we'll use 4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_df['Cluster'] = kmeans.fit_predict(scaled_features)

# Add back to original DataFrame
df = df.join(cluster_df['Cluster'], how='left')

# Visualize cluster distribution
sns.countplot(data=cluster_df, x='Cluster')
plt.title('Customer Segmentation by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select features for clustering
features = ['Age', 'CustAccountBalance', 'TransactionAmount (INR)']
df_cluster = df[features].dropna()

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_cluster)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df_cluster['Cluster'] = kmeans.fit_predict(scaled_features)

# Add cluster back to original dataframe
df.loc[df_cluster.index, 'Cluster'] = df_cluster['Cluster']

# Check the number of customers per cluster
print(df['Cluster'].value_counts())
# Calculate average values for each cluster
cluster_summary = df.groupby('Cluster')[['Age', 'CustAccountBalance', 'TransactionAmount (INR)']].mean()
print(cluster_summary)

import matplotlib.pyplot as plt

# Plot average values for Age, Account Balance, and Transaction Amount by cluster
cluster_summary.plot(kind='bar', figsize=(10,6))
plt.title('Cluster-wise Average Metrics')
plt.ylabel('Average Value')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()
# Define segment names based on cluster number and their characteristics
segment_labels = {
    0: 'Moderate Spenders',
    1: 'Young High Value',
    2: 'High Transaction Customers',
    # Add/adjust labels based on your actual cluster_summary output
}

# Create a new column for segment name
df['Segment'] = df['Cluster'].map(segment_labels)

# Check if it's mapped correctly
print(df[['Cluster', 'Segment']].head())

import matplotlib.pyplot as plt

df['Segment'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Customer Segment Distribution')
plt.ylabel('Customer Count')
plt.xticks(rotation=45)
plt.show()




