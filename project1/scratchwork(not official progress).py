import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('heart.csv')

print(df.info())
print(df.describe())
print(df.head())
df=df.dropna()
df=df.drop_duplicates()
print(df.dtypes)
plt.hist(df['Age'],bins=20)
plt.title('Distribution of Patient Age')
plt.xlabel('Age')
plt.ylabel('Frequency (Number of Patients)')
plt.show()
sex_labels = df['Sex'].map({0: 'Female', 1: 'Male'})
sex_counts=sex_labels.value_counts()
sex_counts.plot(kind='bar')
plt.title('Distribution of Patient Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()
df.plot.scatter(x='Age',y='Chol')

plt.show()
female_bp = df[sex_labels == 'Female']['RestBP']
male_bp = df[sex_labels == 'Male']['RestBP']
plt.boxplot([female_bp, male_bp])
plt.show()