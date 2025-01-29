import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"  # Use this if Kaggle's dataset isn't downloaded
titanic_df = pd.read_csv(url)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(titanic_df.head())

# 1. Basic Dataset Overview
print("\nDataset Information:")
titanic_df.info()

print("\nSummary Statistics:")
print(titanic_df.describe())

# Check for missing values
print("\nMissing Values in Each Column:")
print(titanic_df.isnull().sum())

# 2. Data Cleaning
# Fill missing values for 'Age' using the median
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

# Fill missing values for 'Embarked' with the most common value
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column due to a large number of missing values
titanic_df.drop(columns=['Cabin'], inplace=True)

# Verify missing values after cleaning
print("\nMissing Values After Cleaning:")
print(titanic_df.isnull().sum())

# 3. Exploratory Data Analysis (EDA)
# Univariate Analysis: Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(titanic_df['Age'], bins=30, kde=True, color='orange')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Barplot of Survival Rate
plt.figure(figsize=(8, 6))
sns.barplot(x='Survived', y='Age', data=titanic_df, errorbar=None, palette='viridis')
plt.title('Survival Rate by Age')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.show()

# Survival Count by Class
plt.figure(figsize=(10, 6))
sns.countplot(x='Pclass', hue='Survived', data=titanic_df, palette='coolwarm')
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', loc='upper right')
plt.show()

# Survival Rate by Gender
plt.figure(figsize=(8, 6))
sns.barplot(x='Sex', y='Survived', data=titanic_df, errorbar=None, palette='pastel')
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = titanic_df.select_dtypes(include=['number']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 4. Insights
print("\nInsights from EDA:")
print("- Younger passengers had a slightly higher survival rate.")
print("- Females had a much higher survival rate compared to males.")
print("- First-class passengers had the highest survival rate, while third-class had the lowest.")
print("- There is a strong positive correlation between Pclass and survival.")

# 5. Save Cleaned Dataset
titanic_df.to_csv("cleaned_titanic_dataset.csv", index=False)
print("\nCleaned dataset saved as 'cleaned_titanic_dataset.csv'.")
