import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load Excel and setup
excel_file = 'SupplyChainEmissionFactorsforUSIndustriesCommodities.xlsx'
years = range(2010, 2017)

all_data = []

for year in years:
    try:
        df_com = pd.read_excel(excel_file, sheet_name=f'{year}_Detail_Commodity')
        df_ind = pd.read_excel(excel_file, sheet_name=f'{year}_Detail_Industry')

        # Add source and year columns
        df_com['Source'] = 'Commodity'
        df_ind['Source'] = 'Industry'
        df_com['Year'] = df_ind['Year'] = year

        # Clean column names
        df_com.columns = df_com.columns.str.strip()
        df_ind.columns = df_ind.columns.str.strip()

        # Rename for consistency
        df_com.rename(columns={'Commodity Code': 'Code', 'Commodity Name': 'Name'}, inplace=True)
        df_ind.rename(columns={'Industry Code': 'Code', 'Industry Name': 'Name'}, inplace=True)

        # Combine
        all_data.append(pd.concat([df_com, df_ind], ignore_index=True))

    except Exception as e:
        print(f"❌ Error processing year {year}: {e}")

# Final data
df = pd.concat(all_data, ignore_index=True)
print("✅ Combined Data Loaded")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())

# Remove unnamed extra column if it exists
if 'Unnamed: 7' in df.columns:
    df.drop(columns=['Unnamed: 7'], inplace=True)

# Check data types and null values
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# Plot histogram of target variable
if 'Supply Chain Emission Factors with Margins' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Supply Chain Emission Factors with Margins'], bins=50, kde=True)
    plt.title('Target Variable Distribution')
    plt.xlabel('Emission Factor')
    plt.tight_layout()
    plt.show()
else:
    print("❌ Target column not found!")

# Print unique values in categorical columns
if 'Substance' in df.columns and 'Unit' in df.columns and 'Source' in df.columns:
    print("\nSubstances:\n", df['Substance'].value_counts())
    print("\nUnits:\n", df['Unit'].value_counts())
    print("\nSources:\n", df['Source'].value_counts())

    # Map values
    substance_map = {
        'carbon dioxide': 0,
        'methane': 1,
        'nitrous oxide': 2,
        'other GHGs': 3
    }

    unit_map = {
        'kg/2018 USD, purchaser price': 0,
        'kg CO2e/2018 USD, purchaser price': 1
    }

    source_map = {
        'Commodity': 0,
        'Industry': 1
    }

    df['Substance'] = df['Substance'].map(substance_map)
    df['Unit'] = df['Unit'].map(unit_map)
    df['Source'] = df['Source'].map(source_map)
else:
    print("❌ One or more categorical columns missing.")

# Check if required columns exist before proceeding
required_cols = ['Name', 'Supply Chain Emission Factors with Margins']
if all(col in df.columns for col in required_cols):
    top_emitters = df.groupby('Name')['Supply Chain Emission Factors with Margins'].mean().sort_values(ascending=False).head(10).reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Supply Chain Emission Factors with Margins',
        y='Name',
        data=top_emitters,
        palette='viridis'
    )
    for i, row in top_emitters.iterrows():
        plt.text(row['Supply Chain Emission Factors with Margins'] + 0.01, i, f'#{i+1}', va='center')
    plt.title('Top 10 Emitting Industries')
    plt.xlabel('Emission Factor')
    plt.ylabel('Industry')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("❌ Cannot plot top emitters, required columns missing.")

# Drop non-numeric columns before ML
for col in ['Name', 'Code', 'Year']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Prepare ML inputs
if 'Supply Chain Emission Factors with Margins' in df.columns:
    X = df.drop(columns=['Supply Chain Emission Factors with Margins'])
    y = df['Supply Chain Emission Factors with Margins']

    print("✅ Feature and target data prepared:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
else:
    print("❌ Target column missing for ML preparation.")

# Plot categorical distributions
for cat_col in ['Substance', 'Unit', 'Source']:
    if cat_col in df.columns:
        plt.figure(figsize=(6, 3))
        sns.countplot(x=df[cat_col])
        plt.title(f"Count Plot: {cat_col}")
        plt.tight_layout()
        plt.show()

# Correlation heatmap
num_df = df.select_dtypes(include=np.number)
if not num_df.empty:
    plt.figure(figsize=(12, 8))
    sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()
else:
    print("❌ No numerical columns found for heatmap.")
