import os
os.makedirs('models', exist_ok=True)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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

        df_com['Source'] = 'Commodity'
        df_ind['Source'] = 'Industry'
        df_com['Year'] = df_ind['Year'] = year

        df_com.columns = df_com.columns.str.strip()
        df_ind.columns = df_ind.columns.str.strip()

        df_com.rename(columns={'Commodity Code': 'Code', 'Commodity Name': 'Name'}, inplace=True)
        df_ind.rename(columns={'Industry Code': 'Code', 'Industry Name': 'Name'}, inplace=True)

        all_data.append(pd.concat([df_com, df_ind], ignore_index=True))

    except Exception as e:
        print(f"❌ Error processing year {year}: {e}")

# Final data
df = pd.concat(all_data, ignore_index=True)
print("✅ Combined Data Loaded")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())

if 'Unnamed: 7' in df.columns:
    df.drop(columns=['Unnamed: 7'], inplace=True)

print(df.info())
print("\nMissing values:\n", df.isnull().sum())

if 'Supply Chain Emission Factors with Margins' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Supply Chain Emission Factors with Margins'], bins=50, kde=True)
    plt.title('Target Variable Distribution')
    plt.xlabel('Emission Factor')
    plt.tight_layout()
    plt.show()
else:
    print("❌ Target column not found!")

if 'Substance' in df.columns and 'Unit' in df.columns and 'Source' in df.columns:
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

required_cols = ['Name', 'Supply Chain Emission Factors with Margins']
if all(col in df.columns for col in required_cols):
    top_emitters = df.groupby('Name')['Supply Chain Emission Factors with Margins'].mean().sort_values(ascending=False).head(10).reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Supply Chain Emission Factors with Margins', y='Name', data=top_emitters, palette='viridis')
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

for col in ['Name', 'Code', 'Year']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

if 'Supply Chain Emission Factors with Margins' in df.columns:
    X = df.drop(columns=['Supply Chain Emission Factors with Margins'])
    y = df['Supply Chain Emission Factors with Margins']
    print("✅ Feature and target data prepared:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
else:
    print("❌ Target column missing for ML preparation.")

for cat_col in ['Substance', 'Unit', 'Source']:
    if cat_col in df.columns:
        plt.figure(figsize=(6, 3))
        sns.countplot(x=df[cat_col])
        plt.title(f"Count Plot: {cat_col}")
        plt.tight_layout()
        plt.show()

num_df = df.select_dtypes(include=np.number)
if not num_df.empty:
    plt.figure(figsize=(12, 8))
    sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()
else:
    print("❌ No numerical columns found for heatmap.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Features normalized")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

RF_model = RandomForestRegressor(random_state=42)
RF_model.fit(X_train, y_train)
RF_y_pred = RF_model.predict(X_test)
RF_mse = mean_squared_error(y_test, RF_y_pred)
RF_rmse = np.sqrt(RF_mse)
RF_r2 = r2_score(y_test, RF_y_pred)
print(f"Random Forest - RMSE: {RF_rmse}, R2: {RF_r2}")

LR_model = LinearRegression()
LR_model.fit(X_train, y_train)
LR_y_pred = LR_model.predict(X_test)
LR_mse = mean_squared_error(y_test, LR_y_pred)
LR_rmse = np.sqrt(LR_mse)
LR_r2 = r2_score(y_test, LR_y_pred)
print(f"Linear Regression - RMSE: {LR_rmse}, R2: {LR_r2}")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

y_pred_best = best_model.predict(X_test)
HP_mse = mean_squared_error(y_test, y_pred_best)
HP_rmse = np.sqrt(HP_mse)
HP_r2 = r2_score(y_test, y_pred_best)
print(f"Tuned Random Forest - RMSE: {HP_rmse}, R2: {HP_r2}")

results = {
    'Model': ['Random Forest (Default)', 'Linear Regression', 'Random Forest (Tuned)'],
    'MSE': [RF_mse, LR_mse, HP_mse],
    'RMSE': [RF_rmse, LR_rmse, HP_rmse],
    'R2': [RF_r2, LR_r2, HP_r2]
}
comparison_df = pd.DataFrame(results)
print(comparison_df)

joblib.dump(best_model, 'models/LR_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Model and scaler saved to 'models' folder")
