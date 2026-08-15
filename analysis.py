#Load libraries
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('paddydataset.csv')
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Target engineering: Calculate Yield per Hectare
df.columns = df.columns.str.strip()
df['Yield_per_Ha'] = df['Paddy yield(in Kg)'] / df['Hectares']

# Define Features (X) and Target (y)
x = df.drop(columns=['Paddy yield(in Kg)', 'Yield_per_Ha', 'Hectares', 'Trash(in bundles)'], errors='ignore')
y = df['Yield_per_Ha']
# --- CRITICAL FIX: FORCE NUMERICAL COLUMNS TO BE FLOATS ---
# Define categorical columns explicitly
known_categorical = ['Variety', 'Soil Types', 'Agriblock', 'Nursery']

# Automatically get all other feature columns as numerical
known_numerical = [col for col in x.columns if col not in known_categorical]

# Force convert all numerical columns to numeric (coercing bad text/spaces to NaN)
for col in known_numerical:
    x[col] = pd.to_numeric(x[col], errors='coerce')

# Fill any NaN values caused by conversion with the column median
x[known_numerical] = x[known_numerical].fillna(x[known_numerical].median())

# Ensure categorical columns are strictly strings
for col in known_categorical:
    x[col] = x[col].astype(str)

#EDA
import matplotlib.pyplot as plt
import seaborn as sns

# A. Target Variable Distribution
plt.figure(figsize=(9, 4.5))
sns.histplot(df['Yield_per_Ha'], kde=True, color='teal')
plt.title('Distribution of Paddy Yield per Hectare (Kg/Ha)')
plt.xlabel('Yield (Kg/Ha)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# B. Numerical Correlation Heatmap (Select key agronomic & weather factors)
sample_num_cols = ['Yield_per_Ha', 'Seedrate(in Kg)', 'Urea_40Days', 
                   '30DRain( in mm)', 'Min temp_D1_D30', 'Max temp_D1_D30']
plt.figure(figsize=(10, 6))
sns.heatmap(df[sample_num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Key Inputs vs Yield per Hectare')
plt.show()

# C. Yield by Variety (Categorical EDA)
if 'Variety' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Variety', y='Yield_per_Ha', data=df)
    plt.xticks(rotation=45)
    plt.title('Yield per Hectare by Crop Variety')
    plt.show()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib

# Use the explicitly forced lists
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', known_numerical),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), known_categorical)
    ]
)

# Build & fit pipeline
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

# Train on cleaned X
xgb_pipeline.fit(x, y)

# Save the fitted pipeline
joblib.dump(xgb_pipeline, "paddy_yield_pipeline.pkl")
print("Pipeline re-trained with strict data types and saved successfully!")

#Separate and process features(because dataset contain categorical and numerical data)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#identify categorical and numerical column names
categorical_cols = x.select_dtypes(include=['object','category','str']).columns.tolist()
numerical_cols = x.select_dtypes(include=['int64','float64']).columns.tolist()

#Define preprocessing for numerical(scaling)
numerical_transformer = StandardScaler()

#Define preprocesing for categorical(onehot encoding)
categorical_transformer =OneHotEncoder(handle_unknown='ignore',sparse_output=False)

#combine transformer into a columntransformer
prepocessor =ColumnTransformer(transformers=[('num',numerical_transformer,numerical_cols),('cat',categorical_transformer,categorical_cols)])

#split data in training and test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#train ml models
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Random Forest Model Pipeline
rf_pipeline = Pipeline(steps=[('preprocessor', prepocessor),('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

#XGBoost model pipeline
xgb_pipeline = Pipeline(steps=[('preprocessor', prepocessor), ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))])

#fit models
rf_pipeline.fit(x_train, y_train)
xgb_pipeline.fit(x_train, y_train)

#Performance evaluation 
def evaluate_model(model, x_test, y_test, name="model"):
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"==={name} Evaluation Results===")
    print(f"mae : {mae :.2f} Kg")
    print(f"rmse : {rmse :.2f} Kg\n")
    print(f"r2 : {r2 :.4f}")
evaluate_model(rf_pipeline, x_test, y_test, name="Random Forest")
evaluate_model(xgb_pipeline, x_test, y_test,name="XGBoost")

#Interpretation
import matplotlib.pyplot as plt

#Extract feature names after encoding
if categorical_cols:
    cat_encoder = xgb_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cat_cols = list(cat_encoder.get_feature_names_out(categorical_cols))
else:
    encoded_cat_cols = []

all_feature_names = numerical_cols + encoded_cat_cols

# Extract importances from XGBoost
importances = xgb_pipeline.named_steps['regressor'].feature_importances_
feature_imp = pd.Series(importances, index = all_feature_names).sort_values(ascending=False)

# Plot top 15 features
plt.figure(figsize=(10, 6))
feature_imp.head(15).plot(kind='barh', color='skyblue')
plt.title('Top 15 Agronomic Features Influencing Yield per Hectare')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#Save model
import	joblib
# Save model along with dataset column default values for Streamlit
best_model = xgb_pipeline

# Compute baseline default inputs for numerical and categorical columns
defaults = {}
for col in numerical_cols:
    defaults[col] = float(x[col].median())

for col in categorical_cols:
    defaults[col] = str(x[col].mode()[0])

# Package model and defaults together
model_payload = {
    'pipeline': best_model,
    'default_inputs': defaults,
    'feature_names': x.columns.tolist()}

# Save payload
joblib.dump(model_payload, "paddy_yield_pipeline.pkl")
print("Model pipeline and feature defaults saved to paddy_yield_pipeline.pkl")