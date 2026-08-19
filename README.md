# 🌾 Paddy Yield Prediction System

An end-to-end Machine Learning application designed to forecast pre-harvest paddy rice yield ($Kg/Ha$) and calculate total expected production ($Kg$) based on farm inputs, weather indices, and management practices.

## 📌 1. Agricultural Problem Addressed
Smallholder rice farming across West Africa faces significant yield variability caused by unpredictable rainfall distributions, sub-optimal fertilizer application schedules, and soil variability. Without accessible, pre-harvest predictive tools, farmers and agronomic extension officers struggle to optimize resource allocation, leading to financial loss and regional food insecurity.

This project bridges the gap between machine learning and precision agriculture by transforming multi-variable agronomic parameters into instant, actionable harvest forecasts.

## 📊 2. Dataset & Features
The model was trained on multi-location agricultural survey and experimental trial data covering **30+ agronomic parameters**. Key features include:
* Climatic Data: 30-day rainfall volume ($mm$), average growth-stage temperatures.
* Crop Management: Seed rate ($Kg/Ha$), nursery management duration, and variety type.
* Soil & Nutrients: Soil classification types, Agriblock indexing, and 40-day Urea application timing/volume ($Kg$).

## 🔬 3. Models & Methods Tested
We benchmarked distance-based and tree-based regression architectures within modular `scikit-learn` `ColumnTransformer` pipelines to prevent data leakage during preprocessing:

1. Random Forest Regressor: Evaluated for ensemble stability across non-linear agronomic features.
2. XGBoost Regressor (Selected Model): Optimized using hyperparameter tuning to capture complex interactions between rainfall indices, seed rates, and fertilizer timing.

Preprocessing Strategy: Categorical features were encoded using `OneHotEncoder`, while numerical features passed through raw to maintain feature interpretability and tree split fidelity.

## 📈 4. Model Performance & Key Findings
* Top Influencing Features: Feature importance evaluation revealed that **Seed Rate ($Kg/Ha$)**, **40-Day Urea Application ($Kg$)**, and **30-Day Rainfall ($mm$)** were the primary drivers of yield variance.
* Evaluation Metrics: Evaluated against validation datasets using Mean Absolute Error ($MAE$), Root Mean Squared Error ($RMSE$), and Coefficient of Determination ($R^2$).
* Core Insight: Tree-based algorithms like XGBoost effectively model complex, non-linear interactions across agronomic inputs without requiring strict feature scaling.

## 🚀 5. System Architecture & Web Application
The system is built as a modular, full-stack machine learning solution:
* Machine Learning Pipeline: Trained and serialized via `joblib` into `paddy_yield_pipeline.pkl`.
* API Architecture: RESTful prediction endpoints served via `FastAPI` / `Uvicorn`.
* Frontend Dashboard: Interactive user interface built with `Streamlit`, enabling real-time yield scenario planning for farmers and extension agents.

## 💡 6. Lessons Learned
* Pipeline Integrity: Resolving data type coercion issues (`pd.to_numeric`) and managing cached pipeline objects inside Streamlit deployment environments.
* Feature Engineering: Calculating yield metrics directly per unit area ($Kg/Ha$) ensures model scalability across varying land holding sizes.

## 🧬 7. Future Crop-Science Research Directions
* Spatial & Weather Integration: Incorporating automated satellite-derived NDVI vegetation indices and real-time geolocation weather API fetching.
* Soil Microbiome & Nutrient Modeling: Expanding dataset parameters to include nitrogen-fixation rates and soil organic carbon content.
* Multi-Crop Expansion: Extending the pipeline architecture to support maize, cassava, and sorghum yield estimation.
## 🛠️ Installation & Local Setup
