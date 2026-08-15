import streamlit as st
import joblib
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Paddy Yield Intelligence",
    page_icon="🌾",
    layout="wide"
)

# Load the trained pipeline model
@st.cache_resource
def load_model():
    loaded_file = joblib.load("paddy_yield_pipeline.pkl")
    
    # If it's a dictionary payload, extract the pipeline automatically
    if isinstance(loaded_file, dict):
        return loaded_file.get('pipeline', list(loaded_file.values())[0])
        
    return loaded_file

try:
    model = load_model()
except Exception as e:
    st.error("Model file 'paddy_yield_pipeline.pkl' not found. Please train and export the model first.")

# Header Section
st.title("🌾 Paddy Yield Prediction System")
st.caption("AI-powered agricultural forecasting tool for optimizing farm productivity.")
st.markdown("---")

# Organized Multi-Column Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Farm & Input Parameters")
    
    with st.container(border=True):
        hectares = st.number_input("Land Area (Hectares)", min_value=0.1, value=2.0, step=0.1)
        seed_rate = st.number_input("Seed Rate (Kg)", min_value=1.0, value=25.0, step=1.0)
        urea = st.number_input("Urea Application at 40 Days (Kg)", min_value=0.0, value=50.0, step=5.0)
        
        # Expandable section for advanced parameters
        with st.expander("Advanced Agronomic Settings"):
            variety = st.selectbox("Crop Variety", ["Variety_A", "Ponmani", "CO_43"])
            soil_type = st.selectbox("Soil Type", ["Alluvial", "Clay", "Loam"])
            rain_30d = st.number_input("Rainfall (First 30 Days in mm)", value=120.0)

with col2:
    st.subheader("📊 Yield Forecast")
    
    if st.button("Calculate Expected Yield", type="primary", use_container_width=True):
        # Build complete feature record with user inputs and standard defaults
        feature_data = {
            'Hectares': hectares,
            'Seedrate(in Kg)': seed_rate,
            'Urea_40Days': urea,
            'Variety': variety,
            'Soil Types': soil_type,
            '30DRain( in mm)': rain_30d,
            # Populate defaults for remaining weather/management columns
            'Agriblock': 'Block_1',
            'Nursery': 'dry',
            'Nursery area (Cents)': 10.0,
            'LP_Mainfield(in Tonnes)': 2.0,
            'LP_nurseryarea(in Tonnes)': 0.5,
            'DAP_20days': 30.0,
            'Weed28D_thiobencarb': 1.0,
            'Potassh_50Days': 25.0,
            'Micronutrients_70Days': 5.0,
            'Pest_60Day(in ml)': 100.0,
            '30DAI(in mm)': 50.0,
            '30_50DRain( in mm)': 80.0,
            '30_50DAI(in mm)': 40.0,
            '51_70DRain(in mm)': 60.0,
            '51_70AI(in mm)': 30.0,
            '71_105DRain(in mm)': 40.0,
            '71_105DAI(in mm)': 20.0,
            'Min temp_D1_D30': 22.0,
            'Max temp_D1_D30': 32.0,
            'Min temp_D31_D60': 23.0,
            'Max temp_D31_D60': 33.0,
            'Min temp_D61_D90': 24.0,
            'Max temp_D61_D90': 34.0,
            'Min temp_D91_D120': 22.0,
            'Max temp_D91_D120': 31.0,
            'Inst Wind Speed_D1_D30(in Knots)': 5.0,
            'Inst Wind Speed_D31_D60(in Knots)': 6.0,
            'Inst Wind Speed_D61_D90(in Knots)': 5.5,
            'Inst Wind Speed_D91_D120(in Knots)': 4.8,
            'Wind Direction_D1_D30': 180.0,
            'Wind Direction_D31_D60': 190.0,
            'Wind Direction_D61_D90': 175.0,
            'Wind Direction_D91_D120': 185.0,
            'Relative Humidity_D1_D30': 70.0,
            'Relative Humidity_D31_D60': 75.0,
            'Relative Humidity_D61_D90': 80.0,
            'Relative Humidity_D91_D120': 72.0
        }
        
        # Convert dictionary to DataFrame matching the model's exact expected structure
        input_df = pd.DataFrame([feature_data])
        
        try:
            # Predict yield per hectare
            predicted_yield_per_ha = model.predict(input_df)[0]
            total_predicted_yield = predicted_yield_per_ha * hectares
            
            # Display modern summary metrics
            st.success("Prediction Generated Successfully!")
            
            m1, m2 = st.columns(2)
            m1.metric("Predicted Productivity", f"{predicted_yield_per_ha:,.2f} Kg/Ha")
            m2.metric("Total Estimated Harvest", f"{total_predicted_yield:,.2f} Kg")
            
        except Exception as err:
            st.error(f"Error making prediction: {err}")