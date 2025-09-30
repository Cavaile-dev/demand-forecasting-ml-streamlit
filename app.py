import streamlit as st
import pickle # Changed from joblib to pickle
import pandas as pd
import numpy as np
import plotly.express as px

# --- Configuration and Setup ---

st.set_page_config(
    page_title="Demand Forecasting ML App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the filename of your saved model
MODEL_FILE = 'model\demand_forecasting_random_forest_model.pkl' # Updated file extension
# Define the features used by your FINAL model (must be in the exact order)
TOP_FEATURES = [
    'Avg_Price_per_Unit',
    'Revenue generated',
    'Profit',
    'Production volumes',
    'Manufacturing costs',
    'Defect rates',
    'Price',
    'Lead time',
    'Costs',
    'Manufacturing lead time',
    'Inspection results_Pending'
]

# Load the trained model using pickle
@st.cache_resource # Use Streamlit's caching to load the model only once
def load_model():
    try:
        # Open the file in 'read binary' mode
        with open(MODEL_FILE, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_FILE}' not found. Please ensure it is in the same directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model()

# --- Utility Functions ---

def prepare_input_data(user_inputs):
    """Calculates engineered features and structures input for the model."""
    
    # 1. Calculate Engineered Features
    # Note: We assume Units Sold is the input for calculation
    units_sold_calc = user_inputs['Units Sold'] if user_inputs['Units Sold'] > 0 else 1
    avg_price_per_unit = user_inputs['Revenue generated'] / units_sold_calc
    profit = user_inputs['Revenue generated'] - user_inputs['Manufacturing costs'] - user_inputs.get('Shipping costs', 0)
    
    # 2. Create the feature vector (must match TOP_FEATURES order)
    data = {
        'Avg_Price_per_Unit': [avg_price_per_unit],
        'Revenue generated': [user_inputs['Revenue generated']],
        'Profit': [profit],
        'Production volumes': [user_inputs['Production volumes']],
        'Manufacturing costs': [user_inputs['Manufacturing costs']],
        'Defect rates': [user_inputs['Defect rates']],
        'Price': [user_inputs['Price']],
        'Lead time': [user_inputs['Lead time']],
        'Costs': [user_inputs.get('Costs', 0)],
        'Manufacturing lead time': [user_inputs['Manufacturing lead time']],
        'Inspection results_Pending': [user_inputs['Inspection results_Pending']]
    }
    
    # Return as a DataFrame for the model
    return pd.DataFrame(data, columns=TOP_FEATURES)



st.title("🛍️ AI-Powered Demand Forecasting")
st.markdown("Use the optimized Random Forest Regressor to predict the **Number of Products Sold** based on key product and supply chain metrics.")

# ----------------------------------------------------
# 1. Sidebar for Model Performance & Feature Info
# ----------------------------------------------------
st.sidebar.header("Model Performance")
# Use the values from your final, tuned model evaluation
st.sidebar.metric("R² Score", "0.70+") 
st.sidebar.metric("MAE (Avg. Error)", "± 132 Units") 
st.sidebar.info("Model performance metrics reflect its accuracy on unseen data.")

st.sidebar.header("Key Drivers of Demand")
st.sidebar.markdown(
    """
    **1. Price & Profit:** The model is most sensitive to **Avg. Price per Unit** and **Profit**.
    **2. Supply Chain:** **Production Volumes** and **Manufacturing Costs** are highly influential.
    **3. Quality:** **Defect Rates** are a significant negative predictor of sales.
    """
)

# ----------------------------------------------------
# 2. Main Prediction Input Form
# ----------------------------------------------------
if model:
    st.header("Product Input & Forecast")
    
    col1, col2, col3 = st.columns(3)
    
    # Inputs for key financial features
    with col1:
        st.subheader("Financial Metrics")
        price = st.number_input("Product Price", min_value=1.0, value=50.0, step=0.01)
        units_sold = st.number_input("Est. Units Sold (for Avg. Price Calc.)", min_value=1, value=500)
        revenue = st.number_input("Revenue Generated", min_value=1.0, value=25000.0, step=100.0)
    
    # Inputs for key supply chain features
    with col2:
        st.subheader("Production & Quality")
        production_volumes = st.number_input("Production Volumes", min_value=1, value=750)
        manuf_costs = st.number_input("Manufacturing Costs", min_value=1.0, value=45.0)
        defect_rates = st.slider("Defect Rates (%)", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
        
    # Inputs for time and inspection
    with col3:
        st.subheader("Logistics & Inspection")
        lead_time = st.number_input("Supplier Lead Time (days)", min_value=1, max_value=30, value=15)
        manuf_lead_time = st.number_input("Manufacturing Lead Time (days)", min_value=1, max_value=30, value=10)
        
        inspection_pending = st.selectbox(
            "Inspection Status",
            options=["Not Pending", "Pending"],
            index=0
        )
        inspection_results_pending = 1 if inspection_pending == "Pending" else 0
        
        costs = st.number_input("Other Supply Chain Costs", min_value=0.0, value=500.0)
        
    
    # --- Prediction Button ---
    if st.button("📈 Forecast Demand"):
        
        # Gather all user inputs into a dictionary
        user_inputs = {
            'Price': price,
            'Units Sold': units_sold,
            'Revenue generated': revenue,
            'Production volumes': production_volumes,
            'Manufacturing costs': manuf_costs,
            'Defect rates': defect_rates,
            'Lead time': lead_time,
            'Manufacturing lead time': manuf_lead_time,
            'Inspection results_Pending': inspection_results_pending,
            'Costs': costs
        }
        
        # Prepare data for prediction
        input_df = prepare_input_data(user_inputs)
        
        # Make Prediction
        with st.spinner('Calculating demand forecast...'):
            forecast = model.predict(input_df)[0]
        
        # --- Display Results ---
        st.subheader("Demand Forecast Result")
        
        col_res, col_chart = st.columns(2)
        
        with col_res:
            st.metric(
                label="Predicted Number of Products Sold",
                value=f"{int(round(forecast)):,}",
                delta="Forecasted Demand"
            )
            
            avg_price_calc = input_df['Avg_Price_per_Unit'].iloc[0]
            st.markdown(f"**Calculated Avg. Price per Unit:** ${avg_price_calc:.2f}")
            if avg_price_calc > 50:
                st.info("Price is within the mid-range based on this scenario.")
            
        with col_chart:
            # Simple visualization of the prediction vs a historical range
            historical_data = {
                'Metric': ['Historical Mean', 'Historical Max', 'Your Forecast'],
                'Units': [500, 1000, forecast] # Placeholder values
            }
            df_hist = pd.DataFrame(historical_data)
            
            fig = px.bar(
                df_hist,
                x='Metric',
                y='Units',
                color='Metric',
                title="Forecast vs. Historical Baseline",
                color_discrete_map={'Historical Mean': 'gray', 'Historical Max': 'red', 'Your Forecast': 'green'}
            )
            st.plotly_chart(fig, use_container_width=True)