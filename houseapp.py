import streamlit as st
import pandas as pd
import numpy as np
import pickle  

# ================================================
# Page configuration
# ================================================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"    
)
# ================================================
# Custom CSS
# ================================================

st.markdown("""
<style>

    .main {
        background-color: #f5f7fb;
    }

    .hero {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 18px;
        opacity: 0.9;
    }

    .card {
        background-color: white;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 5px 20px rgba(0,0,0,0.07);
        margin-bottom: 20px;
    }

    .price-card {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        margin-top: 25px;
    }

    .price-title {
        font-size: 18px;
        color: #374151;
    }

    .price {
        font-size: 42px;
        font-weight: bold;
        color: #047857;
        margin-top: 10px;
    }

    .info-card {
        background-color: white;
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        border: 1px solid #e5e7eb;
    }

    .info-number {
        font-size: 28px;
        font-weight: bold;
        color: #2563eb;
    }

    .info-label {
        color: #6b7280;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)

# ================================================
# Load the trained model
# ================================================
def load_model():
    with open('hpp.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)    
    return model    

try:
    model = load_model()

except FileNotFoundError:
    st.error("An error occurred while loading the model.")
    st.stop()

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()    

# ================================================
# Ocean proximity encoding 
# ================================================
def encode_ocean_proximity(value):
    mapping = {
        "NEAR OCEAN": 0,
        "INLAND": 1,
        "NEAR BAY": 2,
        "ISLAND": 3,
        "BACK TO BAY": 4
    }
    return mapping.get(value, -1)  # Return -1 for unknown values   

# ===============================================
# prediction function
# ===============================================   
def predict_price(
        housting_median_age,
        median_income,
        ocean_proximity
):
    encode_ocean = encode_ocean_proximity(ocean_proximity)

    features =[housting_median_age, median_income] + [encode_ocean]
    prediction = model.predict([features])
    return prediction[0]  

#================================================
# sidebar inputs
# ===============================================

with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>House Price Prediction</h2>", unsafe_allow_html=True)

    st.divider()
    page = st.radio("Select Page", ["Home", "Prediction", "About"], index=0)

    st.divider()
    st.subheader("How to use this app")

    st.markdown("""
    **Step 1:** Select the page you want to visit from the sidebar.
    **Step 2:** If you choose the "Prediction Price",fill in the required details  and click the 'Predict'.
    **Step 3:** The predicted house price will be displayed on the page.
    **Step 4:** Fpr more information about the app, visit the "About" page.
    """)
    st.divider()

    st.info("""
        this app is designed to predict house prices based on user inputs.
          """)

#================================================
# Home Page
# ===============================================

if page == "Home":

    st.markdown("""
<div class="hero">
<h1>🏠 House Price Predictor</h1>
<p>
Estimate the median house value using Machine Learning
</p>
</div>
""", unsafe_allow_html=True)


    #===============================================
    # Display Image
    #===============================================

    try:

        st.image(
            "house.png",
            width=500,
            caption="Machine Learning House Price Prediction"
        )

    except FileNotFoundError:

        st.warning("house.png was not found.")


    st.markdown("## Welcome 👋")

    st.markdown("""
<div class="card">
<h3>Predict House Prices</h3>
<p>
This application uses a trained Machine Learning model
to estimate the median value of a house based on selected
characteristics.
</p>
<p>
You can experiment with different house ages, income levels,
and locations to see how the predicted value changes.
</p>
</div>
""", unsafe_allow_html=True)

#===============================================
# Feature Cards
#===============================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
<div class="info-card">
<div class="info-number">🤖</div>
<div class="info-label">
Machine Learning
</div>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="info-card">
<div class="info-number">📊</div>
<div class="info-label">
Data Driven
</div>
</div>
""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
<div class="info-card">
<div class="info-number">⚡</div>
<div class="info-label">
Instant Prediction
</div>
</div>
""", unsafe_allow_html=True)    

#===============================================
# Prediction Price Page
# ==============================================

elif page == "Prediction price":  
    st.markdown("<h2 style='text-align: center;'>Predict House Price</h2>", unsafe_allow_html=True) 
    st.divider()
    st.markdown("## Input Features")

    housing_median_age = st.number_input(
    "Housing Median Age",
    min_value=1,
    max_value=100,
    value=30,
    stop=1
)   
    median_income = st.number_input(
    "Median Income (in tens of thousands)",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    stop=0.1
)
    ocean_proximity = st.selectbox(
    "Ocean Proximity",
    options=["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
)

if st.button("Predict"):
    try:
        predicted_price = predict_price(
            housing_median_age,
            median_income,
            ocean_proximity
        )
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
    else:
        st.success(f""" Predicted Median House Price: ${predicted_price:,.2f} """)       
        st.error(f"An error occurred while predicting the house price: {e}")



            # ===============================================
            # Price Results
            # ===============================================
 
        st.markdown(f"""
<div class="price-card">
<div class="price-title">
🏠 Estimated Median House Value
</div>
<div class="price">
${predicted_price:,.2f}
</div>
</div>
""", unsafe_allow_html=True)

        st.balloons()

            # ===============================================
            # House Summary
            # ===============================================
        st.markdown("### 📋 House Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
                st.metric(
                    "🏠 House Age",
                    f"{housing_median_age} years"
                )

        with col2:
                st.metric(
                    "💵 Median Income",
                    f"{median_income:.2f}"
                )

        with col3:
                st.metric(
                    "🌊 Location",
                    ocean_proximity
                )

# ===============================================
# About Page
# ===============================================

elif page == "About":

    st.markdown("<h2 style='text-align: center;'>About This App</h2>", unsafe_allow_html=True)

    st.divider()

    st.markdown("""
<div class="card">
<h3>House Price Predictor</h3>
<p>
This application uses a trained Machine Learning model to estimate the
median value of a house based on its age, the median income of the area,
and its proximity to the ocean.
</p>
<p>
Enter the details on the <b>Predict Price</b> page and click
<b>Predict</b> to see the estimated value.
</p>
</div>
""", unsafe_allow_html=True)