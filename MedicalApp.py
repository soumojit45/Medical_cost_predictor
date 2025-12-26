import streamlit as st
import pandas as pd
import pickle
import time
from streamlit_lottie import st_lottie
import requests

def Age_Grouper(age):
    if 18<=age<=28:
        return 0
    elif 29<=age<=39:
        return 1
    elif 40<=age<=50:
        return 2
    elif 51<=age<=60:
        return 3
    else:
        return 4
    

def bmi_grouper(bmi):
    if bmi <= 18.5:
        return 0
    elif 18.5 < bmi <= 25:
        return 1
    elif 25 < bmi <= 29.9:
        return 2
    elif bmi > 29.9:
        return 3
    else:
        return 'Unknown' # Handles NaN or unexpected values

# --- PAGE CONFIG ---
st.set_page_config(page_title="HealthPredict AI", page_icon="🏥", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ASSETS ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

lottie_health = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_5njp3v83.json")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        with open('Medical_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('LMM.pkl', 'rb') as f2:
            Finalmodel = pickle.load(f2)
        return model,Finalmodel
    except FileNotFoundError:
        return None

model,FinalModel = load_model()

# --- SIDEBAR INPUTS ---
st.sidebar.header("📋 Patient Information")
st.sidebar.info("Please enter the details below to get a medical assessment.")

with st.sidebar:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=25)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    height = st.slider("Height (cm)", 100, 250, 170)
    weight = st.slider("Weight (kg)", 30, 200, 70)
    smoker = st.radio("Do you smoke?", options=["Yes", "No"])

# --- MAIN CONTENT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.title("HealthPredict AI 🏥")
    st.subheader("Advanced Medical Cost & Risk Analysis")
    st.write("Using machine learning to provide instant health insights based on your biometric data.")
    if lottie_health:
        st_lottie(lottie_health, height=300)

with col2:
    st.markdown("### User Profile Summary")
    
    # Calculate BMI locally for extra UX value
    bmi = weight / ((height/100)**2)
    
    m1, m2 = st.columns(2)
    m1.metric("Age", f"{age} yrs")
    m2.metric("BMI", f"{bmi:.1f}")
    
    m3, m4 = st.columns(2)
    m3.metric("Sex", sex)
    m4.metric("Smoker", smoker)

    st.divider()

    # --- PREDICTION LOGIC ---
    if st.button("Generate Medical Report"):
        if model is None:
            st.error("Error: 'Medical_model.pkl' not found. Please ensure the file is in the same directory.")
        else:
            with st.spinner('Analyzing medical data...'):
                time.sleep(1.5) # Simulating processing for "feel"
                
                # Prepare features (Ensure these match your model's training columns)
                sex_val = 1 if sex == "Male" else 0
                smoker_val = 1 if smoker == "Yes" else 0
                
                age_val = Age_Grouper(age)
                bmi_val = bmi_grouper(bmi)

                features = pd.DataFrame([[bmi_val,age_val,sex_val, smoker_val]], 
                                        columns=[ 'bmi','age', 'sex', 'smoker'])
                
                Refined_features = model.fit_transform(features)
                prediction = FinalModel.predict(Refined_features)[0]
                
                # Display Results
                st.balloons()
                st.markdown(f"""
                    <div class="prediction-card">
                        <h2 style='color: #ff4b4b;'>Prediction Result</h2>
                        <p style='font-size: 24px; color: #1a1a1a;'>Estimated Annual Medical Cost:</p>
                        <h1 style='font-size: 48px; color: #1a1a1a;'>${prediction:,.2f}</h1>
                    </div>
                """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Disclaimer: This tool is for informational purposes only and does not constitute medical advice.")