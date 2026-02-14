import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import pickle

# Page configuration
st.set_page_config(
    page_title="Wine Clustering Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive styling with animated background
st.markdown("""
    <style>
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        min-height: 100vh;
    }
    
    /* Glassmorphism card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin-bottom: 20px;
    }
    
    /* Title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 0 0 30px rgba(255, 107, 107, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px rgba(255, 107, 107, 0.3); }
        to { text-shadow: 0 0 40px rgba(255, 107, 107, 0.6), 0 0 60px rgba(254, 202, 87, 0.3); }
    }
    
    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    
    /* Input labels */
    .input-label {
        color: #feca57 !important;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 40px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Result box */
    .result-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        border: 2px solid rgba(255, 255, 255, 0.2);
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .cluster-number {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .cluster-label {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.3rem;
        margin-top: 10px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        border-radius: 10px;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
    }
    
    /* Section headers */
    .section-header {
        color: #48dbfb;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(72, 219, 251, 0.3);
    }
    
    /* Wine emoji animation */
    .wine-icon {
        font-size: 4rem;
        animation: bounce 2s ease-in-out infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Divider styling */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #feca57 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('wine_clustering_data.csv')
    return df

# Train model
@st.cache_resource
def train_model():
    df = load_data()
    
    # Drop Cluster column if it exists (should not be used for training)
    X = df.drop(columns=["Cluster"], errors="ignore")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    dbscan = DBSCAN(eps=3.5, min_samples=4)
    dbscan.fit(X_scaled)
    
    return dbscan, scaler, X

# Load model and scaler
dbscan, scaler, X = train_model()
df = load_data()

# Main content
st.markdown('<div class="wine-icon" style="text-align: center;">🍷</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">Wine Clustering Predictor</h1>', unsafe_allow_html=True)

# Create input section
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<p class="section-header">📊 Enter Wine Properties</p>', unsafe_allow_html=True)

# Input fields for all 13 features in a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    alcohol = st.number_input("Alcohol", min_value=10.0, max_value=15.0, value=13.0, step=0.1)
    malic_acid = st.number_input("Malic Acid", min_value=0.5, max_value=6.0, value=1.7, step=0.1)
    ash = st.number_input("Ash", min_value=1.0, max_value=3.5, value=2.4, step=0.1)
    ash_alcanity = st.number_input("Ash Alcanity", min_value=5.0, max_value=30.0, value=16.0, step=0.5)
    magnesium = st.number_input("Magnesium", min_value=50.0, max_value=180.0, value=110.0, step=1.0)

with col2:
    total_phenols = st.number_input("Total Phenols", min_value=1.0, max_value=4.0, value=2.7, step=0.1)
    flavanoids = st.number_input("Flavanoids", min_value=0.0, max_value=5.0, value=2.9, step=0.1)
    nonflavanoid_phenols = st.number_input("Nonflavanoid Phenols", min_value=0.0, max_value=1.0, value=0.28, step=0.01)
    proanthocyanins = st.number_input("Proanthocyanins", min_value=0.3, max_value=4.0, value=1.8, step=0.1)
    color_intensity = st.number_input("Color Intensity", min_value=1.0, max_value=14.0, value=5.2, step=0.1)

with col3:
    hue = st.number_input("Hue", min_value=0.5, max_value=2.0, value=1.05, step=0.01)
    od280 = st.number_input("OD280", min_value=1.5, max_value=4.0, value=3.2, step=0.1)
    proline = st.number_input("Proline", min_value=200.0, max_value=1700.0, value=1000.0, step=10.0)

st.markdown('</div>', unsafe_allow_html=True)

# Prediction button
st.markdown('<div style="text-align: center; margin: 30px 0;">', unsafe_allow_html=True)
if st.button("🔮 Predict Cluster"):
    # Create input array
    input_data = np.array([[alcohol, malic_acid, ash, ash_alcanity, magnesium, 
                           total_phenols, flavanoids, nonflavanoid_phenols, 
                           proanthocyanins, color_intensity, hue, od280, proline]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Get training data (features only)
    X_scaled = scaler.transform(X.values)
    
    # Calculate distances to all training points
    distances = pairwise_distances(input_scaled, X_scaled)
    
    # DBSCAN parameters
    eps = 3.5
    min_samples = 4
    
    # Find minimum distance to any training point
    min_distance = np.min(distances[0])
    nearest_idx = np.argmin(distances[0])
    
    # If nearest is noise → noise
    if dbscan.labels_[nearest_idx] == -1:
        cluster = -1
    else:
        core_samples = dbscan.core_sample_indices_
        neighbors = np.sum(distances[0] <= eps)
        
        if neighbors >= min_samples:
            core_distances = distances[0][core_samples]
            
            if np.any(core_distances <= eps):
                nearest_core = core_samples[np.argmin(core_distances)]
                cluster = dbscan.labels_[nearest_core]
            else:
                cluster = -1
        else:
            cluster = -1
    
    # Display result
    st.markdown('---')
    
    if cluster == -1:
        st.success(f"Cluster: Noise (No Cluster)")
    else:
        st.success(f"Wine Cluster: {cluster}")



