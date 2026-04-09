"""
TruthLens Streamlit Dashboard
NLP-Based News Authenticity Detection System
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import re
import string
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Configure page
st.set_page_config(
    page_title="TruthLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load NLTK data
@st.cache_data
def load_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

load_nltk_data()

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('model/truthlens_model.joblib')
    except FileNotFoundError:
        st.error("❌ Model not found! Please run `python train_model.py` first.")
        st.stop()

# Text preprocessing function (same as training)
def preprocess_text(text):
    if pd.isna(text) or text == "":
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

# Load data for analytics
@st.cache_data
def load_data_for_analytics():
    try:
        fake_df = pd.read_csv('data/Fake.csv')
        true_df = pd.read_csv('data/True.csv')
        
        fake_df['label'] = 'Fake'
        true_df['label'] = 'Real'
        fake_df['text'] = fake_df['title'] + ' ' + fake_df['text'].fillna('')
        true_df['text'] = true_df['title'] + ' ' + true_df['text'].fillna('')
        
        combined_df = pd.concat([fake_df[['text', 'label']], true_df[['text', 'label']]], ignore_index=True)
        combined_df['text_length'] = combined_df['text'].str.len()
        combined_df['cleaned_text'] = combined_df['text'].apply(preprocess_text)
        
        return combined_df
    except FileNotFoundError:
        return None

# Page configuration
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.title("🔍 TruthLens")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Data Analytics", "📰 News Detection"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.info("👈 **Place your datasets in `data/` folder and run `python train_model.py` first!")

# Load data and model
df = load_data_for_analytics()
model = load_model()

# Home Page
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">TruthLens</h1>', unsafe_allow_html=True)
    st.markdown('<h2>NLP-Based News Authenticity Detection System</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Purpose</h3>
            <p>Detect fake news using advanced NLP techniques</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 Technology</h3>
            <p>TF-IDF + Logistic Regression</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Features</h3>
            <p>Real-time detection with confidence scores</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🚀 How to Use")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Setup**
        ```bash
        pip install -r requirements.txt
        python train_model.py
        ```
        
        **2. Place datasets**
        - Put `Fake.csv` and `True.csv` in `data/` folder
        """)
    
    with col2:
        st.markdown("""
        **3. Launch Dashboard**
        ```bash
        streamlit run app.py
        ```
        
        **4. Detect News**
        - Go to "News Detection"
        - Paste article text
        - Get instant results!
        """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Technical Details")
    st.info("""
    - **Preprocessing**: Tokenization, stopword removal, punctuation cleaning
    - **Features**: TF-IDF Vectorization (unigrams + bigrams)
    - **Model**: Logistic Regression (high accuracy, fast inference)
    - **Evaluation**: Train/Test split with stratified sampling
    """)

# Data Analytics Page
elif page == "📊 Data Analytics":
    if df is None:
        st.error("❌ Datasets not found in `data/` folder!")
        st.stop()
    
    st.markdown('<h2 style="color: #1f77b4;">📊 Data Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Articles", len(df))
    with col2:
        fake_count = len(df[df['label'] == 'Fake'])
        st.metric("Fake News", fake_count, delta=f"{fake_count/len(df)*100:.1f}%")
    with col3:
        real_count = len(df[df['label'] == 'Real'])
        st.metric("Real News", real_count, delta=f"{real_count/len(df)*100:.1f}%")
    with col4:
        avg_length = df['text_length'].mean()
        st.metric("Avg Length", f"{avg_length:.0f} chars")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution pie chart
        fig_pie = px.pie(
            df, names='label', 
            title="Fake vs Real News Distribution",
            color_discrete_map={'Fake': '#ff6b6b', 'Real': '#4ecdc4'}
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Length distribution
        fig_hist = px.histogram(
            df, x='text_length', color='label',
            nbins=50, opacity=0.7,
            title="Article Length Distribution",
            color_discrete_map={'Fake': '#ff6b6b', 'Real': '#4ecdc4'}
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Word clouds
    st.subheader("☁️ Word Clouds")
    col1, col2 = st.columns(2)
    
    with col1:
        fake_text = ' '.join(df[df['label'] == 'Fake']['cleaned_text'])
        wc_fake = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(fake_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc_fake, interpolation='bilinear')
        plt.axis('off')
        plt.title("Most Common Words in Fake News", fontsize=16, color='red')
        st.pyplot(plt.gcf())
        plt.close()
    
    with col2:
        real_text = ' '.join(df[df['label'] == 'Real']['cleaned_text'])
        wc_real = WordCloud(width=400, height=300, background_color='white', colormap='Blues').generate(real_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc_real, interpolation='bilinear')
        plt.axis('off')
        plt.title("Most Common Words in Real News", fontsize=16, color='blue')
        st.pyplot(plt.gcf())
        plt.close()

# News Detection Page
elif page == "📰 News Detection":
    st.markdown('<h2 style="color: #1f77b4;">📰 Real-time News Detection</h2>', unsafe_allow_html=True)
    
    # Input textarea
    news_text = st.text_area(
        "Paste your news article here:",
        height=200,
        placeholder="Enter the news article text, title, or content..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    if st.button("🔍 Analyze News", type="primary"):
        if news_text:
            # Preprocess
            cleaned_text = preprocess_text(news_text)
            
            if len(cleaned_text) == 0:
                st.warning("⚠️ No meaningful text detected. Please try another article.")
            else:
                # Predict
                prediction = model.predict([cleaned_text])[0]
                probability = model.predict_proba([cleaned_text])[0]
                
                label = "🟢 REAL" if prediction == 1 else "🔴 FAKE"
                confidence = max(probability) * 100
                
                # Display results
                if prediction == 1:
                    col1.metric("Prediction", label, delta=f"{confidence:.1f}% confidence")
                    col1.success(f"**{label}**")
                    col2.metric("Real Probability", f"{probability[1]*100:.1f}%")
                    col3.metric("Fake Probability", f"{probability[0]*100:.1f}%")
                else:
                    col1.metric("Prediction", label, delta=f"{confidence:.1f}% confidence")
                    col1.error(f"**{label}**")
                    col2.metric("Fake Probability", f"{probability[0]*100:.1f}%")
                    col3.metric("Real Probability", f"{probability[1]*100:.1f}%")
                
                st.markdown("---")
                st.subheader("📝 Analysis Details")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Processed text length**: {len(cleaned_text)} words")
                    st.info(f"**Cleaned sample**: {cleaned_text[:200]}...")
                
                with col2:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=confidence,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Confidence"},
                        delta={'reference': 90},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90}
                        }))
                    st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.warning("⚠️ Please enter some text to analyze!")

## Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>"
    "TruthLens © 2024 | NLP-Powered Fake News Detection | "
    "<a href='https://github.com/yourusername/truthlens' target='_blank'>Source Code</a>"
    "</p>", 
    unsafe_allow_html=True
)
