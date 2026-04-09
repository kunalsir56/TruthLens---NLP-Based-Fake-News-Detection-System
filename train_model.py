"""
TruthLens Training Script
Trains and saves the NLP-based fake news detection model.
Run: python train_model.py
"""

import pandas as pd
import numpy as np
import nltk
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import os
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Create directories if they don't exist
os.makedirs('model', exist_ok=True)

def load_data():
    """Load and combine fake and real news datasets."""
    print("Loading datasets...")
    
    # Load datasets
    fake_df = pd.read_csv('data/Fake.csv')
    true_df = pd.read_csv('data/True.csv')
    
    # Add labels (0 = Fake, 1 = Real)
    fake_df['label'] = 0
    true_df['label'] = 1
    
    # Combine datasets (using 'title' + 'text' for better context)
    fake_df['text'] = fake_df['title'] + ' ' + fake_df['text'].fillna('')
    true_df['text'] = true_df['title'] + ' ' + true_df['text'].fillna('')
    
    combined_df = pd.concat([fake_df[['text', 'label']], true_df[['text', 'label']]], ignore_index=True)
    
    print(f"Loaded {len(fake_df)} fake and {len(true_df)} real articles")
    print(f"Total dataset size: {len(combined_df)} articles")
    return combined_df

def preprocess_text(text):
    """Clean and preprocess text data."""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

def train_model(df):
    """Train the fake news detection model."""
    print("\nPreprocessing text...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
    
    # Split data
    X = df['cleaned_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Train model
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    return pipeline, accuracy

def save_model(pipeline):
    """Save trained model."""
    model_path = 'model/truthlens_model.joblib'
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")

def main():
    """Main training function."""
    print("🚀 TruthLens: Training Fake News Detection Model")
    print("=" * 50)
    
    # Load data
    df = load_data()
    
    # Train model
    pipeline, accuracy = train_model(df)
    
    # Save model
    save_model(pipeline)
    
    print("\n✅ Training completed successfully!")
    print("Run: streamlit run app.py to launch the dashboard")

if __name__ == "__main__":
    main()
