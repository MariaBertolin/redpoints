from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import pandas as pd

ASSET_DISCARDED = 4  # ejemplo, según tu dataset
STAGE2_NEGATIVE_LABEL = 7  # INFRINGEMENT_DISCARDED
# STAGE2_POSITIVE_LABELS = [5, 8, 9, 10, 15, 16] # Sense infrigement on review, segons pdf

def save_model(model, vectorizer, model_path, vectorizer_path):
    """Save the trained model and vectorizer to disk."""
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

def load_model(model_path, vectorizer_path):
    """Load the trained model and vectorizer from disk."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print(f"Model loaded from {model_path}")
    print(f"Vectorizer loaded from {vectorizer_path}")
    return model, vectorizer

def binarize_stage1(labels):
    labels = pd.Series(labels)
    return (labels != ASSET_DISCARDED).astype(int)

def binarize_stage2(labels):
    labels = pd.Series(labels)
    return (labels != STAGE2_NEGATIVE_LABEL).astype(int)

def build_vectorizer():
    """
    Create the shared TF-IDF vectorizer used across the pipeline.
    """
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5)
    )