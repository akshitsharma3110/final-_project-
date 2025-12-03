import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

def load_and_combine_data():
    """Load and combine all disaster text data"""
    print("Loading disaster text data...")
    
    # Load main dataset
    df1 = pd.read_csv('disasters/All the Texts.txt')
    print(f"Main dataset: {df1.shape[0]} samples")
    
    # Load extended dataset
    df2 = pd.read_csv('disasters/Extended_Texts.txt')
    print(f"Extended dataset: {df2.shape[0]} samples")
    
    # Combine datasets
    df_combined = pd.concat([df1, df2], ignore_index=True)
    
    # Remove None/null labels for disaster classification
    df_combined = df_combined[df_combined['label'] != 'None'].dropna()
    
    print(f"Combined dataset: {df_combined.shape[0]} samples")
    print(f"\nClass distribution:\n{df_combined['label'].value_counts()}")
    
    return df_combined

def preprocess_data(df):
    """Preprocess the text data"""
    print("\nPreprocessing data...")
    
    # Clean text data
    df['text'] = df['text'].str.lower().str.strip()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['text'])
    print(f"After removing duplicates: {df.shape[0]} samples")
    
    return df

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and create ensemble"""
    print("\nTraining models...")
    
    # Basic TF-IDF vectorizer
    vectorizer_basic = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_basic = vectorizer_basic.fit_transform(X_train)
    X_test_basic = vectorizer_basic.transform(X_test)
    
    # Basic Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_basic, y_train)
    rf_pred = rf_model.predict(X_test_basic)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
    
    # Advanced TF-IDF vectorizer with n-grams
    vectorizer_advanced = TfidfVectorizer(
        max_features=2000, 
        stop_words='english', 
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_train_advanced = vectorizer_advanced.fit_transform(X_train)
    X_test_advanced = vectorizer_advanced.transform(X_test)
    
    # Ensemble model with multiple classifiers
    ensemble = VotingClassifier([
        ('svm', SVC(kernel='linear', probability=True, random_state=42, C=1.0)),
        ('nb', MultinomialNB(alpha=0.1)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000, C=1.0)),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10))
    ], voting='soft')
    
    ensemble.fit(X_train_advanced, y_train)
    ensemble_pred = ensemble.predict(X_test_advanced)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    print(f"Ensemble Accuracy: {ensemble_accuracy:.3f}")
    print(f"\nEnsemble Classification Report:\n{classification_report(y_test, ensemble_pred)}")
    
    # Save the best model
    if ensemble_accuracy > rf_accuracy:
        print("\nSaving ensemble model as the best model...")
        joblib.dump(ensemble, 'disaster_model_FINAL.pkl')
        joblib.dump(vectorizer_advanced, 'vectorizer_FINAL.pkl')
        return ensemble, vectorizer_advanced, ensemble_accuracy
    else:
        print("\nSaving Random Forest model as the best model...")
        joblib.dump(rf_model, 'disaster_model_FINAL.pkl')
        joblib.dump(vectorizer_basic, 'vectorizer_FINAL.pkl')
        return rf_model, vectorizer_basic, rf_accuracy

def test_model_predictions(model, vectorizer):
    """Test the trained model with sample predictions"""
    print("\nTesting model predictions...")
    
    test_texts = [
        "The building collapsed after strong tremors",
        "Water levels are rising rapidly in the city",
        "Smoke is spreading across the forest",
        "Strong winds are approaching the coast",
        "Mudslide blocked the highway",
        "No rain for months, crops are dying",
        "Massive waves hit the shore",
        "Car accident on the highway",
        "Volcanic ash covers the city",
        "Tornado touched down in the suburb"
    ]
    
    for text in test_texts:
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]
        probability = model.predict_proba(text_vec)[0].max()
        print(f"'{text}' -> {prediction} (confidence: {probability:.3f})")

def main():
    """Main training function"""
    print("=== Disaster Text Classification Model Training ===")
    
    # Load and combine data
    df = load_and_combine_data()
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Prepare features and labels
    X = df['text']
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train models
    best_model, best_vectorizer, best_accuracy = train_models(X_train, X_test, y_train, y_test)
    
    # Test predictions
    test_model_predictions(best_model, best_vectorizer)
    
    print(f"\n=== Training Complete ===")
    print(f"Best model accuracy: {best_accuracy:.3f}")
    print("Model files saved:")
    print("- disaster_model_FINAL.pkl")
    print("- vectorizer_FINAL.pkl")
    
    # Also save backup models
    print("\nBackup models also available:")
    if os.path.exists('disaster_model_improved.pkl'):
        print("- disaster_model_improved.pkl")
        print("- vectorizer_improved.pkl")

if __name__ == "__main__":
    main()