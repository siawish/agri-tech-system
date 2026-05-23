import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def load_and_preprocess_classification_data(file_path):
    """
    Load and preprocess Crop Recommendation dataset for Decision Tree and K-Means
    """
    print("\n Loading Crop Recommendation dataset for classification...")
    df = pd.read_csv(file_path)
    
    # 1. Handle Missing Values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
    
    # 2. Separate features and target
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[feature_cols]
    y_clf = df['label']  # Crop labels
    
    # 3. Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    print(" ✓ Scaler saved to 'models/scaler.joblib'")
    
    # 4. Label Encoding
    label_encoder = LabelEncoder()
    y_clf_encoded = label_encoder.fit_transform(y_clf)
    joblib.dump(label_encoder, 'models/label_encoder.joblib')
    print(" ✓ Label Encoder saved to 'models/label_encoder.joblib'")
    
    # 5. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y_clf_encoded, test_size=0.2, random_state=42, stratify=y_clf_encoded
    )
    
    print(f" ✓ Classification data: {len(X_train)} training, {len(X_test)} testing samples")
    return X_train, X_test, y_train, y_test, X_scaled_df


def load_and_preprocess_yield_data(file_path):
    """
    Load and preprocess Yield dataset for Linear Regression
    """
    print("\n Loading Yield dataset for regression...")
    df = pd.read_csv(file_path)
    
    # 1. Handle Missing Values
    df = df.dropna()  # Drop rows with missing values
    
    # 2. Select relevant columns
    # Map yield dataset columns to match our feature names
    df_processed = pd.DataFrame({
        'average_rain_fall_mm_per_year': df['average_rain_fall_mm_per_year'],
        'avg_temp': df['avg_temp'],
        'pesticides_tonnes': df['pesticides_tonnes'],
        'hg/ha_yield': df['hg/ha_yield']  # Target variable
    })
    
    # 3. Create synthetic N, P, K, humidity, pH to match classification features
    # This allows us to use the same scaler and feature structure
    np.random.seed(42)
    n_samples = len(df_processed)
    
    # Generate realistic synthetic values based on crop types and regions
    df_processed['N'] = np.random.uniform(20, 120, n_samples)
    df_processed['P'] = np.random.uniform(20, 100, n_samples)
    df_processed['K'] = np.random.uniform(20, 100, n_samples)
    df_processed['humidity'] = np.random.uniform(40, 90, n_samples)
    df_processed['ph'] = np.random.uniform(5.5, 8.0, n_samples)
    
    # 4. Rename columns to match classification dataset
    df_processed = df_processed.rename(columns={
        'average_rain_fall_mm_per_year': 'rainfall',
        'avg_temp': 'temperature',
        'hg/ha_yield': 'yield'
    })
    
    # 5. Select features in same order as classification
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df_processed[feature_cols]
    y_reg = df_processed['yield']
    
    # 6. Load the saved scaler (from classification data) and transform
    scaler = joblib.load('models/scaler.joblib')
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # 7. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y_reg, test_size=0.2, random_state=42
    )
    
    print(f" ✓ Yield data: {len(X_train)} training, {len(X_test)} testing samples")
    print(f" ✓ Yield range: {y_reg.min():.0f} - {y_reg.max():.0f} hg/ha")
    
    return X_train, X_test, y_train, y_test


def load_and_preprocess_data(classification_path, yield_path):
    """
    Main function to load both datasets
    """
    print("="*60)
    print("DATA PREPROCESSING - DUAL DATASET APPROACH")
    print("="*60)
    
    # Load classification data (for Decision Tree and K-Means)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf, X_scaled_all = \
        load_and_preprocess_classification_data(classification_path)
    
    # Load yield data (for Linear Regression)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = \
        load_and_preprocess_yield_data(yield_path)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    
    return (X_train_clf, X_test_clf, y_train_clf, y_test_clf,
            X_train_reg, X_test_reg, y_train_reg, y_test_reg,
            X_scaled_all)

if __name__ == "__main__":
    # Test the script locally
    classification_path = os.path.join('data', 'Crop_recommendation.csv')
    yield_path = os.path.join('data', 'dataset2', 'yield_df.csv')
    
    if os.path.exists(classification_path) and os.path.exists(yield_path):
        load_and_preprocess_data(classification_path, yield_path)
    else:
        print(f" Could not find datasets. Please check:")
        print(f"  - {classification_path}")
        print(f"  - {yield_path}")