"""
Multi-Model Agricultural Intelligence System
Models Module: Decision Tree, KNN Clustering, Linear Regression
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score
)
import joblib
from preprocessing import load_and_preprocess_data

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def train_decision_tree(X_train, X_test, y_train, y_test, feature_names):
    """
    Train Decision Tree Classifier for Crop Recommendation
    Returns: trained model, metrics dictionary, feature importance
    """
    print("\n" + "="*60)
    print("TRAINING DECISION TREE CLASSIFIER")
    print("="*60)
    
    # Initialize and train
    dt_model = DecisionTreeClassifier(
        max_depth=30,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = dt_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }
    
    print(f"\n✓ Accuracy:  {accuracy:.4f}")
    print(f"✓ Precision: {precision:.4f}")
    print(f"✓ Recall:    {recall:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n Feature Importance:")
    print(feature_importance.to_string(index=False))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(dt_model, 'models/decision_tree.joblib')
    print("\n✓ Model saved to 'models/decision_tree.joblib'")
    
    # Visualizations
    os.makedirs('results', exist_ok=True)
    
    # 1. Feature Importance Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Decision Tree - Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/dt_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Feature importance plot saved to 'results/dt_feature_importance.png'")
    plt.close()
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Decision Tree - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/dt_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved to 'results/dt_confusion_matrix.png'")
    plt.close()
    
    # 3. Decision Tree Visualization (simplified)
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, 
              feature_names=feature_names,
              filled=True,
              rounded=True,
              max_depth=3,  # Show only top 3 levels for clarity
              fontsize=10)
    plt.title('Decision Tree Structure (Top 3 Levels)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/dt_tree_structure.png', dpi=300, bbox_inches='tight')
    print("✓ Tree structure saved to 'results/dt_tree_structure.png'")
    plt.close()
    
    return dt_model, metrics, feature_importance


def train_knn_clustering(X_scaled, n_clusters=5):
    """
    Train KNN Clustering for Soil Profile Segmentation
    Returns: trained model, metrics dictionary
    """
    print("\n" + "="*60)
    print("TRAINING K-MEANS CLUSTERING MODEL")
    print("="*60)
    
    # Train KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    
    metrics = {
        'silhouette_score': silhouette_avg,
        'n_clusters': n_clusters
    }
    
    print(f"\n✓ Number of Clusters: {n_clusters}")
    print(f"✓ Silhouette Score: {silhouette_avg:.4f}")
    print("  (Score closer to 1 indicates better-defined clusters)")
    
    # Cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("\n Cluster Distribution:")
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
    
    # Save model
    joblib.dump(kmeans, 'models/kmeans_clustering.joblib')
    print("\n✓ Model saved to 'models/kmeans_clustering.joblib'")
    
    # Visualizations
    # 1. Cluster Distribution Bar Chart
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts, color='coral', edgecolor='black')
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('K-Means Clustering - Cluster Distribution', fontsize=14, fontweight='bold')
    plt.xticks(unique)
    for i, count in enumerate(counts):
        plt.text(i, count + 5, str(count), ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('results/kmeans_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Cluster distribution saved to 'results/kmeans_distribution.png'")
    plt.close()
    
    # 2. 2D Scatter Plot (using first 2 principal features)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                         c=cluster_labels, cmap='viridis', 
                         alpha=0.6, edgecolors='black', s=50)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', marker='X', s=300, edgecolors='black', 
                linewidths=2, label='Centroids')
    plt.colorbar(scatter, label='Cluster ID')
    plt.xlabel('Feature 1 (Scaled N)', fontsize=12)
    plt.ylabel('Feature 2 (Scaled P)', fontsize=12)
    plt.title('K-Means Clustering - 2D Visualization', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/kmeans_scatter.png', dpi=300, bbox_inches='tight')
    print("✓ Cluster scatter plot saved to 'results/kmeans_scatter.png'")
    plt.close()
    
    # 3. Elbow Method for optimal K
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, marker='o', linewidth=2, markersize=8, color='teal')
    plt.axvline(x=n_clusters, color='red', linestyle='--', label=f'Selected K={n_clusters}')
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/kmeans_elbow.png', dpi=300, bbox_inches='tight')
    print("✓ Elbow plot saved to 'results/kmeans_elbow.png'")
    plt.close()
    
    return kmeans, metrics, cluster_labels


def train_linear_regression(X_train, X_test, y_train, y_test, feature_names):
    """
    Train Linear Regression for Crop Yield Prediction
    Returns: trained model, metrics dictionary
    """
    print("\n" + "="*60)
    print("TRAINING LINEAR REGRESSION MODEL")
    print("="*60)
    
    # Train model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = lr_model.predict(X_train)
    y_pred_test = lr_model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2
    }
    
    print(f"\n✓ RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"✓ MAE (Mean Absolute Error):      {mae:.4f}")
    print(f"✓ R² Score:                        {r2:.4f}")
    print(f"  (R² closer to 1 indicates better model fit)")
    
    # Coefficients
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': lr_model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\n Feature Coefficients:")
    print(coef_df.to_string(index=False))
    print(f"\nIntercept: {lr_model.intercept_:.4f}")
    
    # Save model
    joblib.dump(lr_model, 'models/linear_regression.joblib')
    print("\n✓ Model saved to 'models/linear_regression.joblib'")
    
    # Visualizations
    # 1. Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='black', s=50, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Yield', fontsize=12)
    plt.ylabel('Predicted Yield', fontsize=12)
    plt.title('Linear Regression - Actual vs Predicted Yield', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/lr_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    print("✓ Actual vs Predicted plot saved to 'results/lr_actual_vs_predicted.png'")
    plt.close()
    
    # 2. Residual Plot
    residuals = y_test - y_pred_test
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_test, residuals, alpha=0.6, edgecolors='black', s=50, color='purple')
    plt.axhline(y=0, color='red', linestyle='--', lw=2)
    plt.xlabel('Predicted Yield', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.title('Linear Regression - Residual Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/lr_residuals.png', dpi=300, bbox_inches='tight')
    print("✓ Residual plot saved to 'results/lr_residuals.png'")
    plt.close()
    
    # 3. Residual Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, edgecolor='black', color='skyblue', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error')
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Linear Regression - Residual Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/lr_residual_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Residual distribution saved to 'results/lr_residual_distribution.png'")
    plt.close()
    
    # 4. Coefficient Plot
    plt.figure(figsize=(10, 6))
    colors = ['green' if x > 0 else 'red' for x in coef_df['coefficient']]
    plt.barh(coef_df['feature'], coef_df['coefficient'], color=colors, edgecolor='black')
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Linear Regression - Feature Coefficients', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', lw=1)
    plt.tight_layout()
    plt.savefig('results/lr_coefficients.png', dpi=300, bbox_inches='tight')
    print("✓ Coefficient plot saved to 'results/lr_coefficients.png'")
    plt.close()
    
    return lr_model, metrics


def main():
    """
    Main training pipeline using DUAL datasets
    """
    print("\n" + "="*60)
    print("AGRICULTURAL INTELLIGENCE SYSTEM - MODEL TRAINING")
    print("DUAL DATASET APPROACH")
    print("="*60)
    
    # Define paths
    classification_path = os.path.join('data', 'Crop_recommendation.csv')
    yield_path = os.path.join('data', 'dataset2', 'yield_df.csv')
    
    # Check if datasets exist
    if not os.path.exists(classification_path):
        print(f"\n ERROR: Classification dataset not found at {classification_path}")
        return
    
    if not os.path.exists(yield_path):
        print(f"\n ERROR: Yield dataset not found at {yield_path}")
        return
    
    # Load and preprocess both datasets
    (X_train_clf, X_test_clf, y_train_clf, y_test_clf,
     X_train_reg, X_test_reg, y_train_reg, y_test_reg,
     X_scaled_all) = load_and_preprocess_data(classification_path, yield_path)
    
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # Train Decision Tree (using classification data)
    print("\n" + "="*60)
    print("USING CROP RECOMMENDATION DATASET")
    print("="*60)
    dt_model, dt_metrics, dt_importance = train_decision_tree(
        X_train_clf, X_test_clf, y_train_clf, y_test_clf, feature_names
    )
    
    # Train K-Means (using classification data)
    kmeans_model, kmeans_metrics, cluster_labels = train_knn_clustering(
        X_scaled_all.values, n_clusters=5
    )
    
    # Train Linear Regression (using REAL yield data)
    print("\n" + "="*60)
    print("USING REAL YIELD DATASET")
    print("="*60)
    lr_model, lr_metrics = train_linear_regression(
        X_train_reg, X_test_reg, y_train_reg, y_test_reg, feature_names
    )
    
    # Summary Report
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    print("\n Decision Tree Classifier (Crop Recommendation Dataset):")
    print(f"  • Accuracy:  {dt_metrics['accuracy']:.4f}")
    print(f"  • Precision: {dt_metrics['precision']:.4f}")
    print(f"  • Recall:    {dt_metrics['recall']:.4f}")
    
    print("\n K-Means Clustering (Crop Recommendation Dataset):")
    print(f"  • Silhouette Score: {kmeans_metrics['silhouette_score']:.4f}")
    print(f"  • Number of Clusters: {kmeans_metrics['n_clusters']}")
    
    print("\n Linear Regression (REAL Yield Dataset):")
    print(f"  • RMSE: {lr_metrics['rmse']:.4f} hg/ha")
    print(f"  • MAE:  {lr_metrics['mae']:.4f} hg/ha")
    print(f"  • R²:   {lr_metrics['r2_score']:.4f}")
    
    print("\n" + "="*60)
    print("All models trained and saved successfully!")
    print("✓ Decision Tree & K-Means: Crop Recommendation data")
    print("✓ Linear Regression: Real Yield data")
    print("Check the 'models/' and 'results/' directories for outputs.")
    print("="*60 + "\n")
    
    # Save metrics to file
    with open('results/metrics_summary.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("AGRICULTURAL INTELLIGENCE SYSTEM - METRICS SUMMARY\n")
        f.write("DUAL DATASET APPROACH\n")
        f.write("="*60 + "\n\n")
        f.write("Decision Tree Classifier (Crop Recommendation Dataset):\n")
        f.write(f"  Accuracy:  {dt_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {dt_metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {dt_metrics['recall']:.4f}\n\n")
        f.write("K-Means Clustering (Crop Recommendation Dataset):\n")
        f.write(f"  Silhouette Score: {kmeans_metrics['silhouette_score']:.4f}\n")
        f.write(f"  Number of Clusters: {kmeans_metrics['n_clusters']}\n\n")
        f.write("Linear Regression (REAL Yield Dataset):\n")
        f.write(f"  RMSE: {lr_metrics['rmse']:.4f} hg/ha\n")
        f.write(f"  MAE:  {lr_metrics['mae']:.4f} hg/ha\n")
        f.write(f"  R²:   {lr_metrics['r2_score']:.4f}\n\n")
        f.write("Dataset Information:\n")
        f.write(f"  Classification: Crop_recommendation.csv (2,200 samples)\n")
        f.write(f"  Regression: yield_df.csv (real yield data)\n")
    
    print("✓ Metrics summary saved to 'results/metrics_summary.txt'\n")


if __name__ == "__main__":
    main()
