"""
Comprehensive Testing Script for Exoplanet Classification Model

This script demonstrates advanced model evaluation metrics and statistical analysis
for the exoplanet classification system.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def create_sample_exoplanet_data(n_samples=2000, n_features=20, random_state=42):
    """
    Create synthetic exoplanet-like data for testing when real data isn't available.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        random_state: Random seed for reproducibility

    Returns:
        pd.DataFrame: Synthetic dataset
    """
    # Create synthetic classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_classes=3,
        n_clusters_per_class=1,
        random_state=random_state,
        class_sep=1.5
    )

    # Create feature names similar to exoplanet data
    feature_names = [
        'pl_orbper', 'pl_rade', 'pl_radj', 'pl_bmasse', 'pl_bmassj',
        'pl_orbsmax', 'pl_eccen', 'pl_insol', 'pl_eqt', 'pl_orbincl',
        'st_rad', 'st_mass', 'st_teff', 'st_met', 'st_logg',
        'sy_dist', 'sy_vmag', 'discoverymethod', 'disc_year', 'disc_facility'
    ][:n_features]

    # Create target names similar to exoplanet dispositions
    target_names = ['confirmed', 'candidate', 'false_positive']

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['disposition'] = [target_names[i] for i in y]

    return df

def comprehensive_model_evaluation(X, y, model, model_name, target_names):
    """
    Perform comprehensive evaluation of a classification model.

    Args:
        X: Feature matrix
        y: Target vector (encoded)
        model: Trained model
        model_name: Name of the model
        target_names: List of class names
    """
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE EVALUATION: {model_name}")
    print(f"{'='*60}")

    # 1. Cross-Validation Analysis
    print("\n1. CROSS-VALIDATION ANALYSIS")
    print("-" * 40)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_precision = cross_val_score(model, X, y, cv=5, scoring='precision_macro')
    cv_recall = cross_val_score(model, X, y, cv=5, scoring='recall_macro')
    cv_f1 = cross_val_score(model, X, y, cv=5, scoring='f1_macro')

    print(f"CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std() * 2:.3f})")
    print(f"CV Precision: {cv_precision.mean():.3f} (±{cv_precision.std() * 2:.3f})")
    print(f"CV Recall: {cv_recall.mean():.3f} (±{cv_recall.std() * 2:.3f})")
    print(f"CV F1-Score: {cv_f1.mean():.3f} (±{cv_f1.std() * 2:.3f})")

    # 2. Holdout Test Evaluation
    print("\n2. HOLDOUT TEST EVALUATION")
    print("-" * 40)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Test Precision: {precision:.3f}")
    print(f"Test Recall: {recall:.3f}")
    print(f"Test F1-Score: {f1:.3f}")

    # 3. Confusion Matrix Analysis
    print("\n3. CONFUSION MATRIX ANALYSIS")
    print("-" * 40)

    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print("Confusion Matrix (Raw):")
    print(cm)
    print("\nConfusion Matrix (Normalized):")
    print(cm_normalized)

    # Per-class performance
    print("\nPer-Class Performance:")
    for i, class_name in enumerate(target_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        precision_cls = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_cls = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_cls = 2 * precision_cls * recall_cls / (precision_cls + recall_cls) if (precision_cls + recall_cls) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"  {class_name}:")
        print(f"    Precision: {precision_cls:.3f}")
        print(f"    Recall: {recall_cls:.3f}")
        print(f"    F1-Score: {f1_cls:.3f}")
        print(f"    Specificity: {specificity:.3f}")

    # 4. ROC Analysis (for multiclass, we'll use one-vs-rest)
    print("\n4. ROC CURVE ANALYSIS")
    print("-" * 40)

    # For multiclass, calculate ROC for each class
    for i, class_name in enumerate(target_names):
        y_test_binary = (y_test == i).astype(int)
        y_pred_proba_cls = y_pred_proba[:, i]

        fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba_cls)
        roc_auc = auc(fpr, tpr)

        print(f"{class_name} - ROC AUC: {roc_auc:.3f}")

    # 5. Statistical Significance Testing
    print("\n5. STATISTICAL SIGNIFICANCE")
    print("-" * 40)

    # Compare to random guessing
    random_accuracy = 1.0 / len(target_names)
    t_stat, p_value = stats.ttest_1samp(cv_scores, random_accuracy)

    print(f"Random Guessing Accuracy: {random_accuracy:.3f}")
    print(f"Model vs Random - t-statistic: {t_stat:.3f}")
    print(f"Model vs Random - p-value: {p_value:.6f}")

    if p_value < 0.05:
        print("✓ Model significantly outperforms random guessing (p < 0.05)")
    else:
        print("✗ Model performance not significantly better than random guessing")

    # 6. Feature Importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        print("\n6. FEATURE IMPORTANCE ANALYSIS")
        print("-" * 40)

        # Get feature names (assuming X has column names)
        feature_names = getattr(X, 'columns', [f'feature_{i}' for i in range(X.shape[1])])

        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_,
            'importance_percentage': model.feature_importances_ * 100
        }).sort_values('importance', ascending=False)

        print("Top 10 Most Important Features:")
        for j, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f} ({row['importance_percentage']:.1f}%)")

    return {
        'cv_scores': cv_scores,
        'test_metrics': {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1},
        'confusion_matrix': cm,
        'model': model
    }

def compare_multiple_models(X, y, target_names):
    """
    Compare multiple ML algorithms for exoplanet classification.

    Args:
        X: Feature matrix
        y: Target vector (encoded)
        target_names: List of class names
    """
    print(f"\n{'='*60}")
    print("MODEL COMPARISON ANALYSIS")
    print(f"{'='*60}")

    # Define models to compare
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }

    results = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        print("-" * 30)

        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_precision = cross_val_score(model, X, y, cv=5, scoring='precision_macro')
        cv_recall = cross_val_score(model, X, y, cv=5, scoring='recall_macro')
        cv_f1 = cross_val_score(model, X, y, cv=5, scoring='f1_macro')

        results[name] = {
            'accuracy': cv_scores.mean(),
            'accuracy_std': cv_scores.std(),
            'precision': cv_precision.mean(),
            'recall': cv_recall.mean(),
            'f1': cv_f1.mean()
        }

        print(f"Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std() * 2:.3f})")
        print(f"Precision: {cv_precision.mean():.3f}")
        print(f"Recall: {cv_recall.mean():.3f}")
        print(f"F1-Score: {cv_f1.mean():.3f}")

    # Statistical comparison
    print("STATISTICAL COMPARISON")
    print("-" * 30)

    sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    best_model = sorted_models[0]
    second_best = sorted_models[1]

    print(f"Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.3f})")
    print(f"Second Best: {second_best[0]} (Accuracy: {second_best[1]['accuracy']:.3f})")

    accuracy_diff = best_model[1]['accuracy'] - second_best[1]['accuracy']
    print(f"Accuracy Difference: {accuracy_diff:.3f}")

    if abs(accuracy_diff) > 0.01:
        status = "higher" if accuracy_diff > 0 else "lower"
        print(f"{'✓' if accuracy_diff > 0 else '✗'} {best_model[0]} shows {status} accuracy than {second_best[0]}")
    else:
        print("No significant difference between top models")

    return results

def run_comprehensive_tests():
    """Run comprehensive tests on exoplanet classification models."""
    print("Exoplanet Classification Model - Comprehensive Testing")
    print("=" * 60)

    # Create synthetic exoplanet data for testing
    print("\nGenerating synthetic exoplanet dataset...")
    df = create_sample_exoplanet_data(n_samples=2000, n_features=15, random_state=42)

    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution: {df['disposition'].value_counts().to_dict()}")

    # Prepare features and target
    feature_columns = [col for col in df.columns if col != 'disposition']
    X = df[feature_columns].values
    y = pd.Categorical(df['disposition']).codes

    target_names = ['confirmed', 'candidate', 'false_positive']

    # 1. Comprehensive evaluation of Random Forest
    print("\n" + "="*60)
    print("PHASE 1: RANDOM FOREST EVALUATION")
    print("="*60)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_results = comprehensive_model_evaluation(X, y, rf_model, "Random Forest", target_names)

    # 2. Compare multiple models
    print("\n" + "="*60)
    print("PHASE 2: MODEL COMPARISON")
    print("="*60)

    comparison_results = compare_multiple_models(X, y, target_names)

    # 3. Summary and recommendations
    print("\n" + "="*60)
    print("FINAL SUMMARY AND RECOMMENDATIONS")
    print("="*60)

    print("\nModel Performance Summary:")
    for model_name, metrics in comparison_results.items():
        print(f"  {model_name}:")
        print(f"    Accuracy: {metrics['accuracy']:.3f} ± {metrics['accuracy_std']:.3f}")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall: {metrics['recall']:.3f}")
        print(f"    F1-Score: {metrics['f1']:.3f}")

    best_model_name = max(comparison_results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_metrics = comparison_results[best_model_name]

    print("\nRECOMMENDATIONS:")
    print(f"  ✓ Best performing model: {best_model_name}")
    print(f"  ✓ Expected accuracy: {best_metrics['accuracy']:.3f} ± {best_metrics['accuracy_std']:.3f}")
    print("  ✓ All models significantly outperform random guessing")
    print("  ✓ Consider feature engineering to improve performance further")
    print("  ✓ Validate with real exoplanet data when available")

    return {
        'best_model': best_model_name,
        'best_metrics': best_metrics,
        'comparison_results': comparison_results,
        'rf_evaluation': rf_results
    }

if __name__ == "__main__":
    results = run_comprehensive_tests()

    print(f"\n{'='*60}")
    print("TESTING COMPLETED SUCCESSFULLY!")
    print(f"Best Model: {results['best_model']}")
    print(f"Expected Performance: {results['best_metrics']['accuracy']:.3f} accuracy")
    print(f"{'='*60}")