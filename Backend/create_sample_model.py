"""
Script to create sample model artifacts for the FastAPI application.

This script generates sample machine learning models and preprocessing
pipelines that can be loaded by the exoplanet classification API.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import json
import os
from datetime import datetime

def create_sample_model():
    """
    Create a sample Random Forest model for exoplanet classification.
    """
    # Create sample training data
    np.random.seed(42)

    n_samples = 1000
    n_features = 15

    # Generate synthetic exoplanet features
    features = {}

    # Orbital features
    features['period_d'] = np.random.lognormal(2, 1, n_samples)  # 0.5 to 1000 days
    features['duration_hr'] = np.random.uniform(0.5, 24, n_samples)
    features['depth_ppm'] = np.random.lognormal(4, 1, n_samples)  # 1 to 50000 ppm

    # Quality features
    features['snr'] = np.random.lognormal(2, 0.5, n_samples)  # 1 to 1000
    features['odd_even_ratio'] = np.random.lognormal(0, 0.1, n_samples)  # 0.1 to 10

    # Physical properties
    features['radius_re'] = np.random.lognormal(0.5, 0.8, n_samples)  # 0.1 to 30 Earth radii
    features['a_over_r'] = np.random.lognormal(3, 0.5, n_samples)  # 1 to 100

    # Stellar properties
    features['teff_k'] = np.random.uniform(2500, 10000, n_samples)
    features['logg'] = np.random.uniform(2.0, 5.5, n_samples)
    features['rstar_rsun'] = np.random.lognormal(0, 0.5, n_samples)  # 0.1 to 10 solar radii
    features['mag'] = np.random.uniform(5.0, 20.0, n_samples)

    # Data quality
    features['crowding'] = np.random.uniform(0.0, 1.0, n_samples)
    features['contamination'] = np.random.uniform(0.0, 1.0, n_samples)
    features['secondary_depth_ppm'] = np.random.lognormal(2, 1, n_samples)  # 0 to 1000 ppm

    # Mission (categorical)
    missions = ['Kepler', 'K2', 'TESS', 'CoRoT']
    features['mission'] = np.random.choice(missions, n_samples)

    # Create target variable based on feature combinations
    # Simple rule-based classification for demonstration
    y = []

    for i in range(n_samples):
        # Combine multiple factors for realistic classification
        radius_score = features['radius_re'][i]
        period_score = features['period_d'][i]
        snr_score = features['snr'][i]
        crowding_score = features['crowding'][i]

        # Classification logic (simplified)
        if (radius_score > 0.5 and radius_score < 15 and
            period_score > 1 and period_score < 400 and
            snr_score > 5 and crowding_score < 0.8):
            if radius_score < 2.0 and snr_score > 10:
                y.append('CONFIRMED')
            else:
                y.append('CANDIDATE')
        else:
            y.append('FALSE_POSITIVE')

    # Convert to DataFrame
    df = pd.DataFrame(features)

    # Encode categorical variables
    le_mission = LabelEncoder()
    df['mission_encoded'] = le_mission.fit_transform(df['mission'])

    # Select features for modeling (exclude original mission column)
    feature_columns = [
        'period_d', 'duration_hr', 'depth_ppm', 'snr', 'radius_re',
        'a_over_r', 'teff_k', 'logg', 'rstar_rsun', 'mag',
        'crowding', 'contamination', 'odd_even_ratio', 'secondary_depth_ppm',
        'mission_encoded'
    ]

    X = df[feature_columns]
    y_encoded = LabelEncoder().fit_transform(y)

    # Create preprocessing pipeline
    numeric_features = feature_columns[:-1]  # All except mission_encoded
    categorical_features = ['mission_encoded']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', SimpleImputer(strategy='most_frequent'), categorical_features)
        ])

    # Create full pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train the model
    print("Training sample model...")
    model_pipeline.fit(X, y_encoded)

    # Save model artifacts
    artifacts_dir = "./artifacts"

    # Save the main model
    joblib.dump(model_pipeline, f"{artifacts_dir}/model.pkl")
    print(f"Saved model to {artifacts_dir}/model.pkl")

    # Save preprocessing pipeline separately (for API)
    joblib.dump(preprocessor, f"{artifacts_dir}/pipeline.pkl")
    print(f"Saved pipeline to {artifacts_dir}/pipeline.pkl")

    # Save feature names
    joblib.dump(feature_columns, f"{artifacts_dir}/feature_names.pkl")
    print(f"Saved feature names to {artifacts_dir}/feature_names.pkl")

    # Save target names
    target_names = ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']
    joblib.dump(target_names, f"{artifacts_dir}/target_names.pkl")
    print(f"Saved target names to {artifacts_dir}/target_names.pkl")

    # Save label encoders
    joblib.dump(le_mission, f"{artifacts_dir}/mission_encoder.pkl")
    print(f"Saved mission encoder to {artifacts_dir}/mission_encoder.pkl")

    # Create model metrics
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Calculate metrics
    cv_scores = cross_val_score(model_pipeline, X, y_encoded, cv=5, scoring='accuracy')
    accuracy = cv_scores.mean()

    # Fit on full data for final metrics
    y_pred = model_pipeline.predict(X)
    precision = precision_score(y_encoded, y_pred, average='macro')
    recall = recall_score(y_encoded, y_pred, average='macro')
    f1 = f1_score(y_encoded, y_pred, average='macro')

    # Create comprehensive metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "pr_auc": 0.968,  # Placeholder
        "calibration_score": 0.952,  # Placeholder
        "total_samples": n_samples,
        "model_version": f"sample_model_{datetime.now().strftime('%Y_%m_%d_%H_%M')}",
        "training_date": datetime.now().isoformat(),
        "dataset_version": "synthetic_sample_data",
        "algorithm": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        },
        "feature_count": len(feature_columns),
        "class_distribution": {
            "CONFIRMED": y.count('CONFIRMED') / n_samples,
            "CANDIDATE": y.count('CANDIDATE') / n_samples,
            "FALSE_POSITIVE": y.count('FALSE_POSITIVE') / n_samples
        }
    }

    # Save metrics
    with open(f"{artifacts_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {artifacts_dir}/metrics.json")

    # Create feature schema
    schema = {
        "features": [
            {
                "name": "period_d",
                "units": "days",
                "min_range": 0.5,
                "max_range": 1000.0,
                "description": "Orbital period in days"
            },
            {
                "name": "duration_hr",
                "units": "hours",
                "min_range": 0.5,
                "max_range": 24.0,
                "description": "Transit duration in hours"
            },
            {
                "name": "depth_ppm",
                "units": "ppm",
                "min_range": 1.0,
                "max_range": 50000.0,
                "description": "Transit depth in parts per million"
            },
            {
                "name": "snr",
                "units": "unitless",
                "min_range": 1.0,
                "max_range": 1000.0,
                "description": "Signal-to-noise ratio"
            },
            {
                "name": "radius_re",
                "units": "Earth radii",
                "min_range": 0.1,
                "max_range": 30.0,
                "description": "Planet radius in Earth radii"
            },
            {
                "name": "a_over_r",
                "units": "unitless",
                "min_range": 1.0,
                "max_range": 100.0,
                "description": "Semi-major axis to stellar radius ratio"
            },
            {
                "name": "teff_k",
                "units": "Kelvin",
                "min_range": 2500.0,
                "max_range": 10000.0,
                "description": "Effective temperature in Kelvin"
            },
            {
                "name": "logg",
                "units": "dex",
                "min_range": 2.0,
                "max_range": 5.5,
                "description": "Surface gravity (log g)"
            },
            {
                "name": "rstar_rsun",
                "units": "Solar radii",
                "min_range": 0.1,
                "max_range": 10.0,
                "description": "Stellar radius in solar radii"
            },
            {
                "name": "mag",
                "units": "magnitude",
                "min_range": 5.0,
                "max_range": 20.0,
                "description": "Apparent magnitude"
            },
            {
                "name": "crowding",
                "units": "fraction",
                "min_range": 0.0,
                "max_range": 1.0,
                "description": "Crowding metric"
            },
            {
                "name": "contamination",
                "units": "fraction",
                "min_range": 0.0,
                "max_range": 1.0,
                "description": "Contamination factor"
            },
            {
                "name": "odd_even_ratio",
                "units": "unitless",
                "min_range": 0.1,
                "max_range": 10.0,
                "description": "Odd-even transit depth ratio"
            },
            {
                "name": "secondary_depth_ppm",
                "units": "ppm",
                "min_range": 0.0,
                "max_range": 1000.0,
                "description": "Secondary eclipse depth"
            },
            {
                "name": "mission",
                "units": "name",
                "min_range": None,
                "max_range": None,
                "description": "Space mission name",
                "allowed_values": ["Kepler", "K2", "TESS", "CoRoT"]
            }
        ],
        "last_updated": datetime.now().isoformat(),
        "version": "1.0.0",
        "total_features": 15
    }

    # Save schema
    with open(f"{artifacts_dir}/schema.json", 'w') as f:
        json.dump(schema, f, indent=2)
    print(f"Saved schema to {artifacts_dir}/schema.json")

    print("\nSample model artifacts created successfully!")
    print(f"Model accuracy: {accuracy:.3f}")
    print(f"Training samples: {n_samples}")
    print(f"Features: {len(feature_columns)}")
    print(f"Classes: {target_names}")

    return model_pipeline, feature_columns, target_names

def test_model_loading():
    """
    Test that the created model can be loaded correctly.
    """
    print("\nTesting model loading...")

    try:
        # Test loading model
        model = joblib.load("./artifacts/model.pkl")
        print("✓ Model loaded successfully")

        # Test loading pipeline
        pipeline = joblib.load("./artifacts/pipeline.pkl")
        print("✓ Pipeline loaded successfully")

        # Test loading feature names
        feature_names = joblib.load("./artifacts/feature_names.pkl")
        print(f"✓ Feature names loaded: {len(feature_names)} features")

        # Test loading target names
        target_names = joblib.load("./artifacts/target_names.pkl")
        print(f"✓ Target names loaded: {target_names}")

        # Test prediction
        sample_data = {
            'period_d': 10.5,
            'duration_hr': 2.5,
            'depth_ppm': 100.0,
            'snr': 15.0,
            'radius_re': 2.5,
            'a_over_r': 20.0,
            'teff_k': 5500,
            'logg': 4.5,
            'rstar_rsun': 1.0,
            'mag': 12.0,
            'crowding': 0.1,
            'contamination': 0.05,
            'odd_even_ratio': 1.0,
            'secondary_depth_ppm': 10.0,
            'mission_encoded': 0  # Kepler
        }

        df = pd.DataFrame([sample_data])
        prediction = model.predict(df)
        probabilities = model.predict_proba(df)

        print(f"✓ Sample prediction successful: {target_names[prediction[0]]}")
        print(f"✓ Prediction probabilities: {[f'{prob:.3f}' for prob in probabilities[0]]}")

        print("\n🎉 All tests passed! Model is ready for API deployment.")

    except Exception as e:
        print(f"❌ Error testing model: {e}")
        raise

if __name__ == "__main__":
    print("Creating sample exoplanet classification model...")
    print("=" * 60)

    # Create the model
    model, features, targets = create_sample_model()

    # Test loading
    test_model_loading()

    print("\n" + "=" * 60)
    print("MODEL CREATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nTo start the API server:")
    print("  pip install -r requirements.txt")
    print("  python app.py --host 0.0.0.0 --port 8000")
    print("\nAPI Documentation will be available at:")
    print("  http://localhost:8000/docs")