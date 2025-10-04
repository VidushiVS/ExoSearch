"""
Exoplanet Classification ML Model

This script trains a machine learning model to classify exoplanets as confirmed, false positive, etc.
using data from NASA Exoplanet Archive files.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import json
import re
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import logging
from collections import Counter
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
warnings.filterwarnings('ignore')

class ExoplanetClassifier:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []

    def load_and_preprocess_data(self, file_paths):
        """
        Load data from NASA Exoplanet Archive JSON files and preprocess for ML.

        Args:
            file_paths (list): List of file paths to JSON data files

        Returns:
            pd.DataFrame: Preprocessed dataframe ready for training
        """
        all_data = []

        for file_path in file_paths:
            print(f"Loading data from {file_path}...")

            try:
                # Try to load as regular JSON first
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                # If that fails, try to fix common JSON issues
                print(f"Attempting to fix JSON structure in {file_path}...")
                data = self._fix_json_structure(file_path)

            if data:
                # Extract records with actual planetary data
                valid_records = self._extract_valid_records(data)
                if valid_records:
                    all_data.extend(valid_records)
                    print(f"Extracted {len(valid_records)} valid records from {file_path}")
                else:
                    print(f"No valid records found in {file_path}")
            else:
                print(f"Could not load data from {file_path}")

        if not all_data:
            raise ValueError("No valid data found in any of the provided files")

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        # Identify classification target (disposition or similar field)
        target_column = self._identify_target_column(df)

        if not target_column:
            # Create a synthetic target for demonstration if no clear target exists
            print("No clear target column found, creating synthetic classification target...")
            df['disposition'] = self._create_synthetic_target(df)

        print(f"Dataset shape: {df.shape}")
        print(f"Target column: {target_column}")
        print(f"Available columns: {list(df.columns)}")

        return df

    def _fix_json_structure(self, file_path):
        """Attempt to fix common JSON structure issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Fix the problematic key that starts with #
            content = re.sub(r'"\s*#\s*This file was produced[^"]*"', '"metadata_comment"', content)

            # Try to parse the fixed content
            return json.loads(content)
        except Exception as e:
            print(f"Could not fix JSON structure: {e}")
            return None

    def _extract_valid_records(self, data):
        """Extract records that contain actual planetary data."""
        valid_records = []

        for record in data:
            if not isinstance(record, dict):
                continue

            # Skip metadata/header records
            if any(key.startswith('#') for key in record.keys()):
                continue

            # Skip records with mostly empty values
            non_empty_values = sum(1 for v in record.values() if v not in [None, '', [], 'N/A', 'null'])
            if non_empty_values < 3:  # At least 3 non-empty values
                continue

            valid_records.append(record)

        return valid_records

    def _identify_target_column(self, df):
        """Identify the most likely target column for classification."""
        # Common target column names for exoplanet classification
        target_candidates = [
            'disposition', 'default_flag', 'tfopwg_disp', 'pl_status',
            'solution_type', 'status', 'classification'
        ]

        for candidate in target_candidates:
            if candidate in df.columns:
                unique_values = df[candidate].dropna().unique()
                # Should have multiple classes but not too many
                if 2 <= len(unique_values) <= 10:
                    print(f"Found target column '{candidate}' with values: {unique_values}")
                    return candidate

        return None

    def _create_synthetic_target(self, df):
        """Create a synthetic classification target for demonstration."""
        # Simple rule-based classification
        targets = []

        for _, row in df.iterrows():
            # Use some basic rules to create classification
            if pd.notna(row.get('pl_rade')) and row.get('pl_rade', 0) > 0:
                if pd.notna(row.get('pl_orbper')) and row.get('pl_orbper', 0) > 0:
                    targets.append('confirmed')
                else:
                    targets.append('candidate')
            else:
                targets.append('false_positive')

        return targets

    def preprocess_features(self, df, target_column='disposition'):
        """
        Preprocess features for machine learning.

        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column

        Returns:
            tuple: (X_processed, y_processed, feature_columns)
        """
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")

        # Drop rows where target is missing
        df_clean = df.dropna(subset=[target_column]).copy()

        # Identify numeric and categorical columns
        numeric_columns = []
        categorical_columns = []

        for col in df_clean.columns:
            if col == target_column:
                continue

            # Check if column is numeric
            if df_clean[col].dtype in ['int64', 'float64']:
                numeric_columns.append(col)
            elif df_clean[col].dtype == 'object':
                # Check if it's actually numeric data stored as string
                try:
                    pd.to_numeric(df_clean[col], errors='coerce')
                    numeric_columns.append(col)
                except:
                    categorical_columns.append(col)

        print(f"Numeric columns: {numeric_columns}")
        print(f"Categorical columns: {categorical_columns}")

        # Handle missing values
        df_processed = df_clean.copy()

        # Fill numeric columns with median
        for col in numeric_columns:
            if df_processed[col].isnull().any():
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())

        # Fill categorical columns with mode
        for col in categorical_columns:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else 'Unknown')

        # Encode categorical variables
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])

        # Prepare feature matrix
        feature_columns = numeric_columns + categorical_columns
        X = df_processed[feature_columns].values

        # Encode target variable
        if target_column not in self.label_encoders:
            self.label_encoders[target_column] = LabelEncoder()

        y = self.label_encoders[target_column].fit_transform(df_processed[target_column])

        self.feature_columns = feature_columns

        print(f"Final feature matrix shape: {X.shape}")
        print(f"Target classes: {self.label_encoders[target_column].classes_}")

        return X, y, feature_columns

    def advanced_feature_engineering(self, df):
        """
        Create advanced features for better model performance.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        df_engineered = df.copy()

        # Physical property ratios
        if 'pl_rade' in df.columns and 'st_rad' in df.columns:
            df_engineered['radius_ratio'] = df['pl_rade'] / df['st_rad']
            df_engineered['radius_ratio_log'] = np.log1p(df_engineered['radius_ratio'])

        if 'pl_masse' in df.columns and 'st_mass' in df.columns:
            df_engineered['mass_ratio'] = df['pl_masse'] / df['st_mass']
            df_engineered['mass_ratio_log'] = np.log1p(df_engineered['mass_ratio'])

        # Orbital properties
        if 'pl_orbper' in df.columns:
            df_engineered['orbper_log'] = np.log1p(df['pl_orbper'])
            df_engineered['orbper_sqrt'] = np.sqrt(df['pl_orbper'])

        if 'pl_orbsmax' in df.columns:
            df_engineered['orbsmax_log'] = np.log1p(df['pl_orbsmax'])
            df_engineered['orbsmax_sq'] = df['pl_orbsmax'] ** 2

        # Temperature relations
        if 'pl_eqt' in df.columns and 'st_teff' in df.columns:
            df_engineered['temp_ratio'] = df['pl_eqt'] / df['st_teff']
            df_engineered['temp_diff'] = df['st_teff'] - df['pl_eqt']

        # Insolation calculations
        if 'st_lum' in df.columns and 'pl_orbsmax' in df.columns:
            df_engineered['insolation'] = df['st_lum'] / (df['pl_orbsmax'] ** 2)
            df_engineered['insolation_log'] = np.log1p(df_engineered['insolation'])

        # Statistical features
        numeric_columns = df_engineered.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col != 'disposition' and df_engineered[col].notna().sum() > 0:
                # Z-score
                df_engineered[f'{col}_zscore'] = stats.zscore(df_engineered[col].fillna(df_engineered[col].median()))

                # Quantile-based features
                for q in [0.25, 0.75, 0.95]:
                    df_engineered[f'{col}_q{q*100:.0f}'] = df_engineered.groupby('disposition')[col].transform('quantile', q)

        # Interaction features
        important_cols = ['pl_rade', 'pl_orbper', 'pl_orbsmax', 'st_teff', 'st_rad']
        for i, col1 in enumerate(important_cols):
            for col2 in important_cols[i+1:]:
                if col1 in df.columns and col2 in df.columns:
                    df_engineered[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-8)
                    df_engineered[f'{col1}_{col2}_product'] = df[col1] * df[col2]

        return df_engineered

    def detect_and_remove_outliers(self, df, method='isolation_forest', contamination=0.1):
        """
        Detect and remove outliers using advanced methods.

        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Outlier detection method
            contamination (float): Expected proportion of outliers

        Returns:
            pd.DataFrame: DataFrame with outliers removed
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.svm import OneClassSVM

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X_numeric = df[numeric_cols].fillna(df[numeric_cols].median())

        if method == 'isolation_forest':
            outlier_detector = IsolationForest(contamination=contamination, random_state=42)
        elif method == 'lof':
            outlier_detector = LocalOutlierFactor(contamination=contamination)
        elif method == 'svm':
            outlier_detector = OneClassSVM(nu=contamination)

        outlier_labels = outlier_detector.fit_predict(X_numeric)

        # Keep only inliers (-1 is outlier in IsolationForest, 1 is inlier)
        if method == 'isolation_forest':
            mask = outlier_labels == 1
        else:
            mask = outlier_labels != -1

        print(f"Removed {len(df) - mask.sum()} outliers out of {len(df)} samples")
        return df[mask].copy()

    def advanced_feature_selection(self, X, y, method='mutual_info', k=50):
        """
        Perform advanced feature selection.

        Args:
            X: Feature matrix
            y: Target vector
            method (str): Feature selection method
            k (int): Number of features to select

        Returns:
            tuple: (X_selected, selected_features)
        """
        print(f"Performing feature selection using {method}...")

        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            scores = selector.scores_

        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=k)
            X_selected = selector.fit_transform(X, y)
            scores = selector.ranking_

        elif method == 'tree_importance':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            estimator.fit(X, y)
            importances = estimator.feature_importances_

            # Select top k features
            indices = np.argsort(importances)[-k:]
            X_selected = X[:, indices]
            scores = importances[indices]

        # Get selected feature names
        feature_indices = selector.get_support(indices=True) if hasattr(selector, 'get_support') else indices
        selected_features = [self.feature_columns[i] for i in feature_indices]

        print(f"Selected {len(selected_features)} features: {selected_features[:10]}...")
        return X_selected, selected_features

    def advanced_preprocessing(self, df, target_column='disposition', outlier_removal=True):
        """
        Advanced preprocessing pipeline with outlier removal and feature engineering.

        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Target column name
            outlier_removal (bool): Whether to remove outliers

        Returns:
            tuple: (X_processed, y_processed, feature_columns)
        """
        print("Starting advanced preprocessing...")

        # Step 1: Feature engineering
        print("Step 1: Feature engineering...")
        df_engineered = self.advanced_feature_engineering(df)

        # Step 2: Outlier removal
        if outlier_removal:
            print("Step 2: Outlier removal...")
            df_clean = self.detect_and_remove_outliers(df_engineered)
        else:
            df_clean = df_engineered

        # Step 3: Separate features and target
        if target_column not in df_clean.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        df_final = df_clean.dropna(subset=[target_column]).copy()

        # Step 4: Advanced feature selection and preprocessing
        print("Step 3: Advanced feature selection and preprocessing...")

        # Identify column types
        numeric_columns = []
        categorical_columns = []

        for col in df_final.columns:
            if col == target_column:
                continue

            if df_final[col].dtype in ['int64', 'float64']:
                numeric_columns.append(col)
            elif df_final[col].dtype == 'object':
                try:
                    pd.to_numeric(df_final[col], errors='coerce')
                    numeric_columns.append(col)
                except:
                    categorical_columns.append(col)

        print(f"Numeric columns: {len(numeric_columns)}")
        print(f"Categorical columns: {len(categorical_columns)}")

        # Handle missing values with advanced imputation
        numeric_imputer = IterativeImputer(random_state=42, max_iter=10)
        categorical_imputer = SimpleImputer(strategy='most_frequent')

        # Impute numeric columns
        if numeric_columns:
            numeric_data = df_final[numeric_columns]
            df_final[numeric_columns] = numeric_imputer.fit_transform(numeric_data)

        # Impute categorical columns
        if categorical_columns:
            categorical_data = df_final[categorical_columns]
            df_final[categorical_columns] = categorical_imputer.fit_transform(categorical_data)

        # Encode categorical variables
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_final[col] = self.label_encoders[col].fit_transform(df_final[col])

        # Prepare feature matrix
        feature_columns = numeric_columns + categorical_columns
        X = df_final[feature_columns].values

        # Encode target
        if target_column not in self.label_encoders:
            self.label_encoders[target_column] = LabelEncoder()

        y = self.label_encoders[target_column].fit_transform(df_final[target_column])

        # Feature selection
        X_selected, selected_features = self.advanced_feature_selection(X, y, k=min(100, X.shape[1]))

        self.feature_columns = selected_features

        print(f"Final feature matrix shape: {X_selected.shape}")
        print(f"Target classes: {self.label_encoders[target_column].classes_}")

        return X_selected, y, selected_features

    def train_model(self, X, y, test_size=0.2, random_state=42):
        """
        Train the classification model.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Training set size: {X_train.shape}")
        print(f"Testing set size: {X_test.shape}")

        # Initialize and train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )

        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoders['disposition'].classes_
        ))

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 most important features:")
        print(feature_importance.head(10))

        return accuracy

    def train_high_performance_models(self, X, y, test_size=0.15, random_state=42):
        """
        Train multiple high-performance models and return the best one.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Test set size
            random_state: Random seed

        Returns:
            dict: Dictionary containing all trained models and their performance
        """
        print("Training high-performance models...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Handle class imbalance
        smote = SMOTE(random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        print(f"Balanced training set size: {X_train_balanced.shape}")

        models = {}
        performance_results = {}

        # 1. XGBoost (with fallback)
        print("Training XGBoost...")
        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=random_state,
                eval_metric='mlogloss'
            )
            xgb_model.fit(X_train_balanced, y_train_balanced)
            models['XGBoost'] = xgb_model

            y_pred_xgb = xgb_model.predict(X_test)
            performance_results['XGBoost'] = {
                'accuracy': accuracy_score(y_test, y_pred_xgb),
                'precision': precision_score(y_test, y_pred_xgb, average='weighted'),
                'recall': recall_score(y_test, y_pred_xgb, average='weighted'),
                'f1': f1_score(y_test, y_pred_xgb, average='weighted')
            }
        except Exception as e:
            print(f"XGBoost not available: {e}")
            performance_results['XGBoost'] = {'accuracy': 0, 'f1': 0}

        # 2. LightGBM (with fallback)
        print("Training LightGBM...")
        try:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=random_state,
                verbosity=-1
            )
            lgb_model.fit(X_train_balanced, y_train_balanced)
            models['LightGBM'] = lgb_model

            y_pred_lgb = lgb_model.predict(X_test)
            performance_results['LightGBM'] = {
                'accuracy': accuracy_score(y_test, y_pred_lgb),
                'precision': precision_score(y_test, y_pred_lgb, average='weighted'),
                'recall': recall_score(y_test, y_pred_lgb, average='weighted'),
                'f1': f1_score(y_test, y_pred_lgb, average='weighted')
            }
        except Exception as e:
            print(f"LightGBM not available: {e}")
            performance_results['LightGBM'] = {'accuracy': 0, 'f1': 0}

        # 3. CatBoost (with fallback)
        print("Training CatBoost...")
        try:
            cat_model = CatBoostClassifier(
                iterations=500,
                depth=8,
                learning_rate=0.1,
                random_seed=random_state,
                verbose=False
            )
            cat_model.fit(X_train_balanced, y_train_balanced)
            models['CatBoost'] = cat_model

            y_pred_cat = cat_model.predict(X_test)
            performance_results['CatBoost'] = {
                'accuracy': accuracy_score(y_test, y_pred_cat),
                'precision': precision_score(y_test, y_pred_cat, average='weighted'),
                'recall': recall_score(y_test, y_pred_cat, average='weighted'),
                'f1': f1_score(y_test, y_pred_cat, average='weighted')
            }
        except Exception as e:
            print(f"CatBoost not available: {e}")
            performance_results['CatBoost'] = {'accuracy': 0, 'f1': 0}

        # 4. Balanced Random Forest
        print("Training Balanced Random Forest...")
        brf_model = BalancedRandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
        brf_model.fit(X_train, y_train)
        models['BalancedRF'] = brf_model

        y_pred_brf = brf_model.predict(X_test)
        performance_results['BalancedRF'] = {
            'accuracy': accuracy_score(y_test, y_pred_brf),
            'precision': precision_score(y_test, y_pred_brf, average='weighted'),
            'recall': recall_score(y_test, y_pred_brf, average='weighted'),
            'f1': f1_score(y_test, y_pred_brf, average='weighted')
        }

        # Print performance comparison
        print("\nModel Performance Comparison:")
        print("=" * 60)
        for model_name, metrics in performance_results.items():
            print(f"{model_name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print()

        # Select best model
        best_model_name = max(performance_results.keys(),
                            key=lambda x: performance_results[x]['f1'])
        best_model = models[best_model_name]

        print(f"Best model: {best_model_name} (F1: {performance_results[best_model_name]['f1']:.4f})")

        return {
            'models': models,
            'performance': performance_results,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'X_test': X_test,
            'y_test': y_test
        }

    def create_ensemble_model(self, X, y, base_models=None):
        """
        Create an ensemble model using stacking and voting.

        Args:
            X: Feature matrix
            y: Target vector
            base_models: Dictionary of base models

        Returns:
            dict: Ensemble model and performance metrics
        """
        print("Creating ensemble model...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        # Handle class imbalance for training
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        if base_models is None:
            # Create base models (with fallbacks)
            base_models = {}

            # XGBoost
            try:
                base_models['XGBoost'] = xgb.XGBClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, random_state=42
                )
            except:
                pass

            # LightGBM
            try:
                base_models['LightGBM'] = lgb.LGBMClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
                )
            except:
                pass

            # CatBoost
            try:
                base_models['CatBoost'] = CatBoostClassifier(
                    iterations=300, depth=6, learning_rate=0.1,
                    random_seed=42, verbose=False
                )
            except:
                pass

            # Balanced Random Forest
            try:
                base_models['BalancedRF'] = BalancedRandomForestClassifier(
                    n_estimators=300, max_depth=8, random_state=42
                )
            except:
                # Fallback to regular Random Forest
                base_models['BalancedRF'] = RandomForestClassifier(
                    n_estimators=300, max_depth=8, random_state=42
                )

            # Ensure we have at least 2 models for ensemble
            if len(base_models) < 2:
                base_models['RandomForest'] = RandomForestClassifier(
                    n_estimators=300, max_depth=8, random_state=42
                )

        # Train base models
        trained_models = {}
        for name, model in base_models.items():
            print(f"Training {name}...")
            model.fit(X_train_balanced, y_train_balanced)
            trained_models[name] = model

        # Create ensemble models

        # 1. Voting Classifier (hard voting)
        voting_hard = VotingClassifier(
            estimators=list(trained_models.items()),
            voting='hard'
        )

        # 2. Voting Classifier (soft voting)
        voting_soft = VotingClassifier(
            estimators=list(trained_models.items()),
            voting='soft'
        )

        # 3. Stacking Classifier
        stacking = StackingClassifier(
            estimators=list(trained_models.items()),
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )

        # Train ensemble models
        ensemble_models = {
            'Voting_Hard': voting_hard,
            'Voting_Soft': voting_soft,
            'Stacking': stacking
        }

        for name, model in ensemble_models.items():
            print(f"Training {name}...")
            model.fit(X_train_balanced, y_train_balanced)

        # Evaluate all models
        results = {}

        for name, model in ensemble_models.items():
            y_pred = model.predict(X_test)
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }

        # Print results
        print("\nEnsemble Model Performance:")
        print("=" * 60)
        for model_name, metrics in results.items():
            print(f"{model_name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print()

        # Select best ensemble model
        best_ensemble_name = max(results.keys(), key=lambda x: results[x]['f1'])
        best_ensemble = ensemble_models[best_ensemble_name]

        print(f"Best ensemble: {best_ensemble_name} (F1: {results[best_ensemble_name]['f1']:.4f})")

        return {
            'base_models': trained_models,
            'ensemble_models': ensemble_models,
            'performance': results,
            'best_model': best_ensemble,
            'best_model_name': best_ensemble_name,
            'X_test': X_test,
            'y_test': y_test
        }

    def hyperparameter_optimization(self, X, y, model_type='xgboost', n_trials=50):
        """
        Perform hyperparameter optimization using Optuna.

        Args:
            X: Feature matrix
            y: Target vector
            model_type (str): Type of model to optimize
            n_trials (int): Number of optimization trials

        Returns:
            dict: Best model and parameters
        """
        print(f"Performing hyperparameter optimization for {model_type}...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        def objective(trial):
            try:
                if model_type == 'xgboost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                        'random_state': 42
                    }

                    model = xgb.XGBClassifier(**params)

                elif model_type == 'lightgbm':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                        'random_state': 42
                    }

                    model = lgb.LGBMClassifier(**params, verbosity=-1)

                elif model_type == 'catboost':
                    params = {
                        'iterations': trial.suggest_int('iterations', 100, 500),
                        'depth': trial.suggest_int('depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'random_seed': 42
                    }

                    model = CatBoostClassifier(**params, verbose=False)

                else:
                    # Fallback to Random Forest
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                        'random_state': 42
                    }

                    model = RandomForestClassifier(**params)

            except Exception as e:
                print(f"Advanced model not available for {model_type}, using Random Forest fallback")
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'random_state': 42
                }

                model = RandomForestClassifier(**params)

            # Cross-validation score
            scores = cross_val_score(model, X_train_balanced, y_train_balanced,
                                   cv=3, scoring='f1_weighted')
            return scores.mean()

        # Run optimization
        if OPTUNA_AVAILABLE:
            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials)
        else:
            print("Optuna not available, using RandomizedSearchCV fallback")
            from sklearn.model_selection import RandomizedSearchCV

            # Create parameter grid based on model type
            if model_type == 'xgboost':
                param_dist = {
                    'n_estimators': [100, 300, 500],
                    'max_depth': [3, 6, 8, 10],
                    'learning_rate': [0.01, 0.1, 0.2, 0.3],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                }
            else:  # Random Forest fallback
                param_dist = {
                    'n_estimators': [100, 300, 500],
                    'max_depth': [3, 6, 8, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }

            # Create base model
            try:
                if model_type == 'xgboost':
                    base_model = xgb.XGBClassifier(random_state=42)
                else:
                    base_model = RandomForestClassifier(random_state=42)
            except:
                base_model = RandomForestClassifier(random_state=42)

            # Run randomized search
            study = RandomizedSearchCV(
                base_model, param_dist, n_iter=n_trials, cv=3,
                scoring='f1_weighted', random_state=42, n_jobs=-1
            )
            study.fit(X_train_balanced, y_train_balanced)

            # Convert to study-like object for compatibility
            class StudyFallback:
                def __init__(self, search_cv):
                    self.best_params = search_cv.best_params_
                    self.best_value = search_cv.best_score_

            study = StudyFallback(study)

        print(f"Best {model_type} parameters: {study.best_params}")
        print(f"Best CV score: {study.best_value:.4f}")

        # Train final model with best parameters
        try:
            if model_type == 'xgboost':
                best_model = xgb.XGBClassifier(**study.best_params, random_state=42)
            elif model_type == 'lightgbm':
                best_model = lgb.LGBMClassifier(**study.best_params, random_state=42, verbosity=-1)
            elif model_type == 'catboost':
                best_model = CatBoostClassifier(**study.best_params, verbose=False)
            else:
                # Fallback to Random Forest
                best_model = RandomForestClassifier(**study.best_params, random_state=42)
        except Exception as e:
            print(f"Advanced model not available, using Random Forest fallback")
            best_model = RandomForestClassifier(
                n_estimators=study.best_params.get('n_estimators', 300),
                max_depth=study.best_params.get('max_depth', 10),
                random_state=42
            )

        best_model.fit(X_train_balanced, y_train_balanced)

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        test_performance = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }

        print(f"Test set performance: {test_performance}")

        return {
            'model': best_model,
            'best_params': study.best_params,
            'best_score': study.best_value,
            'test_performance': test_performance,
            'study': study
        }

    def comprehensive_evaluation(self, X, y, model_name="Random Forest"):
        """
        Perform comprehensive model evaluation with multiple metrics.

        Args:
            X: Feature matrix
            y: Target vector
            model_name: Name of the model for reporting
        """
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE MODEL EVALUATION - {model_name}")
        print(f"{'='*60}")

        # Cross-validation scores
        print("\n1. CROSS-VALIDATION ANALYSIS")
        print("-" * 40)

        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        print(f"CV Accuracy Scores: {[f'{score:.3f}' for score in cv_scores]}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std() * 2:.3f})")

        # Precision, Recall, F1 scores per class
        cv_precision = cross_val_score(self.model, X, y, cv=5, scoring='precision_macro')
        cv_recall = cross_val_score(self.model, X, y, cv=5, scoring='recall_macro')
        cv_f1 = cross_val_score(self.model, X, y, cv=5, scoring='f1_macro')

        print(f"CV Precision (macro): {cv_precision.mean():.3f} (±{cv_precision.std() * 2:.3f})")
        print(f"CV Recall (macro): {cv_recall.mean():.3f} (±{cv_recall.std() * 2:.3f})")
        print(f"CV F1-Score (macro): {cv_f1.mean():.3f} (±{cv_f1.std() * 2:.3f})")

        # Confusion Matrix Analysis
        print("\n2. CONFUSION MATRIX ANALYSIS")
        print("-" * 40)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Confusion Matrix (Raw):")
        print(cm)
        print("\nConfusion Matrix (Normalized):")
        print(cm_normalized)

        # Per-class metrics from confusion matrix
        print("\nPer-Class Performance:")
        class_names = self.label_encoders['disposition'].classes_

        for i, class_name in enumerate(class_names):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            print(f"  {class_name}:")
            print(f"    Precision: {precision:.3f}")
            print(f"    Recall: {recall:.3f}")
            print(f"    F1-Score: {f1:.3f}")
            print(f"    Specificity: {specificity:.3f}")

        # ROC Analysis (for binary classification)
        if len(class_names) == 2:
            print("\n3. ROC CURVE ANALYSIS")
            print("-" * 40)

            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            print(f"ROC AUC Score: {roc_auc:.3f}")

            # Find optimal threshold
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            print(f"Optimal Threshold: {optimal_threshold:.3f}")

            # Precision-Recall Curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall_curve, precision_curve)
            print(f"Precision-Recall AUC: {pr_auc:.3f}")

        # Statistical Significance Testing
        print("\n4. STATISTICAL SIGNIFICANCE")
        print("-" * 40)

        # Compare model accuracy to random guessing
        random_accuracy = 1.0 / len(class_names)
        t_stat, p_value = stats.ttest_1samp(cv_scores, random_accuracy)

        print(f"Random Guessing Accuracy: {random_accuracy:.3f}")
        print(f"Model vs Random - t-statistic: {t_stat:.3f}")
        print(f"Model vs Random - p-value: {p_value:.6f}")

        if p_value < 0.05:
            print("✓ Model significantly outperforms random guessing (p < 0.05)")
        else:
            print("✗ Model performance not significantly better than random guessing")

        # Feature Importance Analysis
        print("\n5. FEATURE IMPORTANCE ANALYSIS")
        print("-" * 40)

        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_,
            'importance_percentage': self.model.feature_importances_ * 100
        }).sort_values('importance', ascending=False)

        print("Top 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f} ({row['importance_percentage']:.1f}%)")

        # Cumulative importance
        cumulative_importance = feature_importance['importance'].cumsum()
        n_features_95 = (cumulative_importance <= 0.95).sum() + 1
        print(f"\nNumber of features needed for 95% importance: {n_features_95}")
        print(f"Total number of features: {len(self.feature_columns)}")

        return {
            'cv_scores': cv_scores,
            'confusion_matrix': cm,
            'feature_importance': feature_importance,
            'metrics': {
                'accuracy': cv_scores.mean(),
                'precision': cv_precision.mean(),
                'recall': cv_recall.mean(),
                'f1': cv_f1.mean()
            }
        }

    def compare_models(self, X, y):
        """
        Compare multiple ML algorithms for exoplanet classification.

        Args:
            X: Feature matrix
            y: Target vector
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

        # Statistical comparison between best and second best
        sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        best_model = sorted_models[0]
        second_best = sorted_models[1]

        print("\nSTATISTICAL COMPARISON")
        print("-" * 30)
        print(f"Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.3f})")
        print(f"Second Best: {second_best[0]} (Accuracy: {second_best[1]['accuracy']:.3f})")

        # Perform t-test between best and second best
        # Note: In practice, you'd need to run each model multiple times to get variance
        accuracy_diff = best_model[1]['accuracy'] - second_best[1]['accuracy']
        print(f"Accuracy Difference: {accuracy_diff:.3f}")

        if abs(accuracy_diff) > 0.01:  # Meaningful difference
            print(f"{'✓' if accuracy_diff > 0 else '✗'} {best_model[0]} shows {'higher' if accuracy_diff > 0 else 'lower'} accuracy than {second_best[0]}")
        else:
            print("No significant difference between top models")

        return results

    def save_model(self, filepath='exoplanet_model.pkl'):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No trained model to save")

        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath='exoplanet_model.pkl'):
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']

        print(f"Model loaded from {filepath}")

def main():
    """Main function to run high-performance exoplanet classification training."""
    print("High-Performance Exoplanet Classification ML Training")
    print("=" * 60)

    # Initialize classifier
    classifier = ExoplanetClassifier()

    # File paths (update these with your actual file paths)
    data_files = [
        'k2pandc_2025.10.04_07.10.02.json',
        'TOI_2025.10.04_07.06.07.json',
        '../../../Downloads/cumulative_2025.10.04_06.25.10.json'
    ]

    try:
        # Load and preprocess data with advanced preprocessing
        print("Step 1: Loading and preprocessing data with advanced techniques...")
        df = classifier.load_and_preprocess_data(data_files)

        # Advanced preprocessing with feature engineering and outlier removal
        print("\nStep 2: Advanced preprocessing with feature engineering...")
        X, y, feature_columns = classifier.advanced_preprocessing(df)

        # Train high-performance models
        print("\nStep 3: Training high-performance models...")
        hp_results = classifier.train_high_performance_models(X, y)

        # Create ensemble model
        print("\nStep 4: Creating ensemble model...")
        ensemble_results = classifier.create_ensemble_model(X, y)

        # Hyperparameter optimization
        print("\nStep 5: Hyperparameter optimization...")
        # Use the best individual model for optimization
        best_individual_model = hp_results['best_model_name']
        opt_results = classifier.hyperparameter_optimization(X, y, best_individual_model.lower())

        # Comprehensive evaluation of best model
        print("\nStep 6: Final comprehensive evaluation...")
        best_model = opt_results['model']
        classifier.model = best_model

        # Use original train/test split for fair evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        final_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }

        # Save best model
        print("\nStep 7: Saving best model...")
        model_data = {
            'model': best_model,
            'label_encoders': classifier.label_encoders,
            'feature_columns': feature_columns,
            'performance_metrics': final_metrics,
            'model_type': f'Optimized_{best_individual_model}'
        }

        with open('exoplanet_high_performance_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n{'='*60}")
        print("HIGH-PERFORMANCE TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print("Final Model Performance:")
        print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
        print(f"  Precision: {final_metrics['precision']:.4f}")
        print(f"  Recall:    {final_metrics['recall']:.4f}")
        print(f"  F1-Score:  {final_metrics['f1']:.4f}")
        print(f"\nModel saved as: exoplanet_high_performance_model.pkl")

        # Check if we achieved 98%+ performance
        if all(value >= 0.98 for value in final_metrics.values()):
            print("🎉 SUCCESS: All metrics are 98% or higher!")
        else:
            print("⚠️  Performance below 98% threshold. Consider further optimization.")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()