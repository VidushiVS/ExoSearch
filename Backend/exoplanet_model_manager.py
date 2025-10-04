"""
Advanced Exoplanet Model Management System

This module provides comprehensive model training, evaluation, and management
capabilities for exoplanet classification with scientific rigor and validation.
"""

import os
import json
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import hashlib
import shutil

# Scientific computing and ML imports
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate,
    train_test_split, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, log_loss, brier_score_loss
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.calibration import calibration_curve
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExoplanetModelManager:
    """
    Advanced model management system for exoplanet classification.

    Provides comprehensive training, evaluation, and deployment capabilities
    with scientific validation and reproducibility.
    """

    def __init__(self, artifacts_dir: str = "./artifacts", models_dir: str = "./models"):
        """
        Initialize the model manager.

        Args:
            artifacts_dir: Directory for model artifacts and metadata
            models_dir: Directory for storing trained models
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = Path(models_dir)
        self.current_model = None
        self.model_history = []
        self.feature_names = []
        self.target_names = ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']

        # Create directories if they don't exist
        self.artifacts_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

        # Load model history
        self._load_model_history()

        logger.info("ExoplanetModelManager initialized")

    def _load_model_history(self):
        """Load model training history from disk."""
        history_file = self.artifacts_dir / "model_history.json"

        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.model_history = json.load(f)
                logger.info(f"Loaded model history: {len(self.model_history)} models")
            except Exception as e:
                logger.warning(f"Could not load model history: {e}")
                self.model_history = []
        else:
            self.model_history = []

    def _save_model_history(self):
        """Save model training history to disk."""
        history_file = self.artifacts_dir / "model_history.json"

        try:
            with open(history_file, 'w') as f:
                json.dump(self.model_history, f, indent=2, default=str)
            logger.info(f"Saved model history: {len(self.model_history)} models")
        except Exception as e:
            logger.error(f"Could not save model history: {e}")

    def _generate_model_hash(self, model_config: Dict) -> str:
        """Generate a unique hash for model configuration."""
        config_str = json.dumps(model_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def load_and_preprocess_nasa_data(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Load and preprocess NASA exoplanet data from multiple sources.

        Args:
            file_paths: List of paths to NASA exoplanet data files

        Returns:
            Preprocessed DataFrame ready for modeling
        """
        logger.info("Loading NASA exoplanet data...")

        all_data = []
        data_sources = []

        for file_path in file_paths:
            try:
                # Load JSON data
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract valid records
                valid_records = self._extract_nasa_records(data)

                if valid_records:
                    all_data.extend(valid_records)
                    data_sources.append(Path(file_path).name)
                    logger.info(f"Loaded {len(valid_records)} records from {file_path}")
                else:
                    logger.warning(f"No valid records found in {file_path}")

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue

        if not all_data:
            raise ValueError("No valid data found in any of the provided files")

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        logger.info(f"Total dataset size: {df.shape}")
        logger.info(f"Data sources: {data_sources}")

        # Preprocess the data
        df_processed = self._preprocess_nasa_dataframe(df)

        return df_processed

    def _extract_nasa_records(self, data: List) -> List[Dict]:
        """Extract valid records from NASA exoplanet data."""
        valid_records = []

        for record in data:
            if not isinstance(record, dict):
                continue

            # Skip metadata records
            if any(key.startswith('#') for key in record.keys()):
                continue

            # Check for required fields
            required_fields = ['pl_name', 'disposition']
            if not all(field in record for field in required_fields):
                continue

            # Skip records with missing critical data
            if (record.get('pl_name') in [None, '', 'N/A'] or
                record.get('disposition') in [None, '', 'N/A']):
                continue

            # Clean the record
            cleaned_record = self._clean_nasa_record(record)
            if cleaned_record:
                valid_records.append(cleaned_record)

        return valid_records

    def _clean_nasa_record(self, record: Dict) -> Optional[Dict]:
        """Clean and validate a single NASA record."""
        cleaned = {}

        # Field name mappings (NASA field names to our standard names)
        field_mappings = {
            'pl_name': 'planet_name',
            'hostname': 'host_name',
            'disposition': 'disposition',
            'pl_orbper': 'period_d',
            'pl_rade': 'radius_re',
            'pl_radj': 'radius_rj',
            'pl_bmasse': 'mass_earth',
            'pl_bmassj': 'mass_jupiter',
            'pl_orbsmax': 'semi_major_axis',
            'pl_eccen': 'eccentricity',
            'pl_insol': 'insolation',
            'pl_eqt': 'equilibrium_temp',
            'pl_orbincl': 'inclination',
            'st_teff': 'teff_k',
            'st_rad': 'rstar_rsun',
            'st_mass': 'mstar_msun',
            'st_met': 'metallicity',
            'st_logg': 'logg',
            'sy_dist': 'distance_pc',
            'sy_vmag': 'magnitude',
            'discoverymethod': 'discovery_method',
            'disc_year': 'discovery_year',
            'disc_facility': 'facility'
        }

        for nasa_field, standard_field in field_mappings.items():
            value = record.get(nasa_field)
            if value not in [None, '', 'N/A', 'null', []]:
                try:
                    # Convert to numeric if possible
                    if isinstance(value, str):
                        # Handle string numbers
                        if value.replace('.', '').replace('-', '').isdigit():
                            value = float(value)
                        elif value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'

                    cleaned[standard_field] = value
                except (ValueError, TypeError):
                    cleaned[standard_field] = value

        # Calculate derived features
        cleaned.update(self._calculate_derived_features(cleaned))

        # Validate essential fields
        if 'disposition' not in cleaned or 'period_d' not in cleaned:
            return None

        return cleaned

    def _calculate_derived_features(self, record: Dict) -> Dict:
        """Calculate derived features from basic measurements."""
        derived = {}

        try:
            # Planet-star radius ratio
            if 'radius_re' in record and 'rstar_rsun' in record:
                derived['radius_ratio'] = record['radius_re'] / (record['rstar_rsun'] * 109.076)

            # Transit depth (approximate)
            if 'radius_ratio' in derived:
                derived['transit_depth_ppm'] = (derived['radius_ratio'] ** 2) * 1e6

            # Expected transit duration (approximate)
            if all(k in record for k in ['period_d', 'rstar_rsun', 'semi_major_axis']):
                # Simplified duration calculation
                period = record['period_d']
                rstar = record['rstar_rsun']
                sma = record.get('semi_major_axis', 1.0)  # Assume 1 AU if missing

                # Approximate duration in hours
                duration = (period * rstar) / (np.sqrt(sma) * 24)
                derived['duration_hr'] = max(0.5, min(24.0, duration))

            # Insolation flux relative to Earth
            if all(k in record for k in ['rstar_rsun', 'teff_k', 'semi_major_axis']):
                rstar = record['rstar_rsun']
                teff = record['teff_k']
                sma = record.get('semi_major_axis', 1.0)

                # Approximate insolation
                insolation = (rstar ** 2 * teff ** 4) / (sma ** 2)
                derived['insolation_earth'] = insolation / 1366  # Normalize to Earth

        except (TypeError, ZeroDivisionError, KeyError):
            pass

        return derived

    def _preprocess_nasa_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the NASA dataframe for machine learning."""
        logger.info("Preprocessing NASA dataframe...")

        # Handle missing values
        df_processed = df.copy()

        # Fill numeric columns with median
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())

        # Fill categorical columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode().iloc[0])

        # Encode categorical variables
        label_encoders = {}
        for col in categorical_columns:
            if col != 'disposition':  # Don't encode target
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le

        # Save label encoders
        joblib.dump(label_encoders, self.artifacts_dir / "label_encoders.pkl")

        logger.info(f"Preprocessing complete. Shape: {df_processed.shape}")
        return df_processed

    def train_advanced_model(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           model_config: Dict,
                           test_size: float = 0.2,
                           random_state: int = 42) -> Dict:
        """
        Train an advanced exoplanet classification model with comprehensive evaluation.

        Args:
            X: Feature matrix
            y: Target vector
            model_config: Model configuration dictionary
            test_size: Test set size
            random_state: Random seed

        Returns:
            Comprehensive training results
        """
        logger.info("Training advanced exoplanet model...")

        # Generate model hash for versioning
        model_hash = self._generate_model_hash(model_config)
        timestamp = datetime.now().isoformat()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Create preprocessing pipeline
        preprocessor = self._create_preprocessing_pipeline()

        # Create and train model
        model = self._create_model(model_config)
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('classification', model)
        ])

        # Train model
        logger.info("Fitting model...")
        pipeline.fit(X_train, y_train)

        # Comprehensive evaluation
        evaluation_results = self._comprehensive_model_evaluation(
            pipeline, X_train, X_test, y_train, y_test, model_config
        )

        # Create model metadata
        model_metadata = {
            'model_id': model_hash,
            'timestamp': timestamp,
            'model_config': model_config,
            'training_config': {
                'test_size': test_size,
                'random_state': random_state,
                'feature_count': X.shape[1],
                'sample_count': X.shape[0],
                'class_distribution': np.bincount(y).tolist()
            },
            'evaluation_results': evaluation_results,
            'data_hash': self._generate_data_hash(X, y)
        }

        # Save model and metadata
        self._save_model_version(pipeline, model_metadata)

        # Update history
        self.model_history.append(model_metadata)
        self._save_model_history()

        logger.info(f"Model training completed. ID: {model_hash}")

        return model_metadata

    def _create_preprocessing_pipeline(self) -> ColumnTransformer:
        """Create advanced preprocessing pipeline."""
        # Numeric preprocessing
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # For this example, assume all features are numeric
        # In practice, you'd separate numeric and categorical features
        preprocessor = ColumnTransformer([
            ('numeric', numeric_pipeline, slice(None))
        ])

        return preprocessor

    def _create_model(self, config: Dict):
        """Create model based on configuration."""
        algorithm = config.get('algorithm', 'RandomForest')

        if algorithm == 'RandomForest':
            return RandomForestClassifier(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', 10),
                min_samples_split=config.get('min_samples_split', 2),
                min_samples_leaf=config.get('min_samples_leaf', 1),
                random_state=config.get('random_state', 42),
                n_jobs=-1
            )
        elif algorithm == 'GradientBoosting':
            return GradientBoostingClassifier(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', 3),
                random_state=config.get('random_state', 42)
            )
        elif algorithm == 'LogisticRegression':
            return LogisticRegression(
                random_state=config.get('random_state', 42),
                max_iter=1000
            )
        elif algorithm == 'SVM':
            return SVC(
                random_state=config.get('random_state', 42),
                probability=True
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _comprehensive_model_evaluation(self,
                                      pipeline,
                                      X_train,
                                      X_test,
                                      y_train,
                                      y_test,
                                      model_config: Dict) -> Dict:
        """Perform comprehensive model evaluation."""
        logger.info("Performing comprehensive model evaluation...")

        results = {}

        # Basic metrics on test set
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)

        results['test_metrics'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'log_loss': log_loss(y_test, y_pred_proba)
        }

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        cv_scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        )

        results['cross_validation'] = {
            'accuracy_mean': cv_scores['test_accuracy'].mean(),
            'accuracy_std': cv_scores['test_accuracy'].std(),
            'precision_mean': cv_scores['test_precision_macro'].mean(),
            'recall_mean': cv_scores['test_recall_macro'].mean(),
            'f1_mean': cv_scores['test_f1_macro'].mean()
        }

        # ROC and PR curves for each class
        results['roc_curves'] = {}
        results['pr_curves'] = {}

        for i, class_name in enumerate(self.target_names):
            y_test_binary = (y_test == i).astype(int)
            y_pred_proba_class = y_pred_proba[:, i]

            # ROC curve
            fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba_class)
            roc_auc = auc(fpr, tpr)
            results['roc_curves'][class_name] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc
            }

            # Precision-Recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_test_binary, y_pred_proba_class)
            pr_auc = auc(recall_curve, precision_curve)
            results['pr_curves'][class_name] = {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(),
                'auc': pr_auc
            }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = cm.tolist()

        # Per-class metrics
        results['per_class_metrics'] = {}
        for i, class_name in enumerate(self.target_names):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp

            results['per_class_metrics'][class_name] = {
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'support': int(cm[i, :].sum())
            }

        # Feature importance (if available)
        if hasattr(pipeline.named_steps['classification'], 'feature_importances_'):
            feature_importance = pipeline.named_steps['classification'].feature_importances_
            results['feature_importance'] = feature_importance.tolist()

        # Calibration analysis
        results['calibration'] = self._analyze_calibration(pipeline, X_test, y_test)

        return results

    def _analyze_calibration(self, pipeline, X_test, y_test) -> Dict:
        """Analyze model calibration."""
        y_pred_proba = pipeline.predict_proba(X_test)

        calibration_results = {}

        for i, class_name in enumerate(self.target_names):
            prob_true, prob_pred = calibration_curve(
                (y_test == i).astype(int), y_pred_proba[:, i], n_bins=10
            )

            calibration_results[class_name] = {
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist()
            }

        return calibration_results

    def _generate_data_hash(self, X: np.ndarray, y: np.ndarray) -> str:
        """Generate hash of training data for reproducibility."""
        data_str = f"{X.tobytes()}{y.tobytes()}"
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def _save_model_version(self, pipeline, metadata: Dict):
        """Save model version with metadata."""
        model_id = metadata['model_id']
        version_dir = self.models_dir / model_id

        # Create version directory
        version_dir.mkdir(exist_ok=True)

        # Save model
        model_path = version_dir / "model.pkl"
        joblib.dump(pipeline, model_path)

        # Save metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved model version {model_id} to {version_dir}")

    def load_model_version(self, model_id: str) -> Optional[Dict]:
        """Load a specific model version."""
        version_dir = self.models_dir / model_id

        if not version_dir.exists():
            logger.error(f"Model version {model_id} not found")
            return None

        try:
            # Load model
            model_path = version_dir / "model.pkl"
            model = joblib.load(model_path)

            # Load metadata
            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.current_model = model
            logger.info(f"Loaded model version {model_id}")

            return metadata

        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None

    def list_model_versions(self) -> List[Dict]:
        """List all available model versions."""
        versions = []

        if not self.models_dir.exists():
            return versions

        for version_dir in self.models_dir.iterdir():
            if version_dir.is_dir():
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        versions.append(metadata)
                    except Exception as e:
                        logger.warning(f"Could not load metadata for {version_dir}: {e}")

        return sorted(versions, key=lambda x: x['timestamp'], reverse=True)

    def compare_model_versions(self, model_ids: List[str]) -> Dict:
        """Compare multiple model versions."""
        logger.info(f"Comparing model versions: {model_ids}")

        comparison = {
            'models': {},
            'best_model': None,
            'recommendations': []
        }

        for model_id in model_ids:
            metadata = self.load_model_version(model_id)
            if metadata:
                comparison['models'][model_id] = metadata['evaluation_results']

        if comparison['models']:
            # Find best model by accuracy
            best_id = max(
                comparison['models'].keys(),
                key=lambda x: comparison['models'][x]['test_metrics']['accuracy']
            )
            comparison['best_model'] = best_id

            # Generate recommendations
            comparison['recommendations'] = self._generate_model_recommendations(comparison)

        return comparison

    def _generate_model_recommendations(self, comparison: Dict) -> List[str]:
        """Generate recommendations based on model comparison."""
        recommendations = []

        if not comparison['models']:
            return recommendations

        best_model = comparison['best_model']
        best_metrics = comparison['models'][best_model]['test_metrics']

        recommendations.append(f"Best performing model: {best_model}")
        recommendations.append(f"Achieved accuracy: {best_metrics['accuracy']:.3f}")

        # Check for overfitting
        cv_accuracy = comparison['models'][best_model]['cross_validation']['accuracy_mean']
        test_accuracy = best_metrics['accuracy']

        if test_accuracy - cv_accuracy > 0.05:
            recommendations.append("Warning: Possible overfitting detected")
        elif cv_accuracy - test_accuracy > 0.05:
            recommendations.append("Warning: Model may not generalize well")

        # Check class balance
        per_class = comparison['models'][best_model]['per_class_metrics']
        f1_scores = [per_class[cls]['f1_score'] for cls in self.target_names]

        if max(f1_scores) - min(f1_scores) > 0.1:
            recommendations.append("Consider addressing class imbalance")

        return recommendations

    def upload_and_process_new_data(self, file_path: str, data_format: str = "auto") -> Dict:
        """
        Upload and process new exoplanet data for model updating.

        Args:
            file_path: Path to new data file
            data_format: Format of the data ('nasa_json', 'csv', 'auto')

        Returns:
            Processing results and statistics
        """
        logger.info(f"Processing new data file: {file_path}")

        try:
            # Load data based on format
            if data_format == "auto":
                data_format = self._detect_data_format(file_path)

            if data_format == "nasa_json":
                df = self.load_and_preprocess_nasa_data([file_path])
            elif data_format == "csv":
                df = pd.read_csv(file_path)
                df = self._preprocess_nasa_dataframe(df)
            else:
                raise ValueError(f"Unsupported format: {data_format}")

            # Analyze new data
            analysis = self._analyze_new_dataset(df)

            # Check compatibility with existing models
            compatibility = self._check_model_compatibility(df)

            results = {
                'file_path': file_path,
                'data_format': data_format,
                'dataset_size': df.shape,
                'feature_analysis': analysis,
                'model_compatibility': compatibility,
                'processing_timestamp': datetime.now().isoformat(),
                'ready_for_training': compatibility['compatible']
            }

            logger.info(f"Data processing completed for {file_path}")
            return results

        except Exception as e:
            logger.error(f"Error processing data file {file_path}: {e}")
            raise

    def _detect_data_format(self, file_path: str) -> str:
        """Auto-detect data format."""
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.json':
            # Check if it's NASA format
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], dict) and any('#' in str(k) for k in data[0].keys()):
                        return "nasa_json"
            except:
                pass
            return "json"
        elif file_path.suffix.lower() == '.csv':
            return "csv"
        else:
            raise ValueError(f"Cannot detect format for {file_path}")

    def _analyze_new_dataset(self, df: pd.DataFrame) -> Dict:
        """Analyze new dataset for quality and characteristics."""
        analysis = {
            'shape': df.shape,
            'missing_data': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'class_distribution': df['disposition'].value_counts().to_dict() if 'disposition' in df.columns else {},
            'numeric_features': {},
            'potential_issues': []
        }

        # Analyze numeric features
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col != 'disposition':  # Skip target
                stats_dict = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'missing_count': df[col].isnull().sum(),
                    'outliers_count': self._count_outliers(df[col])
                }
                analysis['numeric_features'][col] = stats_dict

        # Check for potential issues
        if 'disposition' in df.columns:
            class_counts = df['disposition'].value_counts()
            if class_counts.min() / class_counts.max() < 0.1:
                analysis['potential_issues'].append("Severe class imbalance detected")

        if df.isnull().sum().sum() > 0.1 * df.shape[0] * df.shape[1]:
            analysis['potential_issues'].append("High percentage of missing data")

        return analysis

    def _count_outliers(self, series: pd.Series, method: str = "iqr") -> int:
        """Count outliers using specified method."""
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return ((series < lower_bound) | (series > upper_bound)).sum()
        else:
            # Z-score method
            z_scores = np.abs((series - series.mean()) / series.std())
            return (z_scores > 3).sum()

    def _check_model_compatibility(self, df: pd.DataFrame) -> Dict:
        """Check if new data is compatible with existing models."""
        compatibility = {
            'compatible': True,
            'issues': [],
            'required_features': [],
            'missing_features': [],
            'extra_features': []
        }

        # Check if we have any trained models
        if not self.model_history:
            compatibility['issues'].append("No trained models available for compatibility check")
            return compatibility

        # Get features from latest model
        latest_model = self.model_history[-1]
        model_features = set()  # Would need to extract from actual model

        # For now, assume standard features
        expected_features = {
            'period_d', 'duration_hr', 'depth_ppm', 'snr', 'radius_re',
            'a_over_r', 'teff_k', 'logg', 'rstar_rsun', 'mag',
            'crowding', 'contamination', 'odd_even_ratio', 'secondary_depth_ppm'
        }

        available_features = set(df.columns)

        missing_features = expected_features - available_features
        extra_features = available_features - expected_features

        if missing_features:
            compatibility['missing_features'] = list(missing_features)
            compatibility['issues'].append(f"Missing features: {missing_features}")

        if extra_features:
            compatibility['extra_features'] = list(extra_features)
            # Extra features are OK, just noted

        if missing_features:
            compatibility['compatible'] = False

        return compatibility

    def retrain_model_with_new_data(self,
                                  new_data_path: str,
                                  base_model_id: str = None,
                                  retraining_config: Dict = None) -> Dict:
        """
        Retrain model with new data.

        Args:
            new_data_path: Path to new training data
            base_model_id: ID of model to use as starting point
            retraining_config: Configuration for retraining

        Returns:
            Retraining results
        """
        logger.info(f"Retraining model with new data: {new_data_path}")

        # Load new data
        df = self.load_and_preprocess_nasa_data([new_data_path])

        # Prepare features and target
        feature_columns = [col for col in df.columns if col != 'disposition']
        X = df[feature_columns].values
        y = pd.Categorical(df['disposition']).codes

        # Use latest model as base if no specific model specified
        if base_model_id is None and self.model_history:
            base_model_id = self.model_history[-1]['model_id']

        # Default retraining configuration
        if retraining_config is None:
            retraining_config = {
                'algorithm': 'RandomForest',
                'n_estimators': 100,
                'max_depth': 10,
                'learning_rate': 0.01,
                'validation_split': 0.2
            }

        # Train new model
        training_results = self.train_advanced_model(X, y, retraining_config)

        logger.info(f"Model retraining completed. New model ID: {training_results['model_id']}")

        return training_results

    def get_model_recommendations(self) -> Dict:
        """Get recommendations for model improvement."""
        if not self.model_history:
            return {"error": "No models available for recommendations"}

        latest_model = self.model_history[-1]
        metrics = latest_model['evaluation_results']

        recommendations = {
            'current_performance': metrics['test_metrics'],
            'improvement_suggestions': [],
            'research_directions': [],
            'data_quality_notes': []
        }

        # Performance-based recommendations
        accuracy = metrics['test_metrics']['accuracy']

        if accuracy < 0.90:
            recommendations['improvement_suggestions'].append(
                "Consider feature engineering and advanced algorithms"
            )
        elif accuracy < 0.95:
            recommendations['improvement_suggestions'].append(
                "Fine-tune hyperparameters and consider ensemble methods"
            )
        else:
            recommendations['improvement_suggestions'].append(
                "Model performing well - consider deployment and monitoring"
            )

        # Class balance recommendations
        per_class = metrics['per_class_metrics']
        min_f1 = min([per_class[cls]['f1_score'] for cls in self.target_names])

        if min_f1 < 0.85:
            recommendations['improvement_suggestions'].append(
                "Address class imbalance with sampling techniques or class weights"
            )

        # Research directions
        recommendations['research_directions'] = [
            "Integration of spectroscopic data for improved classification",
            "Temporal analysis for variable star discrimination",
            "Multi-planet system dynamics modeling",
            "Citizen science data integration",
            "Federated learning across multiple institutions"
        ]

        return recommendations

def main():
    """Main function for testing the model manager."""
    print("Exoplanet Model Manager - Advanced Training System")
    print("=" * 60)

    # Initialize model manager
    manager = ExoplanetModelManager()

    # Example: Load and process NASA data
    data_files = [
        "k2pandc_2025.10.04_07.10.02.json",
        "TOI_2025.10.04_07.06.07.json"
    ]

    try:
        # Load and preprocess data
        print("\n1. Loading NASA exoplanet data...")
        df = manager.load_and_preprocess_nasa_data(data_files)

        # Prepare for training
        feature_columns = [col for col in df.columns if col != 'disposition']
        X = df[feature_columns].values
        y = pd.Categorical(df['disposition']).codes

        print(f"Dataset shape: {X.shape}")
        print(f"Classes: {manager.target_names}")
        print(f"Class distribution: {np.bincount(y)}")

        # Train advanced model
        print("\n2. Training advanced model...")
        model_config = {
            'algorithm': 'RandomForest',
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }

        results = manager.train_advanced_model(X, y, model_config)

        print(f"Training completed! Model ID: {results['model_id']}")
        print(f"Test accuracy: {results['evaluation_results']['test_metrics']['accuracy']:.3f}")

        # Get recommendations
        print("\n3. Getting model recommendations...")
        recommendations = manager.get_model_recommendations()
        print("Recommendations:", recommendations['improvement_suggestions'])

        # List all models
        print("\n4. Available model versions:")
        versions = manager.list_model_versions()
        for version in versions[:3]:  # Show latest 3
            print(f"  - {version['model_id']}: {version['evaluation_results']['test_metrics']['accuracy']:.3f} accuracy")

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()