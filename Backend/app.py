"""
FastAPI application for exoplanet classification prediction service.

This service loads pretrained models from artifacts directory and provides
prediction endpoints for exoplanet classification with detailed analysis.
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================

import os
import json
import joblib
import logging
import warnings
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
import uvicorn
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE VALIDATION
# =============================================================================

class PredictionRequest(BaseModel):
    """
    Request model for exoplanet prediction.

    Accepts features for one or more exoplanet candidates.
    All features are optional to allow flexible input.
    """

    # Orbital and transit features
    period_d: Optional[float] = Field(None, description="Orbital period in days", ge=0)
    duration_hr: Optional[float] = Field(None, description="Transit duration in hours", ge=0)
    depth_ppm: Optional[float] = Field(None, description="Transit depth in parts per million", ge=0)

    # Signal quality features
    snr: Optional[float] = Field(None, description="Signal-to-noise ratio", ge=0)
    odd_even_ratio: Optional[float] = Field(None, description="Odd-even transit depth ratio")

    # Physical properties
    radius_re: Optional[float] = Field(None, description="Planet radius in Earth radii", ge=0)
    a_over_r: Optional[float] = Field(None, description="Semi-major axis to stellar radius ratio", ge=0)

    # Stellar properties
    teff_k: Optional[float] = Field(None, description="Effective temperature in Kelvin", ge=0)
    logg: Optional[float] = Field(None, description="Surface gravity (log g)", ge=0)
    rstar_rsun: Optional[float] = Field(None, description="Stellar radius in solar radii", ge=0)
    mag: Optional[float] = Field(None, description="Apparent magnitude")

    # Data quality features
    crowding: Optional[float] = Field(None, description="Crowding metric", ge=0, le=1)
    contamination: Optional[float] = Field(None, description="Contamination factor", ge=0, le=1)
    secondary_depth_ppm: Optional[float] = Field(None, description="Secondary eclipse depth")

    # Mission metadata
    mission: Optional[str] = Field(None, description="Mission name (Kepler, TESS, etc.)")

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "period_d": 10.5,
                "duration_hr": 2.5,
                "depth_ppm": 100.0,
                "snr": 15.0,
                "radius_re": 2.5,
                "a_over_r": 20.0,
                "teff_k": 5500,
                "logg": 4.5,
                "rstar_rsun": 1.0,
                "mag": 12.0,
                "crowding": 0.1,
                "contamination": 0.05,
                "odd_even_ratio": 1.0,
                "secondary_depth_ppm": 10.0,
                "mission": "Kepler"
            }
        }

class FeatureContribution(BaseModel):
    """
    Model representing a feature's contribution to prediction.
    """
    feature: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="Absolute contribution value")
    sign: str = Field(..., description="Direction of contribution (+ or -)")
    reason: str = Field(..., description="Explanation of feature impact")

class PredictionResponse(BaseModel):
    """
    Response model for exoplanet prediction.
    """
    prediction: str = Field(..., description="Predicted class (CONFIRMED/CANDIDATE/FALSE_POSITIVE)")
    probabilities: Dict[str, float] = Field(..., description="Prediction probabilities for each class")
    feature_drivers: List[FeatureContribution] = Field(..., description="Top contributing features")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")

class MetricsResponse(BaseModel):
    """
    Response model for model metrics.
    """
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="F1 score")
    pr_auc: float = Field(..., description="Precision-Recall AUC")
    calibration_score: float = Field(..., description="Model calibration score")
    total_samples: int = Field(..., description="Total samples in training data")
    model_version: str = Field(..., description="Model version/timestamp")

class FeatureSchema(BaseModel):
    """
    Schema definition for a feature.
    """
    name: str = Field(..., description="Feature name")
    units: str = Field(..., description="Feature units")
    min_range: Optional[float] = Field(None, description="Minimum expected value")
    max_range: Optional[float] = Field(None, description="Maximum expected value")
    description: str = Field(..., description="Feature description")

class SchemaResponse(BaseModel):
    """
    Response model for feature schema.
    """
    features: List[FeatureSchema] = Field(..., description="List of feature definitions")
    last_updated: str = Field(..., description="Schema last update timestamp")

# =============================================================================
# EXOPLANET PREDICTION SERVICE CLASS
# =============================================================================

class ExoplanetPredictionService:
    """
    Service class for exoplanet classification predictions.

    Handles model loading, prediction, and feature analysis.
    """

    def __init__(self, artifacts_dir: str = "./artifacts"):
        """
        Initialize the prediction service.

        Args:
            artifacts_dir: Directory containing model artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.model = None
        self.pipeline = None
        self.feature_names = None
        self.target_names = None
        self.metrics = None
        self.schema = None

        # Load models and artifacts
        self._load_artifacts()

        logger.info("ExoplanetPredictionService initialized successfully")

    def _load_artifacts(self):
        """
        Load pretrained models and artifacts from artifacts directory.
        """
        try:
            # Load the main model
            model_path = self.artifacts_dir / "model.pkl"
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"Loaded model from {model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load preprocessing pipeline
            pipeline_path = self.artifacts_dir / "pipeline.pkl"
            if pipeline_path.exists():
                self.pipeline = joblib.load(pipeline_path)
                logger.info(f"Loaded pipeline from {pipeline_path}")
            else:
                logger.warning(f"Pipeline file not found: {pipeline_path}")

            # Load feature names
            features_path = self.artifacts_dir / "feature_names.pkl"
            if features_path.exists():
                self.feature_names = joblib.load(features_path)
                logger.info(f"Loaded feature names: {len(self.feature_names)} features")
            else:
                logger.warning(f"Feature names file not found: {features_path}")

            # Load target names
            targets_path = self.artifacts_dir / "target_names.pkl"
            if targets_path.exists():
                self.target_names = joblib.load(targets_path)
                logger.info(f"Loaded target names: {self.target_names}")
            else:
                # Default target names if not found
                self.target_names = ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]
                logger.warning(f"Target names file not found, using defaults: {self.target_names}")

            # Load metrics
            metrics_path = self.artifacts_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
                logger.info(f"Loaded metrics from {metrics_path}")
            else:
                logger.warning(f"Metrics file not found: {metrics_path}")

            # Load feature schema
            schema_path = self.artifacts_dir / "schema.json"
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    schema_data = json.load(f)
                self.schema = schema_data
                logger.info(f"Loaded schema from {schema_path}")
            else:
                logger.warning(f"Schema file not found: {schema_path}")

        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model artifacts: {str(e)}"
            )

    def _validate_input_features(self, features_dict: Dict[str, Any]) -> List[str]:
        """
        Validate input features and return warnings.

        Args:
            features_dict: Dictionary of input features

        Returns:
            List of validation warnings
        """
        warnings_list = []

        if not self.schema:
            return warnings_list

        features_schema = self.schema.get('features', [])

        for feature_info in features_schema:
            feature_name = feature_info['name']
            feature_value = features_dict.get(feature_name)

            if feature_value is not None:
                # Check range if specified
                min_val = feature_info.get('min_range')
                max_val = feature_info.get('max_range')

                if min_val is not None and feature_value < min_val:
                    warnings_list.append(
                        f"{feature_name} ({feature_value}) below expected range [{min_val}, {max_val or 'inf'}]"
                    )

                if max_val is not None and feature_value > max_val:
                    warnings_list.append(
                        f"{feature_name} ({feature_value}) above expected range [{min_val or 0}, {max_val}]"
                    )

        return warnings_list

    def _calculate_feature_contributions(self, features_array: np.ndarray,
                                       prediction_proba: np.ndarray) -> List[FeatureContribution]:
        """
        Calculate feature contributions to the prediction.

        Args:
            features_array: Array of input features
            prediction_proba: Prediction probabilities

        Returns:
            List of feature contributions
        """
        contributions = []

        if not self.model or not hasattr(self.model, 'feature_importances_'):
            # Fallback: use simple correlation-based contributions
            return self._calculate_simple_contributions(features_array, prediction_proba)

        # Use tree-based feature importance
        feature_importance = self.model.feature_importances_

        # Get top contributing features
        top_indices = np.argsort(feature_importance)[-10:][::-1]  # Top 10

        for idx in top_indices:
            if idx < len(self.feature_names):
                feature_name = self.feature_names[idx]
                importance = feature_importance[idx]

                # Determine sign based on feature value and prediction
                feature_value = features_array[idx]
                contribution_value = importance * feature_value

                # Simple heuristic for sign determination
                if contribution_value > 0:
                    sign = "+"
                    reason = "Positive contribution to prediction"
                else:
                    sign = "-"
                    reason = "Negative contribution to prediction"

                contributions.append(FeatureContribution(
                    feature=feature_name,
                    contribution=abs(contribution_value),
                    sign=sign,
                    reason=reason
                ))

        return contributions[:5]  # Return top 5

    def _calculate_simple_contributions(self, features_array: np.ndarray,
                                     prediction_proba: np.ndarray) -> List[FeatureContribution]:
        """
        Calculate simple correlation-based feature contributions.

        Args:
            features_array: Array of input features
            prediction_proba: Prediction probabilities

        Returns:
            List of feature contributions
        """
        contributions = []

        # Simple heuristic based on feature values and probabilities
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(features_array))]

        for i, (feature_name, feature_value) in enumerate(zip(feature_names, features_array)):
            if feature_value != 0 and not np.isnan(feature_value):
                # Simple contribution based on feature magnitude
                contribution = abs(feature_value) * 0.1  # Scale factor

                contributions.append(FeatureContribution(
                    feature=feature_name,
                    contribution=contribution,
                    sign="+" if feature_value > 0 else "-",
                    reason="Feature magnitude contribution"
                ))

        return sorted(contributions, key=lambda x: x.contribution, reverse=True)[:5]

    def predict_single(self, features: Dict[str, Any]) -> PredictionResponse:
        """
        Make prediction for a single set of features.

        Args:
            features: Dictionary of input features

        Returns:
            PredictionResponse with prediction results
        """
        # Validate input features
        warnings_list = self._validate_input_features(features)

        # Convert to DataFrame for processing
        df = pd.DataFrame([features])

        # Apply preprocessing pipeline if available
        if self.pipeline:
            try:
                df_processed = self.pipeline.transform(df)
            except Exception as e:
                logger.warning(f"Pipeline processing failed: {e}")
                df_processed = df.values
        else:
            df_processed = df.values

        # Make prediction
        if self.model:
            try:
                # Get prediction probabilities
                probabilities = self.model.predict_proba(df_processed)[0]

                # Get predicted class
                predicted_class_idx = np.argmax(probabilities)
                predicted_class = self.target_names[predicted_class_idx] if self.target_names else str(predicted_class_idx)

                # Calculate feature contributions
                feature_drivers = self._calculate_feature_contributions(
                    df_processed[0], probabilities
                )

                # Create response
                response = PredictionResponse(
                    prediction=predicted_class,
                    probabilities={
                        self.target_names[i] if self.target_names else str(i): float(prob)
                        for i, prob in enumerate(probabilities)
                    },
                    feature_drivers=feature_drivers,
                    warnings=warnings_list
                )

                return response

            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Prediction failed: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model not loaded"
            )

    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[PredictionResponse]:
        """
        Make predictions for multiple sets of features.

        Args:
            features_list: List of feature dictionaries

        Returns:
            List of PredictionResponse objects
        """
        responses = []

        for features in features_list:
            response = self.predict_single(features)
            responses.append(response)

        return responses

# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

# Create FastAPI application
app = FastAPI(
    title="Exoplanet Classification API",
    description="API for classifying exoplanet candidates using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
prediction_service = None

# =============================================================================
# DEPENDENCY FUNCTIONS
# =============================================================================

def get_prediction_service() -> ExoplanetPredictionService:
    """
    Dependency function to get prediction service instance.
    """
    global prediction_service
    if prediction_service is None:
        prediction_service = ExoplanetPredictionService()
    return prediction_service

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/predict", response_model=Union[PredictionResponse, List[PredictionResponse]])
async def predict(
    request: Union[PredictionRequest, List[PredictionRequest]],
    service: ExoplanetPredictionService = Depends(get_prediction_service)
):
    """
    Make exoplanet classification predictions.

    Accepts single or multiple feature sets and returns predictions with
    probabilities and feature contributions.
    """
    try:
        if isinstance(request, list):
            # Batch prediction
            features_list = [item.dict(exclude_unset=True) for item in request]
            responses = service.predict_batch(features_list)
            return responses
        else:
            # Single prediction
            features = request.dict(exclude_unset=True)
            response = service.predict_single(features)
            return response

    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    service: ExoplanetPredictionService = Depends(get_prediction_service)
):
    """
    Get model performance metrics.

    Returns comprehensive model evaluation metrics including accuracy,
    precision, recall, F1-score, and calibration metrics.
    """
    if not service.metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics data not available"
        )

    return MetricsResponse(**service.metrics)

@app.get("/schema", response_model=SchemaResponse)
async def get_schema(
    service: ExoplanetPredictionService = Depends(get_prediction_service)
):
    """
    Get feature schema information.

    Returns the canonical list of features with units, expected ranges,
    and descriptions for proper input validation.
    """
    if not service.schema:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schema data not available"
        )

    return SchemaResponse(**service.schema)

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "Exoplanet Classification API",
        "version": "1.0.0",
        "description": "Machine learning API for exoplanet candidate classification",
        "endpoints": {
            "predict": "POST /predict - Make predictions",
            "metrics": "GET /metrics - Get model metrics",
            "schema": "GET /schema - Get feature schema",
            "docs": "GET /docs - API documentation"
        }
    }

@app.get("/health")
async def health_check(
    service: ExoplanetPredictionService = Depends(get_prediction_service)
):
    """
    Health check endpoint.

    Verifies that the service and models are loaded correctly.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": service.model is not None,
        "pipeline_loaded": service.pipeline is not None,
        "features_count": len(service.feature_names) if service.feature_names else 0
    }

# =============================================================================
# MAIN APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Main entry point for running the FastAPI application.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Exoplanet Classification API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    logger.info(f"Starting Exoplanet Classification API on {args.host}:{args.port}")

    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )