"""
This module loads a pre-trained PyMC3 model and defines a web
service with three prediction endpoints (point, interval and density),
using FastAPI and Pydantic.
"""
import logging
import sys
from typing import Tuple

import arviz as az
import boto3 as aws
import joblib
import numpy as np
import pymc3 as pm
import uvicorn
from botocore.exceptions import ClientError
from fastapi import FastAPI
from pydantic import BaseModel, Field

AWS_S3_BUCKET_NAME = "bodywork-pymc3-project"
MODEL_DEFINITION_BUCKET_PATH = "models/pymc.joblib"
INFERENCE_DATA_BUCKET_PATH = "inference_data/pymc.nc"
N_PREDICTION_SAMPLES = 100

app = FastAPI(debug=False)


class FeatureDataInstance(BaseModel):
    """Pydantic schema for instances of feature data."""

    x: float
    category: int


class AlgoParam(BaseModel):
    """Pydantic schema for algorithm config."""

    n_samples: int = Field(N_PREDICTION_SAMPLES, gt=0)


class PointPredictionRequest(BaseModel):
    """Pydantic schema for point-estimate requests."""

    data: FeatureDataInstance
    algo_param: AlgoParam = AlgoParam()


class IntervalPredictionRequest(BaseModel):
    """Pydantic schema for interval requests."""

    data: FeatureDataInstance
    hdi_probability: float = Field(0.95, gt=0, lt=1)
    algo_param: AlgoParam = AlgoParam()


class DensityPredictionRequest(BaseModel):
    """Pydantic schema for density requests."""

    data: FeatureDataInstance
    bins: int = Field(5, gt=0)
    algo_param: AlgoParam = AlgoParam()


@app.post("/predict/v1.0.0/point", status_code=200)
def predict_point_estimate(request: PointPredictionRequest):
    """Return point-estimate prediction."""
    y_pred_samples = generate_label_samples(
        request.data.x, request.data.category, request.algo_param.n_samples
    )
    y_pred = np.median(y_pred_samples)
    return {"y_pred": y_pred, "algo_param": request.algo_param.n_samples}


@app.post("/predict/v1.0.0/interval", status_code=200)
def predict_interval(request: IntervalPredictionRequest):
    """Return point-estimate prediction."""
    y_pred_samples = generate_label_samples(
        request.data.x, request.data.category, request.algo_param.n_samples
    )
    y_hdi = pm.hdi(y_pred_samples, request.hdi_probability)
    return {
        "y_pred_lower": y_hdi[0],
        "y_pred_upper": y_hdi[1],
        "algo_param": request.algo_param.n_samples,
    }


@app.post("/predict/v1.0.0/density", status_code=200)
def predict_density(request: DensityPredictionRequest):
    """Return density prediction."""
    y_pred_samples = generate_label_samples(
        request.data.x, request.data.category, request.algo_param.n_samples
    )
    y_pred_density, bin_edges = np.histogram(
        y_pred_samples, bins=request.bins, density=True
    )
    bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return {
        "y_pred_bin_mids": bin_mids.tolist(),
        "y_pred_density": y_pred_density.tolist(),
        "algo_param": request.algo_param.n_samples,
    }


def generate_label_samples(x: float, category: int, n_samples) -> np.ndarray:
    """Sample posterior predictve distribution given feature data."""
    pm.set_data({"y": [0], "x": [x], "category": [category]}, model=model)
    posterior_pred = pm.sample_posterior_predictive(
        inference_data.posterior,
        model=model,
        samples=n_samples,
        random_seed=42,
        progressbar=False,
    )
    return posterior_pred["obs"].reshape(-1)


def configure_logger() -> logging.Logger:
    """Configure a logger that will write to stdout."""
    log_handler = logging.StreamHandler(sys.stdout)
    log_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s"
    )
    log_handler.setFormatter(log_format)
    log = logging.getLogger(__name__)
    log.addHandler(log_handler)
    log.setLevel(logging.INFO)
    return log


def get_model_artefacts() -> Tuple[pm.Model, az.InferenceData]:
    """Get model definition and inference data from AWS S3 bucket."""
    try:
        s3_client = aws.client("s3")
        s3_client.download_file(
            AWS_S3_BUCKET_NAME, MODEL_DEFINITION_BUCKET_PATH, "model.joblib"
        )
        s3_client.download_file(
            AWS_S3_BUCKET_NAME, INFERENCE_DATA_BUCKET_PATH, "inference_data.nc"
        )
        model = joblib.load("model.joblib")
        inference_data = az.from_netcdf("inference_data.nc")
    except ClientError as e:
        log.error(e)
        raise RuntimeError(f"failed to get model files from s3://{AWS_S3_BUCKET_NAME}")
    except Exception as e:
        log.error(e)
        raise RuntimeError("could not load model data")
    return (model, inference_data)


if __name__ == "__main__":
    log = configure_logger()
    model, inference_data = get_model_artefacts()
    uvicorn.run(app, host="0.0.0.0", workers=1)
