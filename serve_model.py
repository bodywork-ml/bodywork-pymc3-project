"""
This module loads a pre-trained SciKit-Learn model and defines a web
service using FastAPI to serve predictions.
"""
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(debug=False)


class FeatureDataInstance(BaseModel):
    """Define JSON data schema for prediction requests."""
    X: float


@app.post('/api/v1/', status_code=200)
def predict(data: FeatureDataInstance):
    """Generate predictions for data sent to the /api/v1/ route."""
    prediction = model.predict([data.X])
    return {'y_pred': prediction[0]}


if __name__ == '__main__':
    model = joblib.load('dummy_model.joblib')
    uvicorn.run(app, host='0.0.0.0', workers=1)
