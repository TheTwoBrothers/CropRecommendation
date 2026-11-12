from fastapi import FastAPI, HTTPException

from app.schemas.prediction import CropPredictionRequest

from app.services.crop_recommendation_service import (
    CropRecommendationService
)


app = FastAPI(
    title="Smart Crop Recommendation System",
    description="API for recommending suitable crops.",
    version="1.0.0"
)


# Create the service object once
crop_service = CropRecommendationService()


@app.get("/")
def home():

    return {
        "message": "Welcome to Smart Crop Recommendation System"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


@app.post("/predict_crop")
def predict_crop(data: CropPredictionRequest):

    input_features = {
        "SOIL": data.soil,
        "SEASON": data.season,
        "WATER_SOURCE": data.water_source,
        "TEMP": data.temperature_c,
        "MAX_TEMP": data.max_temperature_c,
        "RELATIVE_HUMIDITY": data.relative_humidity_pct,
        "RELATIVE_HUMIDITY_MAX": data.max_relative_humidity_pct
    }

    try:
        result = crop_service.recommend(input_features)

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )