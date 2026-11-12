from pydantic import BaseModel, ConfigDict, Field


class CropPredictionRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    soil: str = Field(min_length=1)

    season: str = Field(min_length=1)

    water_source: str = Field(min_length=1)

    temperature_c: float = Field(ge=-10, le=60)

    max_temperature_c: float = Field(ge=-10, le=60)

    relative_humidity_pct: float = Field(ge=0, le=100)

    max_relative_humidity_pct: float = Field(ge=0, le=100)


class CropCandidate(BaseModel):

    rank: int

    crop: str

    model_score: float


class CropPredictionResponse(BaseModel):

    recommendations: list[CropCandidate]

    warnings: list[str]