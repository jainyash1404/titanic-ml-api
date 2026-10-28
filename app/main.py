from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ConfigDict


# -----------------------------------
# 1. MODEL PATH
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model.joblib"


# -----------------------------------
# 2. LOAD MODEL
# -----------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}"
        )

    app.state.model = joblib.load(MODEL_PATH)

    print("ML Model loaded successfully!")

    yield


# -----------------------------------
# 3. CREATE FASTAPI APP
# -----------------------------------

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict Titanic passenger survival using Machine Learning",
    version="1.0.0",
    lifespan=lifespan
)


# -----------------------------------
# 4. INPUT SCHEMA
# -----------------------------------

class PassengerInput(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    Pclass: int = Field(
        ge=1,
        le=3,
        description="Passenger class: 1, 2 or 3"
    )

    Sex: str = Field(
        description="male or female"
    )

    Age: float | None = Field(
        default=None,
        ge=0,
        le=120
    )

    SibSp: int = Field(
        ge=0
    )

    Parch: int = Field(
        ge=0
    )

    Fare: float | None = Field(
        default=None,
        ge=0
    )

    Embarked: str | None = Field(
        default=None,
        description="S, C or Q"
    )


# -----------------------------------
# 5. HOME ENDPOINT
# -----------------------------------

@app.get("/")
def home():

    return {
        "message": "Welcome to Titanic Survival Prediction API",
        "status": "running",
        "documentation": "/docs"
    }


# -----------------------------------
# 6. HEALTH CHECK
# -----------------------------------

@app.get("/health")
def health(request: Request):

    model_loaded = hasattr(
        request.app.state,
        "model"
    )

    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }


# -----------------------------------
# 7. PREDICTION ENDPOINT
# -----------------------------------

@app.post("/predict")
def predict(
    passenger: PassengerInput,
    request: Request
):

    try:

        model = request.app.state.model

        # Validate categorical inputs
        if passenger.Sex.lower() not in ["male", "female"]:
            raise HTTPException(
                status_code=422,
                detail="Sex must be male or female"
            )

        if (
            passenger.Embarked is not None
            and passenger.Embarked.upper() not in ["S", "C", "Q"]
        ):
            raise HTTPException(
                status_code=422,
                detail="Embarked must be S, C or Q"
            )

        # Convert request into DataFrame
        passenger_data = pd.DataFrame([{
            "Pclass": passenger.Pclass,
            "Sex": passenger.Sex.lower(),
            "Age": passenger.Age,
            "SibSp": passenger.SibSp,
            "Parch": passenger.Parch,
            "Fare": passenger.Fare,
            "Embarked": (
                passenger.Embarked.upper()
                if passenger.Embarked is not None
                else None
            )
        }])

        # Predict class
        prediction = int(
            model.predict(passenger_data)[0]
        )

        # Predict probability
        probability = float(
            model.predict_proba(passenger_data)[0][1]
        )

        return {
            "prediction": prediction,
            "survived": bool(prediction),
            "survival_probability": round(probability, 4),
            "message": (
                "Passenger is predicted to survive"
                if prediction == 1
                else "Passenger is predicted not to survive"
            )
        }

    except HTTPException:
        raise

    except Exception as error:

        print("Prediction error:", str(error))

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from error