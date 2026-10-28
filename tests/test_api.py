from fastapi.testclient import TestClient
from app.main import app


def test_home():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "running"


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True


def test_prediction():
    passenger = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100,
        "Embarked": "S"
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=passenger)
        assert response.status_code == 200

        result = response.json()
        assert result["prediction"] in [0, 1]
        assert isinstance(result["survived"], bool)
        assert 0 <= result["survival_probability"] <= 1


def test_invalid_class():
    passenger = {
        "Pclass": 5,
        "Sex": "female",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100,
        "Embarked": "S"
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=passenger)
        assert response.status_code == 422


def test_invalid_gender():
    passenger = {
        "Pclass": 1,
        "Sex": "unknown",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100,
        "Embarked": "S"
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=passenger)
        assert response.status_code == 422


def test_missing_values():
    passenger = {
        "Pclass": 2,
        "Sex": "female",
        "SibSp": 0,
        "Parch": 0
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=passenger)
        assert response.status_code == 200
