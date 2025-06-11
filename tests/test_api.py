import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.server import app, get_db
from monitoring.database import Base

# Setup the Test Database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override the get_db dependency to use the test database
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Create a TestClient
client = TestClient(app)

@pytest.fixture(scope="function")
def db_session():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_monitor_inference(db_session):
    response = client.post(
        "/monitor/inference",
        json={
            "prompt": "Test prompt",
            "response": "Test response",
            "model_name": "test-model",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["quality_score"] is not None
    assert data["safety_score"] is not None
    assert data["cost_usd"] is not None

def test_get_quality_metrics_no_data(db_session):
    response = client.get("/metrics/quality")
    assert response.status_code == 200
    assert response.json()["message"] == "No data available for this time period."

def test_get_quality_metrics_with_data(db_session):
    client.post(
        "/monitor/inference",
        json={"prompt": "p", "response": "r", "model_name": "m"},
    )
    response = client.get("/metrics/quality")
    assert response.status_code == 200
    data = response.json()
    assert data["total_evaluations"] == 1
    assert data["average_quality"] > 0

def test_get_safety_metrics_with_data(db_session):
    client.post(
        "/monitor/inference",
        json={"prompt": "p", "response": "r", "model_name": "m"},
    )
    response = client.get("/metrics/safety")
    assert response.status_code == 200
    data = response.json()
    assert data["total_interactions"] == 1
    assert data["safety_violations"] == 0

def test_batch_evaluate(db_session):
    response = client.post(
        "/evaluate",
        json=[
            {"prompt": "p1", "response": "r1", "model_name": "m1"},
            {"prompt": "p2", "response": "r2", "model_name": "m2"},
        ],
    )
    assert response.status_code == 200
    data = response.json()
    assert data["evaluated_count"] == 2
    assert len(data["results"]) == 2 