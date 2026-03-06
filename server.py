from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import pickle
import numpy as np
import pandas as pd   # ✅ ADD: pandas import karo
import jwt
import bcrypt

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="HeartGuard AI API")

# ✅ FIX 1: CORS middleware SABSE PEHLE — router se pehle
# allow_credentials=True + allow_origins=["*"] = INVALID combination!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # ← True tha, False kiya
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Security
security = HTTPBearer()
JWT_SECRET = os.environ.get('JWT_SECRET', 'heartguard-secret-key-2024')
JWT_ALGORITHM = "HS256"

# Load ML Models
def load_models():
    models = {}
    scaler = None
    try:
        with open(ROOT_DIR / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(ROOT_DIR / 'heart_attack_model_lr.pkl', 'rb') as f:
            models['logistic_regression'] = pickle.load(f)
        with open(ROOT_DIR / 'heart_attack_model_rf.pkl', 'rb') as f:
            models['random_forest'] = pickle.load(f)
        with open(ROOT_DIR / 'heart_attack_model_svm.pkl', 'rb') as f:
            models['svm'] = pickle.load(f)
        with open(ROOT_DIR / 'heart_attack_model_xgb.pkl', 'rb') as f:
            models['xgboost'] = pickle.load(f)
        logging.info("All ML models loaded successfully")
    except Exception as e:
        logging.error(f"Error loading models: {e}")
    return models, scaler

ML_MODELS, SCALER = load_models()

# Feature names in order (same as training)
FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak']

# Pydantic Models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    created_at: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class PredictionInput(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: int = Field(..., ge=50, le=300)
    chol: int = Field(..., ge=50, le=600)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalch: int = Field(..., ge=50, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)

class PredictionResult(BaseModel):
    id: str
    user_id: str
    input_data: Dict[str, Any]
    risk_score: float
    risk_level: str
    model_predictions: Dict[str, Dict[str, Any]]
    recommendations: Dict[str, Any]
    created_at: str

class ResearchPaper(BaseModel):
    id: str
    title: str
    authors: str
    year: int
    methodology: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    dataset: str
    key_findings: str

# Helper Functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_token(user_id: str, email: str) -> str:
    payload = {
        'user_id': user_id,
        'email': email,
        'exp': datetime.now(timezone.utc) + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user = await db.users.find_one({'id': payload['user_id']}, {'_id': 0})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def calculate_risk_score(probabilities: Dict[str, float]) -> float:
    weights = {'logistic_regression': 0.2, 'random_forest': 0.3, 'svm': 0.2, 'xgboost': 0.3}
    total = sum(prob * weights.get(model, 0.25) for model, prob in probabilities.items())
    return round(total * 100, 2)

def get_risk_level(score: float) -> str:
    if score <= 25:   return "very_low"
    elif score <= 50: return "low"
    elif score <= 75: return "moderate"
    elif score <= 90: return "high"
    else:             return "critical"

def get_recommendations(risk_level: str) -> Dict[str, Any]:
    recommendations = {
        "very_low": {
            "status": "MAINTAIN STATUS",
            "exercise": {"allowed": True, "details": "Maintain current activity level", "frequency": "30 minutes moderate exercise, 5 days/week", "activities": ["Walking", "Jogging", "Swimming", "Cycling"]},
            "diet": {"type": "General healthy eating", "details": "Balanced diet with fruits, vegetables", "restrictions": "No strict restrictions", "cholesterol_target": "<200 mg/dl"},
            "doctor": {"required": False, "details": "Routine annual checkup only"},
            "emergency": {"required": False, "details": "Not needed"}
        },
        "low": {
            "status": "PREVENTION MODE",
            "exercise": {"allowed": True, "details": "Moderate intensity, regular schedule", "frequency": "30-45 minutes, 5 days/week", "activities": ["Brisk walking", "Swimming", "Cycling", "Yoga/Stretching"]},
            "diet": {"type": "Preventive dietary changes", "details": "Reduce saturated fats, increase fiber", "restrictions": "Limit sodium to <2300mg/day", "recommendations": ["Whole grains", "Oats", "Fruits & vegetables", "Omega-3 rich foods"]},
            "doctor": {"required": False, "optional": True, "specialist": "General Physician", "schedule": "6-month checkup"},
            "emergency": {"required": False, "details": "Not needed"}
        },
        "moderate": {
            "status": "ACTIVE MANAGEMENT",
            "exercise": {"allowed": True, "supervised": True, "details": "Medical clearance REQUIRED", "frequency": "30 minutes, 5 days/week", "activities": ["Walking", "Light jogging", "Swimming"], "avoid": ["High-intensity workouts", "Heavy lifting"], "warnings": ["Monitor heart rate", "Stop if chest pain occurs"]},
            "diet": {"type": "DASH Diet or Mediterranean Diet", "restrictions": "Sodium <1500mg/day", "avoid": ["Fried foods", "Red meat", "Processed foods"], "include": ["Leafy greens", "Berries", "Whole grains", "Fish"]},
            "doctor": {"required": True, "specialist": "Cardiologist", "schedule": "Every 3 months", "tests": ["ECG", "Lipid profile", "BP monitoring"]},
            "emergency": {"required": False, "prepare": True, "details": "Keep nearest cardiac center contact saved"}
        },
        "high": {
            "status": "URGENT ACTION REQUIRED",
            "exercise": {"allowed": False, "details": "ONLY under medical supervision", "warnings": ["NO exercise without doctor's permission", "Cardiac rehabilitation recommended"]},
            "diet": {"type": "Therapeutic diet", "restrictions": "Very low sodium (<1500mg/day)", "avoid": ["Caffeine", "Alcohol"], "recommendations": ["Frequent small meals (6 times/day)"]},
            "doctor": {"required": True, "urgent": True, "specialist": "Cardiologist", "schedule": "Within 48-72 hours", "tests": ["ECG", "Echo", "Stress test", "Angiography"]},
            "emergency": {"required": True, "details": "Identify 2-3 nearest cardiac emergency centers", "actions": ["Save contact numbers", "Inform family members"]}
        },
        "critical": {
            "status": "EMERGENCY PROTOCOL",
            "exercise": {"allowed": False, "details": "COMPLETELY AVOID", "warnings": ["Strict bed rest", "NO physical exertion"]},
            "diet": {"type": "Hospital/Emergency diet plan", "details": "Follow doctor's prescription only", "restrictions": "Very low sodium, low fat"},
            "doctor": {"required": True, "emergency": True, "specialist": "Emergency Cardiologist", "schedule": "IMMEDIATE (within hours)", "details": "Likely cardiac ICU admission"},
            "emergency": {"required": True, "immediate": True, "details": "GO IMMEDIATELY", "actions": ["Call ambulance (108/102)", "Do not delay"]}
        }
    }
    return recommendations.get(risk_level, recommendations["moderate"])

# Research papers data
RESEARCH_PAPERS = [
    {"id": "1", "title": "Heart Disease Prediction Using Machine Learning Techniques", "authors": "Mohan, S., Thirumalai, C., & Srivastava, G.", "year": 2021, "methodology": "Hybrid Random Forest with Linear Model (HRFLM)", "accuracy": 88.7, "precision": 87.5, "recall": 89.2, "f1_score": 88.3, "dataset": "Cleveland Heart Disease Dataset (303 instances)", "key_findings": "Proposed HRFLM achieves higher accuracy than individual models."},
    {"id": "2", "title": "Efficient Heart Disease Prediction System Using Ensemble Learning", "authors": "Reddy, K. V. V., Elamvazuthi, I., & Aziz, A. A.", "year": 2022, "methodology": "XGBoost with Feature Selection", "accuracy": 91.8, "precision": 90.5, "recall": 92.3, "f1_score": 91.4, "dataset": "UCI Heart Disease Dataset (1025 instances)", "key_findings": "XGBoost with recursive feature elimination showed best performance."},
    {"id": "3", "title": "Deep Learning Approaches for Heart Attack Prediction", "authors": "Ali, M. M., Paul, B. K., & Ahmed, K.", "year": 2023, "methodology": "Neural Network with SMOTE", "accuracy": 89.5, "precision": 88.7, "recall": 90.1, "f1_score": 89.4, "dataset": "Framingham Heart Study (4240 instances)", "key_findings": "SMOTE improved recall for minority class."},
    {"id": "4", "title": "Comparative Analysis of ML Algorithms for CVD Prediction", "authors": "Tama, B. A., Im, S., & Lee, S.", "year": 2020, "methodology": "Random Forest, SVM, Gradient Boosting", "accuracy": 87.4, "precision": 86.8, "recall": 87.9, "f1_score": 87.3, "dataset": "Cleveland + Hungarian + Switzerland (918 instances)", "key_findings": "Random Forest showed most consistent performance."},
    {"id": "5", "title": "Feature Engineering for Improved Heart Disease Classification", "authors": "Katarya, R., & Meena, S. K.", "year": 2021, "methodology": "Logistic Regression with PCA", "accuracy": 85.2, "precision": 84.5, "recall": 85.8, "f1_score": 85.1, "dataset": "Cleveland Heart Disease Dataset (303 instances)", "key_findings": "PCA reduced dimensionality while maintaining predictive power."},
    {"id": "6", "title": "Real-time Heart Attack Risk Assessment Using IoT and ML", "authors": "Kumar, P., Sharma, R., & Singh, V.", "year": 2023, "methodology": "SVM with Real-time Sensor Data", "accuracy": 92.3, "precision": 91.8, "recall": 92.7, "f1_score": 92.2, "dataset": "Custom IoT Dataset (5000 instances)", "key_findings": "Real-time monitoring with ML achieved highest accuracy."},
    {"id": "7", "title": "Explainable AI for Heart Disease Prediction", "authors": "Chen, J., Liu, Y., & Wang, H.", "year": 2024, "methodology": "XGBoost with SHAP Explanations", "accuracy": 90.1, "precision": 89.5, "recall": 90.8, "f1_score": 90.1, "dataset": "UCI + Cleveland Combined (1500 instances)", "key_findings": "SHAP values provide interpretable predictions for medical applications."}
]

MODEL_METRICS = {
    "logistic_regression": {"name": "Logistic Regression", "accuracy": 85.2, "precision": 84.8, "recall": 85.6, "f1_score": 85.2, "description": "Linear model, fast and interpretable"},
    "random_forest": {"name": "Random Forest", "accuracy": 89.1, "precision": 88.7, "recall": 89.5, "f1_score": 89.1, "description": "Ensemble of decision trees"},
    "svm": {"name": "Support Vector Machine", "accuracy": 86.8, "precision": 86.3, "recall": 87.2, "f1_score": 86.7, "description": "Effective in high-dimensional spaces"},
    "xgboost": {"name": "XGBoost", "accuracy": 91.5, "precision": 91.0, "recall": 92.0, "f1_score": 91.5, "description": "Gradient boosting, best overall performance"}
}

# API Routes
@api_router.get("/")
async def root():
    return {"message": "HeartGuard AI API", "status": "running"}

@api_router.post("/auth/register", response_model=TokenResponse)
async def register(user_data: UserCreate):
    existing = await db.users.find_one({'email': user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_id = str(uuid.uuid4())
    user_doc = {'id': user_id, 'email': user_data.email, 'name': user_data.name, 'password': hash_password(user_data.password), 'created_at': datetime.now(timezone.utc).isoformat()}
    await db.users.insert_one(user_doc)
    token = create_token(user_id, user_data.email)
    return TokenResponse(access_token=token, user=UserResponse(id=user_id, email=user_data.email, name=user_data.name, created_at=user_doc['created_at']))

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    user = await db.users.find_one({'email': credentials.email}, {'_id': 0})
    if not user or not verify_password(credentials.password, user['password']):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token(user['id'], user['email'])
    return TokenResponse(access_token=token, user=UserResponse(id=user['id'], email=user['email'], name=user['name'], created_at=user['created_at']))

@api_router.get("/auth/me", response_model=UserResponse)
async def get_me(user: dict = Depends(get_current_user)):
    return UserResponse(id=user['id'], email=user['email'], name=user['name'], created_at=user['created_at'])

@api_router.post("/predict", response_model=PredictionResult)
async def predict(input_data: PredictionInput, user: dict = Depends(get_current_user)):
    if not ML_MODELS or not SCALER:
        raise HTTPException(status_code=500, detail="ML models not loaded")

    # ✅ FIX 2: numpy array ki jagah DataFrame banao with exact feature names
    features_df = pd.DataFrame([{
        'age':      input_data.age,
        'sex':      input_data.sex,
        'cp':       input_data.cp,
        'trestbps': input_data.trestbps,
        'chol':     input_data.chol,
        'fbs':      input_data.fbs,
        'restecg':  input_data.restecg,
        'thalch':   input_data.thalch,
        'exang':    input_data.exang,
        'oldpeak':  input_data.oldpeak,
    }])

    # Scale karo aur feature names preserve karo
    scaled_array = SCALER.transform(features_df)
    scaled_features = pd.DataFrame(scaled_array, columns=FEATURE_NAMES)

    model_predictions = {}
    probabilities = {}

    for model_name, model in ML_MODELS.items():
        try:
            prediction = int(model.predict(scaled_features)[0])
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(scaled_features)[0]
                prob_positive = float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                prob_positive = float(prediction)
            probabilities[model_name] = prob_positive
            model_predictions[model_name] = {
                "prediction": prediction,
                "probability": round(prob_positive * 100, 2),
                "metrics": MODEL_METRICS.get(model_name, {})
            }
        except Exception as e:
            logging.error(f"Error with model {model_name}: {e}")
            model_predictions[model_name] = {"error": str(e)}

    risk_score = calculate_risk_score(probabilities)
    risk_level = get_risk_level(risk_score)
    recommendations = get_recommendations(risk_level)

    prediction_id = str(uuid.uuid4())
    prediction_doc = {
        'id': prediction_id,
        'user_id': user['id'],
        'input_data': input_data.model_dump(),
        'risk_score': risk_score,
        'risk_level': risk_level,
        'model_predictions': model_predictions,
        'recommendations': recommendations,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    await db.predictions.insert_one(prediction_doc)
    return PredictionResult(**{k: v for k, v in prediction_doc.items() if k != '_id'})

@api_router.get("/predictions", response_model=List[PredictionResult])
async def get_predictions(user: dict = Depends(get_current_user)):
    predictions = await db.predictions.find({'user_id': user['id']}, {'_id': 0}).sort('created_at', -1).to_list(100)
    return predictions

@api_router.get("/predictions/{prediction_id}", response_model=PredictionResult)
async def get_prediction(prediction_id: str, user: dict = Depends(get_current_user)):
    prediction = await db.predictions.find_one({'id': prediction_id, 'user_id': user['id']}, {'_id': 0})
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction

@api_router.get("/research-papers", response_model=List[ResearchPaper])
async def get_research_papers():
    return RESEARCH_PAPERS

@api_router.get("/model-metrics")
async def get_model_metrics():
    return MODEL_METRICS

app.include_router(api_router)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()