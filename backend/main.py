from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

# Load model and vectorizer
with open("../model/fake_job_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Create FastAPI app
app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input format
class JobInput(BaseModel):
    title: str
    company_profile: str
    description: str
    requirements: str

# Prediction route
@app.post("/predict")
def predict(job: JobInput):
    # Combine all text
    text = f"{job.title} {job.company_profile} {job.description} {job.requirements}"
    
    # Vectorize
    text_vec = vectorizer.transform([text])
    
    # Predict
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]
    
    if prediction == 1:
        result = "FAKE"
        confidence = round(probability[1] * 100, 2)
    else:
        result = "REAL"
        confidence = round(probability[0] * 100, 2)
    
    return {
        "result": result,
        "confidence": f"{confidence}%"
    }

# Home route
@app.get("/")
def home():
    return {"message": "Fake Job Detector API is running!"}