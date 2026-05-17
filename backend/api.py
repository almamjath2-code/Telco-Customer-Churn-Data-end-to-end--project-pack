import os
import pandas as pd
import joblib

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware# this is use for frontend and backend communication if they are on different domains or ports


# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")

# --------------------------
# Load model
# --------------------------
train_columns = joblib.load("x_train_columns.pkl")
model = joblib.load("customer_churn_model.pkl")

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------
# Request schema
# --------------------------
class CustomerData(BaseModel):
    Age: float
    Gender: str
    Tenure: float
    MonthlyCharges: float
    Contract: str
    PaymentMethod: str
    TotalCharges: float

# --------------------------
# API Key check function
# --------------------------
def verify_api_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# --------------------------
# Predict API
# --------------------------
@app.get("/health")
def health():
    return {"health": " welcome home to customer churn prediction api"}

@app.post("/predict")
def predict(data: CustomerData, x_api_key: str = Header(None)):

    # Check API key
    verify_api_key(x_api_key)

    new_customer_data = data.dict()

    input_df = pd.DataFrame([{
        "Age": new_customer_data["Age"],
        "Tenure": new_customer_data["Tenure"],
        "MonthlyCharges": new_customer_data["MonthlyCharges"],
        "TotalCharges": new_customer_data["TotalCharges"]
    }])

    # One-hot columns
    for col in train_columns:
        if col.startswith("Gender_") or col.startswith("Contract_") or col.startswith("PaymentMethod_"):
            input_df[col] = 0

    gender_col = "Gender_" + new_customer_data["Gender"]
    contract_col = "Contract_" + new_customer_data["Contract"]
    payment_col = "PaymentMethod_" + new_customer_data["PaymentMethod"]

    if gender_col in input_df.columns:
        input_df[gender_col] = 1

    if contract_col in input_df.columns:
        input_df[contract_col] = 1

    if payment_col in input_df.columns:
        input_df[payment_col] = 1

    input_df = input_df.reindex(columns=train_columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = None

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_df)[0][1])

    # YES / NO output
    result = "YES" if prediction == 1 else "NO"

    return {
        "churn_prediction": result,
        "prediction_value": int(prediction),
        "churn_probability": probability
    }
  #python app.py this for without api code just cheke pridiction
  # uvicorn api:app --reload this use for whith api code run 
  # docker build -t customer-churn-api . this use for build docker image
  # docker run -p 8000:8000 customer-churn-api this use for run docker image and access api on localhost:8000
 