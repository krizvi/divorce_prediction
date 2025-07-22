from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from model import DivorcePredictor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["https://metismesh.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input feature size (match training!)
input_size = 26  # Adjust to your real number of features!

# Load model
model = DivorcePredictor(input_size)
model.load_state_dict(torch.load("divorce_predictor_model.pth")) # Question: load_state_dict?
model.eval()

# Define input data model for the API
class DivorceInput(BaseModel):
    features: list  # a list of 26 numbers Question: WHat is list

@app.post("/predict")
def predict(input: DivorceInput):
    with torch.no_grad(): # Question: what does no_grad
        x = torch.tensor([input.features], dtype=torch.float32)
        output = model(x)
        prediction = float(output.item())
        return {"divorce_probability": prediction}
