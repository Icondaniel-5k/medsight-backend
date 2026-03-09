# api/main.py

import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms

# ------------------------
# App Initialization
# ------------------------
app = FastAPI(title="MedSight AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Model Setup
# ------------------------
MODEL_PATH = "models/malaria_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the full model object directly
try:
    model = torch.load(MODEL_PATH, map_location=device)
    model.to(device)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------------
# Routes
# ------------------------
@app.get("/")
def read_root():
    return {"message": "MedSight AI backend is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            label = "Parasitized" if predicted.item() == 0 else "Uninfected"

        return {"prediction": label}
    except Exception as e:
        return {"error": str(e)}