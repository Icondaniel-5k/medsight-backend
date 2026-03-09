import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms
import io

app = FastAPI(title="MedSight AI Backend")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model setup
MODEL_PATH = "models/malaria_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1️⃣ Define the model architecture exactly like in training
model = models.resnet18()  # <-- replace if you used a different model
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: Parasitized / Uninfected

# 2️⃣ Load the state dict
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)

# 3️⃣ Send model to device and set to eval
model = model.to(device)
model.eval()

# 4️⃣ Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

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