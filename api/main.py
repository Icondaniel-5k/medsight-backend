import io
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

app = FastAPI(title="MedSight AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/malaria_model.pth"


# -------------------------
# Recreate your CNN model
# -------------------------
class MalariaCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# -------------------------
# Load model weights
# -------------------------
model = MalariaCNN()

state_dict = torch.load(MODEL_PATH, map_location=device)

model.load_state_dict(state_dict)

model.to(device)
model.eval()


# -------------------------
# Image preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# -------------------------
# Routes
# -------------------------
@app.get("/")
def home():
    return {"message": "MedSight AI backend is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:

        image_bytes = await file.read()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():

            outputs = model(image)

            _, predicted = torch.max(outputs, 1)

            label = "Parasitized" if predicted.item() == 0 else "Uninfected"

        return {"prediction": label}

    except Exception as e:

        return {"error": str(e)}