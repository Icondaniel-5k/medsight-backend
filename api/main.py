import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import io

app = FastAPI(title="MedSight AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/malaria_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load full model directly
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

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
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            label = "Parasitized" if predicted.item() == 0 else "Uninfected"

        return {"prediction": label}
    except Exception as e:
        return {"error": str(e)}