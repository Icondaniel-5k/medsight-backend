# MedSight AI Backend

This is the FastAPI backend for MedSight AI — a malaria detection application.

## Folder structure

- `api/main.py` → FastAPI application
- `models/malaria_model.pth` → trained PyTorch model
- `requirements.txt` → dependencies

## Deployment on Render.com

1. Push this repo to GitHub.
2. On Render.com, create a **New Web Service**:
   - Connect your GitHub repo.
   - Branch: main
   - Runtime: Python 3.10+
   - Start Command: `uvicorn api.main:app --host 0.0.0.0 --port 10000`
3. Ensure `malaria_model.pth` is inside `models/`.
4. Deploy. Render will provide a public URL.
5. Test:
   - `GET /` → should return {"message": "MedSight AI backend is running"}
   - `POST /predict` → upload image to get prediction.

## Notes

- For frontend requests, enable CORS for your domain.
- `/docs` endpoint is available for testing.