# DermaAI Backend

Skin cancer detection API using DenseNet121 deep learning model.

## Setup

### 1. Create and activate virtual environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.venv\Scripts\activate.bat

# Activate (Linux/macOS)
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python app.py
```

The server will start on:
- **Local**: http://localhost:8000
- **Network**: http://<your-device-ip>:8000

## API Endpoints

### GET `/`
Health check endpoint. Returns API status.

**Response**:
```json
{
  "status": "running",
  "message": "Skin Cancer Detection API"
}
```

### POST `/predict`
Upload a skin lesion image for prediction.

**Request**: `multipart/form-data` with `file` field containing the image.

**Response**:
```json
{
  "filename": "image.jpg",
  "lesion_code": "nv",
  "lesion_name": "Melanocytic Nevus (benign moles)",
  "binary_prediction": "benign",
  "confidence": 0.9523
}
```

## Supported Classes

| Code | Name | Type |
|------|------|------|
| nv | Melanocytic Nevus | Benign |
| bkl | Benign Keratosis | Benign |
| vasc | Vascular Lesion | Benign |
| df | Dermatofibroma | Benign |
| mel | Melanoma | Malignant |
| bcc | Basal Cell Carcinoma | Malignant |
| akiec | Actinic Keratosis | Malignant |
