# DermaAI Backend

Skin cancer detection API using DenseNet121 deep learning model with MongoDB database.

## Prerequisites

- Python 3.10+
- MongoDB Atlas account (or local MongoDB)


## Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/ThesisLab2025/dermaai-backend.git
cd dermaai-backend
```

### Step 2: Create Virtual Environment

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

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the root directory:

```env
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/?appName=Cluster0
```

### Step 5: Add Model Weights

Place your `best_model.weights.h5` file in the root directory.

### Step 6: Run the Server

```bash
python app.py
```

The server will start on:
- **Local**: http://localhost:8000
- **Network**: http://your-device-ip:8000
- **API Docs**: http://localhost:8000/docs

---

## API Endpoints

### Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API health check |

### User Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/users` | Create or update user |

**Request Body:**
```json
{
  "user_id": "user_123",
  "name": "John Doe",
  "email": "john@example.com"
}
```

### Skin Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Analyze skin image (auto-saves if user_id provided) |
| POST | `/api/analysis/save` | Manually save analysis |
| GET | `/api/analysis/stats/{user_id}` | Get user statistics |
| GET | `/api/analysis/history/{user_id}` | Get user scan history |

**POST /predict Request:**
```
Content-Type: multipart/form-data
file: [image]
user_id: "user_123" (optional - enables auto-save)
user_email: "john@example.com" (optional)
```

**Response:**
```json
{
  "success": true,
  "filename": "skin.jpg",
  "lesion_code": "nv",
  "lesion_name": "Melanocytic Nevus (benign moles)",
  "binary_prediction": "benign",
  "confidence": 0.9523,
  "analysis_id": "...",
  "total_scans": 5,
  "auto_saved": true
}
```

### Reports

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/reports/save` | Save a report |
| GET | `/api/reports/{user_id}` | Get all reports for user |
| GET | `/api/reports/detail/{report_id}` | Get single report detail |

---

## Supported Skin Lesion Classes

| Code | Name | Type |
|------|------|------|
| nv | Melanocytic Nevus | Benign |
| bkl | Benign Keratosis | Benign |
| vasc | Vascular Lesion | Benign |
| df | Dermatofibroma | Benign |
| mel | Melanoma | Malignant |
| bcc | Basal Cell Carcinoma | Malignant |
| akiec | Actinic Keratosis | Malignant |

---

## Project Structure

```
dermaai-backend/
├── app.py                    # Main application
├── app_optimized.py          # Optimized version
├── best_model.weights.h5     # Model weights (not in git)
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not in git)
├── .gitignore
└── README.md
```

---

## Deployment (Render)

1. Push code to GitHub
2. Connect repo to Render
3. Set environment variable: `MONGODB_URL`
4. Set start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Deploy!

---

## Troubleshooting

### MongoDB Connection Error
```
ServerSelectionTimeoutError: localhost:27017
```
**Solution:** Check your `.env` file has the correct `MONGODB_URL` pointing to MongoDB Atlas (not localhost).

### Port Already in Use
```
[Errno 10048] error while attempting to bind on address
```
**Solution:** Kill the process using the port:
```powershell
Get-NetTCPConnection -LocalPort 8000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

---

## License

MIT License
