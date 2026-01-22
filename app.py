# app.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io
from pathlib import Path

# -------- CONFIG --------

# Load environment variables
load_dotenv()
MONGODB_URL = os.getenv("MONGODB_URL")

# -------- DERMOSCOPY CHECK (IMPROVED HEURISTIC) --------
def is_dermoscopic_image(img: Image.Image) -> dict:
    """
    Check if image is dermoscopic using two signals:
    1. Border uniformity (edges should be similar)
    2. Clutter (background variance - lower = more dermoscopic)
    """
    # Resize to 256x256
    small = img.convert("RGB").resize((256, 256))
    arr = np.asarray(small).astype("float32") / 255.0
    
    # Convert to grayscale
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
    
    # Signal 1: Edge uniformity
    band = 24
    top = gray[:band, :].ravel()
    bottom = gray[-band:, :].ravel()
    left = gray[:, :band].ravel()
    right = gray[:, -band:].ravel()
    
    edge_means = np.array([top.mean(), bottom.mean(), left.mean(), right.mean()], dtype=np.float32)
    edge_std = float(edge_means.std())
    border_uniform_score = float(np.clip((0.08 - edge_std) / 0.08, 0.0, 1.0))
    
    # Signal 2: Clutter via tile RGB std
    tile = 32
    tile_stds = []
    for y in range(0, 256, tile):
        for x in range(0, 256, tile):
            patch = arr[y:y + tile, x:x + tile, :]
            tile_stds.append(float(np.std(patch)))
    
    clutter = float(np.mean(tile_stds))
    clutter_score = float(np.clip((0.14 - clutter) / 0.14, 0.0, 1.0))
    
    # Combine scores
    score = (0.55 * border_uniform_score) + (0.45 * clutter_score)
    is_dermo = score >= 0.13  # THRESHOLD
    
    return {
        "is_dermoscopic": bool(is_dermo),
        "score": float(np.clip(score, 0.0, 1.0)),
        "signals": {
            "edge_means": [round(float(x), 4) for x in edge_means],
            "edge_std": round(edge_std, 4),
            "border_uniform_score": round(border_uniform_score, 4),
            "clutter": round(clutter, 4),
            "clutter_score": round(clutter_score, 4),
        }
    }

# Must match the image size used in your Kaggle training
IMG_SIZE = 224  # change if you used a different size

BASE_DIR = Path(__file__).resolve().parent   # dermaai-backend folder

# Path to the weights file you exported from Kaggle:
WEIGHTS_PATH = BASE_DIR / "best_model.weights.h5"

# Class order must match your training label order
CLASS_ORDER = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

lesion_info = {
    'nv':   'Melanocytic Nevus (benign moles)',
    'mel':  'Melanoma (malignant)',
    'bkl':  'Benign Keratosis',
    
    'bcc':  'Basal Cell Carcinoma',
    'akiec':'Actinic Keratosis',
    'vasc': 'Vascular Lesion',
    'df':   'Dermatofibroma'
}

benign_classes = ['nv', 'bkl', 'vasc', 'df']
malignant_classes = ['mel', 'bcc', 'akiec']


def get_binary_label(class_name: str) -> str:
    """Map 7-class code to benign / malignant."""
    return "malignant" if class_name in malignant_classes else "benign"


def build_model():
    """
    Rebuild the same architecture you used in Kaggle:
      - DenseNet121 (no top)
      - GlobalAveragePooling2D
      - Dense(256, relu)
      - Dropout(0.5)
      - Dense(7, softmax)
    """
    base_model = DenseNet121(
        weights=None,  # we will load our own trained weights
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(len(CLASS_ORDER), activation='softmax')
    ])
    return model


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize + convert to array + normalize, same as training."""
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr



# -------- DATABASE CONNECTION --------
client = AsyncIOMotorClient(MONGODB_URL)
db = client.dermaai  # database name
users_collection = db.users  # collection name
analysis_collection = db.analysis  # collection for prediction history
reports_collection = db.reports  # collection for generated reports
print("Connected to MongoDB.")


# -------- APP & MODEL LOAD --------

# Lifespan event handler (modern approach)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create unique index on user_id
    await users_collection.create_index("user_id", unique=True)
    print("Database index created.")
    yield
    # Shutdown: Close MongoDB connection
    client.close()
    print("MongoDB connection closed.")

app = FastAPI(title="Skin Cancer Detection API", lifespan=lifespan)


# Enable CORS for all origins (adjust as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Enable CORS for all origins (adjust as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Build architecture and load trained weights (no file is saved when predicting)
model = build_model()
#this is add according to the after adjusting the path
model.load_weights(str(WEIGHTS_PATH))
#model.load_weights(WEIGHTS_PATH)
print("DenseNet121 model built and weights loaded.")

#---------------from here onward ifty  you can change----------------
# -------- USER MODEL --------

class UserCreate(BaseModel):
    user_id: str
    name: str
    email: str

class AnalysisSave(BaseModel):
    user_id: str
    user_email: str
    filename: str
    lesion_code: str
    lesion_name: str
    binary_prediction: str
    confidence: float
    image_data: str  # base64 encoded image
    analyzed_at: str  # ISO date string

class ReportSave(BaseModel):
    user_id: str
    user_email: str
    report_type: str
    filename: str
    lesion_code: str
    lesion_name: str
    binary_prediction: str
    confidence: float
    image_data: str  # base64 encoded image
    report_html: str  # HTML content of report
    generated_at: str  # ISO date string

# -------- ENDPOINTS --------

@app.get("/")
async def index():
    """API health check endpoint."""
    return {"status": "running", "message": "Skin Cancer Detection API"}


# POST /users - Create or update user in MongoDB
@app.post("/users")
async def create_or_update_user(user: UserCreate):
    """
    Handle user login/register - saves to MongoDB.
    If user exists: update name/email
    If user doesn't exist: create new user
    """
    # Check if user exists
    existing_user = await users_collection.find_one({"user_id": user.user_id})
    
    if existing_user:
        # Update existing user
        await users_collection.update_one(
            {"user_id": user.user_id},
            {"$set": {"name": user.name, "email": user.email, "updated_at": datetime.utcnow()}}
        )
        print(f"User updated: {user.user_id}")
        return {
            "message": "User updated",
            "user_id": user.user_id,
            "name": user.name,
            "email": user.email,
            "is_new_user": False
        }
    else:
        # Create new user
        new_user = {
            "user_id": user.user_id,
            "name": user.name,
            "email": user.email,
            "created_at": datetime.utcnow()
        }
        await users_collection.insert_one(new_user)
        print(f"New user created: {user.user_id}")
        return {
            "message": "User created",
            "user_id": user.user_id,
            "name": user.name,
            "email": user.email,
            "is_new_user": True
        }


# POST /api/analysis/save - Save prediction result to database
@app.post("/api/analysis/save")
async def save_analysis(analysis: AnalysisSave):
    """
    Save a prediction/analysis result to the database.
    Returns total scan count for the user.
    """
    try:
        analysis_doc = {
            "user_id": analysis.user_id,
            "user_email": analysis.user_email,
            "filename": analysis.filename,
            "lesion_code": analysis.lesion_code,
            "lesion_name": analysis.lesion_name,
            "binary_prediction": analysis.binary_prediction,
            "confidence": analysis.confidence,
            "image_data": analysis.image_data,
            "analyzed_at": analysis.analyzed_at,
            "saved_at": datetime.utcnow()
        }
        
        # Save analysis to database
        result = await analysis_collection.insert_one(analysis_doc)
        
        # Count total scans for this user
        total_scans = await analysis_collection.count_documents({"user_id": analysis.user_id})
        
        print(f"Analysis saved for user: {analysis.user_id} (Total scans: {total_scans})")
        
        return {
            "success": True,
            "message": "Analysis saved successfully",
            "analysis_id": str(result.inserted_id),
            "total_scans": total_scans
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# GET /api/analysis/stats/{user_id} - Get user analysis statistics
@app.get("/api/analysis/stats/{user_id}")
async def get_analysis_stats(user_id: str):
    """
    Get analysis statistics for a specific user.
    """
    try:
        # Count total scans for this user
        total_scans = await analysis_collection.count_documents({"user_id": user_id})
        
        # Count scans this week (last 7 days)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        scans_this_week = await analysis_collection.count_documents({
            "user_id": user_id,
            "saved_at": {"$gte": seven_days_ago}
        })
        
        # Count pending reviews (malignant results)
        pending_reviews = await analysis_collection.count_documents({
            "user_id": user_id,
            "binary_prediction": {"$regex": "^malignant$", "$options": "i"}
        })
        
        # Count reports generated
        reports_generated = await reports_collection.count_documents({"user_id": user_id})
        
        return {
            "success": True,
            "total_scans": total_scans,
            "scans_this_week": scans_this_week,
            "pending_reviews": pending_reviews,
            "reports_generated": reports_generated
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# POST /api/reports/save - Save generated report to database
@app.post("/api/reports/save")
async def save_report(report: ReportSave):
    """
    Save a generated analysis report to the database.
    """
    try:
        report_doc = {
            "user_id": report.user_id,
            "user_email": report.user_email,
            "report_type": report.report_type,
            "filename": report.filename,
            "lesion_code": report.lesion_code,
            "lesion_name": report.lesion_name,
            "binary_prediction": report.binary_prediction,
            "confidence": report.confidence,
            "image_data": report.image_data,
            "report_html": report.report_html,
            "generated_at": report.generated_at,
            "saved_at": datetime.utcnow()
        }
        
        result = await reports_collection.insert_one(report_doc)
        
        # Count total reports for this user
        total_reports = await reports_collection.count_documents({"user_id": report.user_id})
        
        print(f"Report saved for user: {report.user_id} (Total reports: {total_reports})")
        
        return {
            "success": True,
            "message": "Report saved successfully",
            "report_id": str(result.inserted_id),
            "total_reports": total_reports
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# GET /api/reports/{user_id} - Get all reports for a user
@app.get("/api/reports/{user_id}")
async def get_user_reports(user_id: str):
    """
    Get all reports for a specific user, including image_data and report_html.
    """
    try:
        cursor = reports_collection.find(
            {"user_id": user_id}
        ).sort("saved_at", -1)  # Most recent first
        
        reports = []
        async for report in cursor:
            report_id = str(report["_id"])
            reports.append({
                "_id": report_id,  # MongoDB format
                "id": report_id,   # Alternative format
                "filename": report.get("filename", ""),
                "lesion_code": report.get("lesion_code", ""),
                "lesion_name": report.get("lesion_name", ""),
                "binary_prediction": report.get("binary_prediction", ""),
                "confidence": report.get("confidence", 0),
                "image_data": report.get("image_data", ""),
                "report_html": report.get("report_html", ""),
                "generated_at": report.get("generated_at", "")
            })
        
        return {
            "success": True,
            "reports": reports,
            "total": len(reports)
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# GET /api/reports/detail/{report_id} - Get a single report with full HTML
@app.get("/api/reports/detail/{report_id}")
async def get_report_detail(report_id: str):
    """
    Get a single report by ID, including full report_html.
    """
    try:
        from bson import ObjectId
        report = await reports_collection.find_one({"_id": ObjectId(report_id)})
        
        if not report:
            return JSONResponse({"success": False, "error": "Report not found"}, status_code=404)
        
        report["_id"] = str(report["_id"])
        
        return {
            "success": True,
            "report": report
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# DELETE /api/report/{report_id} - Delete a report
@app.delete("/api/report/{report_id}")
async def delete_report(report_id: str):
    """
    Delete a report by ID and also remove the corresponding analysis entry
    to update the scan counts properly.
    """
    try:
        from bson import ObjectId
        
        # Validate ObjectId format
        if not ObjectId.is_valid(report_id):
            return JSONResponse({"success": False, "error": "Invalid report ID format"}, status_code=400)
        
        # First, get the report to find the corresponding analysis
        report = await reports_collection.find_one({"_id": ObjectId(report_id)})
        
        if not report:
            return JSONResponse({"success": False, "error": "Report not found"}, status_code=404)
        
        # Delete the report
        delete_result = await reports_collection.delete_one({"_id": ObjectId(report_id)})
        
        if delete_result.deleted_count == 0:
            return JSONResponse({"success": False, "error": "Failed to delete report"}, status_code=500)
        
        # Also delete the corresponding analysis entry to update scan counts
        # Match by user_id, filename, and lesion_code to find the corresponding analysis
        analysis_filter = {
            "user_id": report["user_id"],
            "filename": report["filename"],
            "lesion_code": report["lesion_code"],
            # Also match confidence to ensure we delete the right analysis
            "confidence": report["confidence"]
        }
        
        analysis_delete_result = await analysis_collection.delete_one(analysis_filter)
        
        print(f"Report deleted: {report_id}")
        if analysis_delete_result.deleted_count > 0:
            print(f"Corresponding analysis also deleted for user: {report['user_id']}")
        else:
            print(f"Warning: No corresponding analysis found to delete for report: {report_id}")
        
        # Get updated stats for the user
        user_id = report["user_id"]
        total_scans = await analysis_collection.count_documents({"user_id": user_id})
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        scans_this_week = await analysis_collection.count_documents({
            "user_id": user_id,
            "saved_at": {"$gte": seven_days_ago}
        })
        
        return {
            "success": True,
            "message": "Report and corresponding analysis deleted successfully",
            "updated_stats": {
                "total_scans": total_scans,
                "scans_this_week": scans_this_week
            }
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# DELETE /api/analysis/{analysis_id} - Delete an analysis entry
@app.delete("/api/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """
    Delete an analysis entry by ID. This will also update scan counts.
    """
    try:
        from bson import ObjectId
        
        # Validate ObjectId format
        if not ObjectId.is_valid(analysis_id):
            return JSONResponse({"success": False, "error": "Invalid analysis ID format"}, status_code=400)
        
        # First, get the analysis to get user info for updated stats
        analysis = await analysis_collection.find_one({"_id": ObjectId(analysis_id)})
        
        if not analysis:
            return JSONResponse({"success": False, "error": "Analysis not found"}, status_code=404)
        
        # Delete the analysis
        result = await analysis_collection.delete_one({"_id": ObjectId(analysis_id)})
        
        if result.deleted_count == 0:
            return JSONResponse({"success": False, "error": "Failed to delete analysis"}, status_code=500)
        
        # Get updated stats for the user
        user_id = analysis["user_id"]
        total_scans = await analysis_collection.count_documents({"user_id": user_id})
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        scans_this_week = await analysis_collection.count_documents({
            "user_id": user_id,
            "saved_at": {"$gte": seven_days_ago}
        })
        
        print(f"Analysis deleted: {analysis_id} for user: {user_id}")
        
        return {
            "success": True,
            "message": "Analysis deleted successfully",
            "updated_stats": {
                "total_scans": total_scans,
                "scans_this_week": scans_this_week
            }
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# GET /api/analysis/history/{user_id} - Get analysis history for a user
@app.get("/api/analysis/history/{user_id}")
async def get_analysis_history(user_id: str):
    """
    Get all analysis/scan history for a specific user.
    """
    try:
        cursor = analysis_collection.find(
            {"user_id": user_id},
            {"image_data": 0}  # Exclude image_data to reduce payload size
        ).sort("saved_at", -1)  # Most recent first
        
        analyses = []
        async for analysis in cursor:
            analysis["_id"] = str(analysis["_id"])
            analyses.append(analysis)
        
        return {
            "success": True,
            "analyses": analyses,
            "total": len(analyses)
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user_id: str = None,
    user_email: str = None
):
    """
    Receive the uploaded file in memory, validate dermoscopy, run the model, return JSON.
    
    Flow:
    1. Validate file type and read image
    2. Check if image is dermoscopic (using heuristic)
    3. If NOT dermoscopic: return 400 with score
    4. If dermoscopic: run model prediction
    5. Auto-save analysis if user_id is provided
    """
    try:
        # 1) Validate file type
        if file.content_type and not file.content_type.startswith("image/"):
            return JSONResponse(
                {"success": False, "error": "Please upload a valid image file."},
                status_code=400
            )

        # Read image bytes from the uploaded file (in memory)
        contents = await file.read()
        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            return JSONResponse(
                {"success": False, "error": f"Failed to read image: {str(e)}"},
                status_code=400
            )

        # Convert image to base64 for storage
        import base64
        image_base64 = base64.b64encode(contents).decode('utf-8')

        # 2) Check if image is dermoscopic
        dermo = is_dermoscopic_image(img)
        
        if not dermo["is_dermoscopic"]:
            # NOT DERMOSCOPIC - return error with score
            return JSONResponse({
                "filename": file.filename,
                "is_dermoscopic": False,
                "dermoscopy_score": dermo["score"],
                "message": "This image does not look like a dermoscopic image. Please upload a dermoscopy image.",
                "signals": dermo["signals"]
            }, status_code=400)

        # 3) DERMOSCOPIC - run prediction
        x = preprocess_image(img)
        preds = model.predict(x, verbose=0)[0]

        idx = int(np.argmax(preds))
        code = CLASS_ORDER[idx]
        binary = get_binary_label(code)
        confidence = float(np.max(preds))
        
        # Return success response
        result = {
            "success": True,
            "filename": file.filename,
            "is_dermoscopic": True,
            "dermoscopy_score": round(dermo["score"], 4),
            "lesion_code": code,
            "lesion_name": lesion_info[code],
            "binary_prediction": binary,
            "confidence": round(confidence, 4)
        }
        
        # Auto-save analysis if user_id is provided
        if user_id:
            analysis_doc = {
                "user_id": user_id,
                "user_email": user_email or "",
                "filename": file.filename,
                "lesion_code": code,
                "lesion_name": lesion_info[code],
                "binary_prediction": binary,
                "confidence": round(confidence, 4),
                "image_data": image_base64,
                "analyzed_at": datetime.utcnow().isoformat(),
                "saved_at": datetime.utcnow()
            }
            
            save_result = await analysis_collection.insert_one(analysis_doc)
            total_scans = await analysis_collection.count_documents({"user_id": user_id})
            
            result["analysis_id"] = str(save_result.inserted_id)
            result["total_scans"] = total_scans
            result["auto_saved"] = True
            print(f"Analysis auto-saved for user: {user_id}")
        
        return JSONResponse(result)
        
    except Exception as e:
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )



# -------- RUN SERVER --------
if __name__ == "__main__":
    import uvicorn
    import socket
    
    HOST = "0.0.0.0"  # Bind to all network interfaces (accessible from other devices)
    PORT = 8000       # Port number
    
    # Get device IP address
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
    
    local_ip = get_local_ip()
    print(f"\n{'='*50}")
    print(f"Server running on:")
    print(f"  Local:   http://localhost:{PORT}")
    print(f"  Network: http://{local_ip}:{PORT}")
    print(f"{'='*50}\n")
    
    uvicorn.run(app, host=HOST, port=PORT)