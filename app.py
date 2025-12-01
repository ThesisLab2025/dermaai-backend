# app.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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



# -------- APP & MODEL LOAD --------

app = FastAPI(title="Skin Cancer Detection API")

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
# -------- ENDPOINTS --------

@app.get("/")
async def index():
    """API health check endpoint."""
    return {"status": "running", "message": "Skin Cancer Detection API"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receive the uploaded file in memory, run the model, return JSON.
    The file is NOT saved on disk.
    """
    try:
        # Read image bytes from the uploaded file (in memory)
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess and predict
        x = preprocess_image(img)
        preds = model.predict(x)[0]  # shape (7,)

        # Get predicted class
        idx = int(np.argmax(preds))
        code = CLASS_ORDER[idx]
        binary = get_binary_label(code)
        confidence = float(np.max(preds))

        return JSONResponse({
            "filename": file.filename,
            "lesion_code": code,
            "lesion_name": lesion_info[code],
            "binary_prediction": binary,      # "benign" or "malignant"
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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