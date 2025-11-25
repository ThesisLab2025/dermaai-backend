# app.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
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

BASE_DIR = Path(__file__).resolve().parent.parent   # THESIS_PROJECT folder

# Path to the weights file you exported from Kaggle:
# model.save_weights("/kaggle/working/best_model_weights.h5")
WEIGHTS_PATH = BASE_DIR / "model" / "best_model.weights.h5"
#need to change the file path
#WEIGHTS_PATH = r"D:\Indu\Thesis\model\CorrectModel\best_model.weights.h5"

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

# Build architecture and load trained weights (no file is saved when predicting)
model = build_model()
#this is add according to the after adjusting the path
model.load_weights(str(WEIGHTS_PATH))
#model.load_weights(WEIGHTS_PATH)
print("DenseNet121 model built and weights loaded.")

#---------------from here onward ifty  you can change----------------
# -------- ENDPOINTS --------

@app.get("/", response_class=HTMLResponse)
async def index():
    """
    Simple HTML page to upload an image from the browser.
    """
    html = """
    <html>
        <body>
            <h2>Upload a skin lesion image</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html)


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
