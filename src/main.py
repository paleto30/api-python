from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from src.recognizer import load_known_faces, recognize_faces_from_image



app = FastAPI()

# Permitir orígenes que quieras (o * para todos)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o lista de URLs frontales autorizadas
    allow_credentials=True,
    allow_methods=["*"],  # permite OPTIONS, POST, GET, etc
    allow_headers=["*"],
)

# Cargar rostros al iniciar
load_known_faces()

class ImageData(BaseModel):
    image: str


@app.post("/reconocer")
def reconocer(data: ImageData):
    try:
        # Separar encabezado y decodificar base64
        header, encoded = data.image.split(",", 1)
        img_bytes = base64.b64decode(encoded)

        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        np_img = np.array(img)

        results = recognize_faces_from_image(np_img)
 
        return {"results": results}

    except Exception as e:
        return {"error": str(e)}