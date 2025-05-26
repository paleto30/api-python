from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import base64
import cv2
import face_recognition
from PIL import Image
import io
from src.db import get_students_by_group

app = FastAPI()

# Agrega los orígenes permitidos (tu frontend en Vite/Vue)
origins = [
    "http://localhost:5173",  # donde corre tu frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # puedes usar ["*"] para permitir todos (solo en desarrollo)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageBufferPayload(BaseModel):
    fieldname: str
    originalname: str
    encoding: str
    mimetype: str
    buffer: str  # buffer en base64
    size: int

@app.post("/face-encoding")
async def get_face_encoding(payload: ImageBufferPayload):
    try:
        # Decodificar base64 a bytes
        image_bytes = base64.b64decode(payload.buffer)

        # Convertir bytes a NumPy array y luego decodificar como imagen con OpenCV
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=422, detail="Could not decode image")

        # OpenCV usa BGR, face_recognition usa RGB → convertimos
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extraer encodings del rostro
        encodings = face_recognition.face_encodings(rgb_image)

        if not encodings:
            raise HTTPException(status_code=422, detail="No face detected in the image.")

        return {
            "encodings": encodings[0].tolist()  # Convertimos de NumPy a lista
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")



class RecognitionRequest(BaseModel):
    group_id: str
    image: str  # Base64 image

@app.post("/recognize")
async def recognize_faces(request: RecognitionRequest):
    try:
        # Decodificar imagen
        image_data = base64.b64decode(request.image.split(",")[1])
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        frame = np.array(pil_image)

        # Obtener codificaciones del frame
        frame_encodings = face_recognition.face_encodings(frame)
        
        # Obtener estudiantes del grupo con encodings
        students = await get_students_by_group(request.group_id)
        known_encodings = [s["face_encodings"] for s in students]
        known_ids = [str(s["_id"]) for s in students]

        recognized_ids = set()

        if frame_encodings:
            for face_encoding in frame_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                for idx, match in enumerate(matches):
                    if match:
                        recognized_ids.add(known_ids[idx])

        # Construir la lista de asistencia con todos los estudiantes
        assistance = []
        for student in students:
            student_id = str(student["_id"])
            assistance.append({
                "studentId": student_id,
                "present": student_id in recognized_ids
            })

        return {
            "groupId": request.group_id,
            "assistance": assistance
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))