import os
import face_recognition

dataset_path = 'dataset'
known_face_encodings = []
known_face_names = []

# Cargar imágenes conocidas una sola vez
def load_known_faces():
    global known_face_encodings, known_face_names

    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(dataset_path, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])

    print(f"{len(known_face_names)} estudiantes cargados:", known_face_names)

# Procesar una imagen y devolver el nombre
def recognize_faces_from_image(np_image):
    face_locations = face_recognition.face_locations(np_image)
    face_encodings = face_recognition.face_encodings(np_image, face_locations)

    results = []

    for face_encoding, location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconocido"

        if matches:
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        results.append({
            "name": name,
            "location": location  # (top, right, bottom, left)
        })

    return results