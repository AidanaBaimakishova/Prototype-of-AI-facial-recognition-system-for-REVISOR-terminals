import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
import pickle

# Параметры
KNOWFACES_DIR = 'KnowFaces'
EMBEDDINGS_FILE = 'face_embeddings.pkl'

# Инициализация модели InsightFace
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

def extract_embeddings(image_path):
    image = cv2.imread(image_path)
    faces = app.get(image)
    if faces:
        return faces[0].embedding
    return None

def save_face_embeddings():
    embeddings = {}
    for filename in os.listdir(KNOWFACES_DIR):
        if filename.endswith('.jpg'):
            image_path = os.path.join(KNOWFACES_DIR, filename)
            embedding = extract_embeddings(image_path)
            if embedding is not None:
                embeddings[filename] = embedding
    
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)

if __name__ == '__main__':
    save_face_embeddings()
