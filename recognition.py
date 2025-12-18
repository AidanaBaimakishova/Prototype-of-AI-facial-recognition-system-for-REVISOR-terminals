import cv2
import os
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine

# Параметры
VIDEO_DIR = 'test_video'  # Папка с тестовыми видео
EMBEDDINGS_FILE = 'face_embeddings.pkl'
MATCHED_FACES_DIR = 'MatchedFaces'
FRAME_SKIP = 40

# Создание директорий, если они не существуют
os.makedirs(MATCHED_FACES_DIR, exist_ok=True)

# Инициализация модели InsightFace
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(1024, 1024))  # Увеличение размера детекции

# Загрузка сохраненных эмбеддингов
with open(EMBEDDINGS_FILE, 'rb') as f:
    saved_embeddings = pickle.load(f)

# Нормализация сохраненных эмбеддингов
saved_embeddings = {filename: normalize(embedding.reshape(1, -1))[0] for filename, embedding in saved_embeddings.items()}

THRESHOLD = 0.6  # Установите порог, экспериментируйте с ним

def find_matching_face(face_embedding):
    """Поиск самого похожего лица по нормализованному эмбеддингу с порогом."""
    face_embedding = normalize(face_embedding.reshape(1, -1))[0]
    min_distance = float('inf')
    best_match = None
    
    for filename, saved_embedding in saved_embeddings.items():
        distance = cosine(face_embedding, saved_embedding)
        if distance < min_distance:
            min_distance = distance
            best_match = filename
    
    if min_distance < THRESHOLD:
        return best_match
    else:
        return None  # Лицо не соответствует порогу

def calculate_distance_from_center(bbox, frame_shape):
    """Рассчет расстояния от центра кадра до центра лица."""
    frame_center = (frame_shape[1] / 2, frame_shape[0] / 2)
    face_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    distance = np.linalg.norm(np.array(face_center) - np.array(frame_center))
    return distance

def process_frame(frame):
    """Обработка кадра, выбор лица, ближайшего к камере и к центру."""
    faces = app.get(frame)
    
    if not faces:
        return

    # Найдем самое близкое к центру и большое лицо
    best_face = None
    min_distance_to_center = float('inf')
    max_bbox_area = 0

    for face in faces:
        bbox = face.bbox
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        distance_to_center = calculate_distance_from_center(bbox, frame.shape)

        # Условие: выбираем лицо, которое больше и ближе к центру
        if bbox_area > max_bbox_area or (bbox_area == max_bbox_area and distance_to_center < min_distance_to_center):
            best_face = face
            max_bbox_area = bbox_area
            min_distance_to_center = distance_to_center

    if best_face:
        match = find_matching_face(best_face.embedding)
        if match:
            print(f"Match found: {match}")
        else:
            print("Unknown")
        
        label = f"Name: {match}" if match else "Unknown"
        draw_label_with_wrapping(frame, label, (int(best_face.bbox[0]), int(best_face.bbox[1] - 10)))
        save_path = os.path.join(MATCHED_FACES_DIR, f"{best_face.bbox[0]}_{best_face.bbox[1]}.jpg")
        cv2.imwrite(save_path, frame)

def draw_label_with_wrapping(image, label, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 255, 0), thickness=1):
    """Функция для добавления текста с переносом строк."""
    y0, dy = position[1], 20
    for i, line in enumerate(label.split('\n')):
        y = y0 + i * dy
        cv2.putText(image, line, (position[0], y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

def process_video(video_path):
    """Обработка одного видеофайла."""
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % FRAME_SKIP == 0:
            process_frame(frame)

        frame_count += 1

    video_capture.release()
    cv2.destroyAllWindows()

def main():
    # Получаем список всех файлов в папке test_video
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]

    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        print(f"Processing video: {video_file}")
        process_video(video_path)

if __name__ == '__main__':
    main()
