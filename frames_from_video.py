import cv2
import os
from concurrent.futures import ThreadPoolExecutor

def process_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 40 == 0:
            frame_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_count}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()

video_folder = r'C:\Users\user\Downloads\FaceRecognition_upd1\dataset_video'
output_folder = 'KnowFaces'

os.makedirs(output_folder, exist_ok=True)

video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

with ThreadPoolExecutor() as executor:
    executor.map(lambda video: process_video(video, output_folder), video_files)

#cv2.destroyAllWindows()
