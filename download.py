import requests
import os
from concurrent.futures import ThreadPoolExecutor

def download_file(url, save_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def download_from_yandex_disk(public_link, save_folder):
    api_endpoint = 'https://cloud-api.yandex.net/v1/disk/public/resources'
    response = requests.get(api_endpoint, params={'public_key': public_link})
    response.raise_for_status()
    items = response.json()['_embedded']['items']

    def download_item(item):
        file_name = item['name']
        download_url = item['file']
        if download_url:
            save_path = os.path.join(save_folder, file_name)
            download_file(download_url, save_path)
            print(f"Загружено {file_name} в {save_path}")
        else:
            print(f"Не удалось получить ссылку на загрузку для {file_name}")

    with ThreadPoolExecutor() as executor:
        executor.map(download_item, items)

# Пример использования:
public_link = 'https://disk.yandex.ru/d/ivp6ZHBjRJMU5Q'
save_folder = r'C:\Users\user\Downloads\FaceRecognition_upd1\dataset_video'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

download_from_yandex_disk(public_link, save_folder)
