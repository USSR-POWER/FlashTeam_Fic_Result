import requests
import json

# URL вашего API
url = "http://127.0.0.1:8001/predict/"

# Путь к файлу для отправки
file_path = "result_dataset_pinned.json"  # Укажите путь к вашему JSON файлу

# Путь для сохранения результатов
output_path = "output_results.json"

# Отправка файла через POST-запрос
with open(file_path, "rb") as file:
    files = {"file": ("test_input.json", file, "application/json")}
    response = requests.post(url, files=files)

# Проверка ответа
if response.status_code == 200:
    print("Успешный ответ от API!")
    response_data = response.json()

    # Сохранение ответа в файл
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(response_data, output_file, ensure_ascii=False, indent=4)

    print(f"Результаты сохранены в файл: {output_path}")
else:
    print(f"Ошибка: {response.status_code}")
    print(response.text)
