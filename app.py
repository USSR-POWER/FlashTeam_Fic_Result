import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import time
import re
from nltk.corpus import stopwords
import nltk
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Инициализация библиотеки NLTK и загрузка русского стоп-листа
nltk.download('stopwords')
russian_stopwords = set(stopwords.words('russian'))

# Загрузка нашей моделей
print("Загрузка моделей...")
voting_ensemble = joblib.load("model/voting_ensemble_model.pkl")  # Модель ансамблевого голосования
scaler = joblib.load("model/scaler.pkl")  # Стандартный масштабировщик
label_encoder = joblib.load("model/label_encoder.pkl")  # Кодировщик меток
print("Модели успешно загружены.")

# Загрузка моделей SentenceTransformer для предобработки текста
print("Загрузка SentenceTransformer моделей...")
model_work_similarity = SentenceTransformer('deepvk/USER-bge-m3')  # Модель для сравнения текста о работе
model_semantic = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')  # Семантическая модель для навыков и позиции
print("SentenceTransformer модели успешно загружены.")

# Класс FastAPI
app = FastAPI()

# Функция для вычисления количества месяцев в диапазоне дат
def calculate_months(date_range):
    try:
        match = re.match(r'^(\d{4}-\d{2}-\d{2})\s*-\s*(\d{4}-\d{2}-\d{2})?$', date_range)
        if not match:
            return 0
        start_date = datetime.strptime(match.group(1), '%Y-%m-%d')
        end_date = datetime.strptime(match.group(2), '%Y-%m-%d') if match.group(2) else datetime.today()
        return max((end_date.year - start_date.year) * 12 + (end_date.month - start_date.month), 0)
    except Exception as e:
        print(f"Ошибка при расчете месяцев: {e}")
        return 0

# Предобработка текста: удаление лишних символов и стоп-слов
def preprocess_text(text):
    try:
        if not isinstance(text, str):
            return ''
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        words = text.split(',')
        words = [re.sub(r'\W+', '', word.strip()) for word in words]
        words = [word for word in words if word and word not in russian_stopwords]
        return ' '.join(words)
    except Exception as e:
        print(f"Ошибка при обработке текста: {e}")
        return ''

# Функция для вычисления сходства навыков между кандидатом и опытом работы
def calculate_skill_similarity(candidate_skills, work_experience):
    try:
        candidate_skills_set = set(candidate_skills.split(" "))
        work_experience_set = set(work_experience.split(" "))
        all_skills = list(candidate_skills_set.union(work_experience_set))
        candidate_vector = [1 if skill in candidate_skills_set else 0 for skill in all_skills]
        experience_vector = [1 if skill in work_experience_set else 0 for skill in all_skills]
        if not any(candidate_vector) or not any(experience_vector):
            return 0.0
        return cosine_similarity([candidate_vector], [experience_vector])[0][0]
    except Exception as e:
        print(f"Ошибка при расчете сходства навыков: {e}")
        return 0.0

# Функция для вычисления семантического сходства позиции и скилов
def calculate_semantic_similarity(position, key_skills, work_experience):
    try:
        combined_skills = key_skills + " " + work_experience
        position_embedding = model_semantic.encode([position])
        skills_embedding = model_semantic.encode([combined_skills])
        return cosine_similarity(position_embedding, skills_embedding)[0][0]
    except Exception as e:
        print(f"Ошибка при расчете семантического сходства: {e}")
        return 0.0

# Функция обработки входных данных
def process_input_data(input_data):
    results = []
    for i, row in enumerate(input_data):
        print(f"Обработка записи {i + 1}/{len(input_data)}...")
        try:
            work_experience = row.get("work_experience", "")
            position = preprocess_text(row.get("position", ""))
            key_skills = preprocess_text(row.get("key_skills", ""))

            total_months_worked = 0
            processed_date_ranges = set()  # Для уникальности диапазонов дат

            for line in work_experience.split('\n'):
                line = line.strip()
                if not line:
                    continue

                date_part = line.split(':')[0].strip()
                match = re.match(r'^(\d{4}-\d{2}-\d{2})\s*-\s*(\d{4}-\d{2}-\d{2})?$', date_part)
                if match:
                    date_range = match.group(0)
                    details = line.split(':', 1)[-1].strip()
                    comparison_text = ' '.join(details.split()[:4])

                    if date_range in processed_date_ranges:
                        continue
                    processed_date_ranges.add(date_range)

                    months_worked = calculate_months(date_range)

                    embeddings1 = model_work_similarity.encode(position, convert_to_tensor=True)
                    embeddings2 = model_work_similarity.encode(comparison_text, convert_to_tensor=True)
                    similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()

                    if similarity > 0.649:
                        total_months_worked += months_worked

            skill_similarity = calculate_skill_similarity(key_skills, preprocess_text(work_experience))
            semantic_similarity = calculate_semantic_similarity(position, key_skills, preprocess_text(work_experience))

            results.append({
                "age": float(row.get("age", 0)),
                "semantic_similarity": semantic_similarity,
                "skill_similarity": skill_similarity,
                "total_months_worked": total_months_worked
            })
        except Exception as e:
            print(f"Ошибка обработки записи: {e}")

    return pd.DataFrame(results)

# Эндпоинт FastAPI для обработки JSON-файла и отрпавки результатов работы модели
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    print("Получен запрос на предсказание.")
    if file.content_type != "application/json":
        print("Неверный тип файла.")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JSON file.")

    content = await file.read()
    try:
        raw_data = json.loads(content)
    except json.JSONDecodeError as e:
        print("Ошибка JSON:", e)
        raise HTTPException(status_code=400, detail="Invalid JSON format.")

    start_time = time.time()  # Время начала обработки

    print(f"Обработка данных для {len(raw_data)} записей...")
    processed_data = process_input_data(raw_data)
    features = ['age', 'semantic_similarity', 'skill_similarity', 'total_months_worked']
    scaled_data = pd.DataFrame(scaler.transform(processed_data[features]), columns=features)

    predictions_proba = voting_ensemble.predict_proba(scaled_data)[:, 1]
    predictions = voting_ensemble.predict(scaled_data)

    # Подсчет ROC-AUC, если есть истинные метки
    true_labels = [row.get("grade_proof") for row in raw_data if "grade_proof" in row]
    auc_roc = None
    if true_labels:
        try:
            true_labels_encoded = label_encoder.transform(true_labels)
            auc_roc = roc_auc_score(true_labels_encoded, predictions_proba)
        except Exception as e:
            print(f"Ошибка при расчете ROC-AUC: {e}")
            auc_roc = None

    # Создание списка index_mapping
    index_mapping = []

    # Добавление предсказаний и промежуточных метрик к данным
    for i, (row, prob, pred, intermediate) in enumerate(zip(raw_data, predictions_proba, predictions,
                                                            processed_data.to_dict(orient='records')), start=1):
        row["predicted_grade_proof"] = label_encoder.inverse_transform([pred])[0]
        row["probability"] = prob
        row.update(intermediate)

        # Формируем index_mapping
        index_mapping.append({
            "index": i,
            "predicted_grade_proof": label_encoder.inverse_transform([pred])[0]
        })

    end_time = time.time()  # Время окончания обработки
    total_time = end_time - start_time
    avg_time_per_record = total_time / len(raw_data)

    print("Предсказания завершены.")

    # Формирование итогового ответа
    response = {
        "data": raw_data,
        "metrics": {
            "roc_auc": auc_roc,
            "total_records": len(raw_data),
            "total_time_seconds": total_time,
            "average_time_per_record_seconds": avg_time_per_record
        },
        "index_mapping": index_mapping
    }

    return JSONResponse(content=response)

# Запуск FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
