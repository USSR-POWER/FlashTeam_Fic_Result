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

# Инициализация
nltk.download('stopwords')
russian_stopwords = set(stopwords.words('russian'))

# Загрузка обученных моделей и утилит
voting_ensemble = joblib.load("model/voting_ensemble_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# Загрузка SentenceTransformer моделей
model_work_similarity = SentenceTransformer('deepvk/USER-bge-m3')
model_semantic = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


# Функции обработки данных
def calculate_months(date_range):
    try:
        match = re.match(r'^(\d{4}-\d{2}-\d{2})\s*-\s*(\d{4}-\d{2}-\d{2})?$', date_range)
        if not match:
            return 0
        start_date = datetime.strptime(match.group(1), '%Y-%m-%d')
        end_date = datetime.strptime(match.group(2), '%Y-%m-%d') if match.group(2) else datetime.today()
        return max((end_date.year - start_date.year) * 12 + (end_date.month - start_date.month), 0)
    except:
        return 0


def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    words = text.split(',')
    words = [re.sub(r'\W+', '', word.strip()) for word in words]
    words = [word for word in words if word and word not in russian_stopwords]
    return ' '.join(words)


def calculate_skill_similarity(candidate_skills, work_experience):
    candidate_skills_set = set(candidate_skills.split(" "))
    work_experience_set = set(work_experience.split(" "))
    all_skills = list(candidate_skills_set.union(work_experience_set))
    candidate_vector = [1 if skill in candidate_skills_set else 0 for skill in all_skills]
    experience_vector = [1 if skill in work_experience_set else 0 for skill in all_skills]
    if not any(candidate_vector) or not any(experience_vector):
        return 0.0
    return cosine_similarity([candidate_vector], [experience_vector])[0][0]


def calculate_semantic_similarity(position, key_skills, work_experience):
    combined_skills = key_skills + " " + work_experience
    position_embedding = model_semantic.encode([position])
    skills_embedding = model_semantic.encode([combined_skills])
    return cosine_similarity(position_embedding, skills_embedding)[0][0]


def process_input_data(input_data):
    results = []
    for row in input_data:
        work_experience = row.get("work_experience", "")
        position = preprocess_text(row.get("position", ""))
        key_skills = preprocess_text(row.get("key_skills", ""))

        total_months_worked = 0
        processed_date_ranges = set()  # Для уникальности диапазонов дат

        # Проход по строкам опыта работы
        for line in work_experience.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Извлекаем диапазон дат и описание работы
            date_part = line.split(':')[0].strip()
            match = re.match(r'^(\d{4}-\d{2}-\d{2})\s*-\s*(\d{4}-\d{2}-\d{2})?$', date_part)
            if match:
                date_range = match.group(0)
                details = line.split(':', 1)[-1].strip()
                comparison_text = ' '.join(details.split()[:4])  # Используем первые 4 слова для сравнения

                # Если диапазон уже обработан, пропускаем
                if date_range in processed_date_ranges:
                    continue
                processed_date_ranges.add(date_range)

                # Подсчитываем месяцы работы
                months_worked = calculate_months(date_range)

                # Вычисляем сходство между позицией и описанием работы
                embeddings1 = model_work_similarity.encode(position, convert_to_tensor=True)
                embeddings2 = model_work_similarity.encode(comparison_text, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()

                # Учитываем только релевантные месяцы работы
                if similarity > 0.649:
                    total_months_worked += months_worked

        # Вычисление дополнительных метрик
        skill_similarity = calculate_skill_similarity(key_skills, preprocess_text(work_experience))
        semantic_similarity = calculate_semantic_similarity(position, key_skills, preprocess_text(work_experience))

        # Добавляем обработанные данные
        results.append({
            "age": float(row.get("age", 0)),
            "semantic_similarity": semantic_similarity,
            "skill_similarity": skill_similarity,
            "total_months_worked": total_months_worked
        })

    return pd.DataFrame(results)


# Основной процесс
def main(filepath, output_filepath, num_records=1000):
    start_time = time.time()

    # Загрузка исходных данных
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Берем только первые num_records записей
    raw_data = raw_data[:num_records]

    # Сохраняем истинные метки
    true_labels = [row["grade_proof"] for row in raw_data]

    # Обработка данных
    processed_data = process_input_data(raw_data)

    # Масштабирование
    features = ['age', 'semantic_similarity', 'skill_similarity', 'total_months_worked']
    scaled_data = pd.DataFrame(scaler.transform(processed_data[features]), columns=features)

    # Предсказание
    predictions_proba = voting_ensemble.predict_proba(scaled_data)[:, 1]
    predictions = voting_ensemble.predict(scaled_data)

    # Расчет метрики AUC-ROC
    auc_roc = roc_auc_score(label_encoder.transform(true_labels), predictions_proba)

    # Добавляем предсказания к данным
    for row, prob, pred, true_label in zip(raw_data, predictions_proba, predictions, true_labels):
        row["predicted_grade_proof"] = label_encoder.inverse_transform([pred])[0]
        row["probability"] = prob
        row["grade_proof"] = true_label  # Оригинальное значение

    # Сохранение результатов
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=4)

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_record = total_time / len(raw_data)

    print(f"ROC-AUC: {auc_roc}")
    print(f"Results saved to {output_filepath}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Number of records processed: {len(raw_data)}")
    print(f"Average time per record: {avg_time_per_record:.4f} seconds")


# Выполнение
input_filepath = "result_dataset_pinned.json"  # Путь к вашему большому JSON файлу
output_filepath = "output_results.json"  # Путь для сохранения результатов
main(input_filepath, output_filepath, num_records=100)
