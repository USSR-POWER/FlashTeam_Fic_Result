import streamlit as st
import requests
import json

# URL API
API_URL = "http://localhost:8001/predict/"

st.title("FlashTeam")
st.subheader("Загрузите JSON-файл для анализа")

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите JSON-файл", type=["json"])

if uploaded_file is not None:
    st.write("Файл успешно загружен!")

    try:
        file_content = uploaded_file.read().decode("utf-8")
        input_data = json.loads(file_content)

        # Проверка формата данных
        if not isinstance(input_data, list) or not all(
            isinstance(record, dict) and
            {"position", "age", "country", "city", "key_skills", "client_name",
             "grade_proof", "salary", "work_experience"}.issubset(record.keys()) and
            isinstance(record["grade_proof"], str)
            for record in input_data
        ):
            st.error("Ошибка: JSON-файл не соответствует ожидаемому формату!")
            st.stop()

        st.success(f"Файл загружен и успешно проверен. Количество записей: {len(input_data)}")

    except json.JSONDecodeError:
        st.error("Ошибка: Неверный формат JSON.")
        st.stop()

    # Отправка на сервер
    if st.button("Отправить на API"):
        st.info("Выполняем оценку грейдов сотрудников (одна запись обрабатывается ~1 сек...")

        # Преобразование в JSON
        json_data = json.dumps(input_data)
        response = requests.post(API_URL, files={"file": ("uploaded_file.json", json_data, "application/json")})

        if response.status_code == 200:
            st.success("Файл успешно обработан!")
            response_data = response.json()

            # Вывод метрик
            st.subheader("Метрики")
            metrics = response_data.get("metrics", {})
            st.write(f"**ROC-AUC:** {metrics.get('roc_auc', 'N/A')}")
            st.write(f"**Всего записей:** {metrics.get('total_records', 'N/A')}")
            st.write(f"**Общее время обработки (секунды):** {metrics.get('total_time_seconds', 'N/A')}")
            st.write(f"**Среднее время на запись (секунды):** {metrics.get('average_time_per_record_seconds', 'N/A')}")

            # Вывод кратко номеров строк и результатов
            st.subheader("Номер строки и Резлуьтат предсказания")
            index_mapping = response_data.get("index_mapping", [])
            if index_mapping:
                st.write("Количество строк:", len(index_mapping))
                st.dataframe(index_mapping)

                # Формирование JSON для загрузки
                json_result = json.dumps(index_mapping, ensure_ascii=False, indent=4)
                st.download_button(
                    label="Скачать результат в json",
                    data=json_result,
                    file_name="result_index_mapping.json",
                    mime="application/json"
                )
            else:
                st.warning("Индексный маппинг отсутствует в ответе.")
        else:
            st.error(f"Ошибка при обработке файла: {response.status_code}")
            st.write(response.text)
