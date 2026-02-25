FROM python:3.11-slim

WORKDIR /app

# Зависимости — отдельный слой для кеширования
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения
COPY popper /app/popper
COPY web_app.py /app/web_app.py

# Директории для данных и логов
RUN mkdir -p /app/data /app/.logs && chmod -R 777 /app/data /app/.logs

# Порт Gradio
EXPOSE 7860

CMD ["python", "web_app.py"]
