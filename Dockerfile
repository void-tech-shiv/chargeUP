FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x scripts/start_production.sh
EXPOSE 8000
CMD ["sh", "scripts/start_production.sh"]
