FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .

# Expose port 8000 (FastAPI default)
EXPOSE 8000

# Start the FastAPI server
# --host 0.0.0.0 means accept connections from outside the container
# same flag you use in your other containerized services
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]