# Use official Python slim image — same approach as your other services
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy and install dependencies first (Docker layer caching — you know this)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script and data
COPY train.py .
COPY train.csv .

ENV GIT_PYTHON_REFRESH=quiet

# When container runs, execute the training script
# After training, model.pkl will be saved — we mount a volume to get it out
CMD ["python", "train.py"]