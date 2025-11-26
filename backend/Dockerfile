# Use Python 3.9
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Run the app
# --host 0.0.0.0 is CRITICAL for Docker to accept outside connections
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]