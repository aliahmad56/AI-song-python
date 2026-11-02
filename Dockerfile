# Use official Python image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg libsndfile1


COPY . .

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application file
COPY main.py .

# Expose FastAPI's default port
EXPOSE 8000

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
