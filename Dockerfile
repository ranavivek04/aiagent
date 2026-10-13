# Use a lightweight official Python runtime base image
FROM python:3.12-slim

# Set system environment optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the functional workspace directory inside the container
WORKDIR /app

# Chroma's hnswlib dependency needs a C++ compiler when no wheel is available.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency tree manifest file into the image layer
COPY requirements.txt .

# Install the verified package matrix
RUN pip install --no-cache-dir -r requirements.txt

# Copy all local project script modules into the container image space
COPY . .

# Expose the API server gateway communication port
EXPOSE 8000

# Trigger the Uvicorn production server process on initialization
CMD ["python", "agentic-api.py"]
