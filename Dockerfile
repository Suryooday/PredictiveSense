# Use python-slim as a lightweight base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for compiling certain python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files to the container
COPY . .

# Expose port 8501 (default Streamlit port)
EXPOSE 8501

# Healthcheck to verify container health
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit when the container launches
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
