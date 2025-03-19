# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files to disk
ENV PYTHONUNBUFFERED=1

# Create and set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install system dependencies for torchaudio and libgomp
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port Flask will run on
EXPOSE 5000

# Run the Flask app when the container starts
CMD ["python", "app.py"]

# Use an official Python runtime as the base image
# FROM python:3.9-slim

# # Set environment variables to prevent Python from writing .pyc files to disc
# ENV PYTHONUNBUFFERED=1

# # Install system dependencies for torchaudio and libomp
# RUN apt-get update && apt-get install -y \
#     libsndfile1 \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# # Create and set the working directory
# WORKDIR /app

# # Copy the current directory contents into the container at /app
# COPY . /app/

# # Install Python dependencies
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# # Install Gunicorn for production
# RUN pip install gunicorn

# # Expose the port Flask will run on
# EXPOSE 5000

# # Use Gunicorn to run the Flask app in production mode
# CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
