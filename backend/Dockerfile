# Use a lightweight Python base image (matches your project)
FROM python:3.12-slim

# Set the working directory
WORKDIR /code

# Copy the requirements file and install dependencies
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the rest of your backend application code
COPY . /code

# Hugging Face Spaces run as a non-root user (UID 1000). 
# We need to give this user ownership of the folders so your app can read/write caches.
RUN useradd -m -u 1000 user
RUN chown -R user:user /code
USER user

# Hugging Face exposes port 7860 by default
EXPOSE 7860

# Start the FastAPI application via Uvicorn on port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]