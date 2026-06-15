# Use a lighweight python image
FROM python:3.12-slim

# Create working directory in the container
WORKDIR /app

# Shows the docker container where api.py is.
ENV PYTHONPATH=/app/src

# This lets the container know where the index is.
ENV INDEX_PATH=/app/storage/index

# Copy the requirements file into the containers working
# directory
COPY requirements.txt .

# Install dependencies in the image. This is done first because
# docker caches in layers. Our dependencies change less often than
# our code, so Docker can reuse the installed dependency layer 
# when we rebuild, making rebuilds quicker.  
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the files from our source code
# directory into /app/src in the container.
COPY src ./src

# Tell docker the app uses port 8000
EXPOSE 8000

# This command runs when the container starts
CMD ["uvicorn", "api:app", "--app-dir", "/app/src", "--host", "0.0.0.0", "--port", "8000"]