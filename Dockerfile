FROM python:3.12-slim

# Working Directory
WORKDIR /app

# Copy source code to working directory
COPY .  /app/

# Install packages from requirements.txt

RUN pip install --no-cache-dir --upgrade pip &&\
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

CMD ["streamlit", "run", "app.py","--server.port=8501", "--server.address=0.0.0.0"]
