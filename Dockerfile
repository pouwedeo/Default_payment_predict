FROM python:3.12-slim

# Set working directory
WORKDIR /app



# Copy source code
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip &&\
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt &&\
    pip install streamlit && streamlit --version\
    pip uninstall -y lightgbm && pip install --no-cache-dir lightgbm

# Expose Streamlit port
EXPOSE 8501

# Set entrypoint and command
ENTRYPOINT ["streamlit"]
CMD ["run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
