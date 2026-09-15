FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for SQL Server ODBC, OpenCV, etc.
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    unixodbc-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]