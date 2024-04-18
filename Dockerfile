# Use the official lightweight Python image for Python 3.11
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt README.md ./

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest the app's code and directories
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
