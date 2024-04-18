# Use the official lightweight Python image.
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and README into the working directory
COPY requirements.txt README.md ./

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code and directories
COPY . .

# The dataset and notebook directories are covered by the above command
# COPY dataset/ /app/dataset/
# COPY notebook/ /app/notebook/

# Expose the port Streamlit will run on
EXPOSE 8501

# The command to run your Streamlit app
CMD ["streamlit", "run", "app.py"]
