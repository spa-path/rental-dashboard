# Use a lightweight official Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system-level packages required for Python builds (optional but safe)
RUN apt-get update && apt-get install -y build-essential

# Copy your entire project into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Prevent Streamlit telemetry prompts
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Set the port to match Render’s dynamic environment
ENV PORT 8501

# Expose the port for local testing or Render deployment
EXPOSE $PORT

# Start the Streamlit app from the app subdirectory
CMD ["streamlit", "run", "app/dashboard_app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
