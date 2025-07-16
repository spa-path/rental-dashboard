# Use a lightweight official Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system-level packages required for Python builds (optional but safe)
RUN apt-get update && apt-get install -y build-essential

# Copy your entire project into the container
COPY . .

# Install Python dependencies from a file (you'll create this next)
RUN pip install --no-cache-dir -r requirements.txt

# Prevent Streamlit from showing telemetry prompts
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose the port Streamlit runs on
EXPOSE 8501

# Start the app
CMD ["streamlit", "run", "dashboard_app.py"]
