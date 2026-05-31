FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY app/requirements.txt .
RUN pip cache purge && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ .

# Expose the port
EXPOSE 7860

# Command to run the application
# Use the default Hugging Face port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]