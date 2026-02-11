FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server.py .

EXPOSE 8000

# Environment variables (override at runtime)
ENV DEVICE=cpu
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEFAULT_VOICE=./homesoul.wav
ENV TEMPERATURE=0.7
ENV LSD_DECODE_STEPS=1
ENV WORKERS=4

CMD ["python", "server.py"]
