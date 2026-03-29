# 1. Use a stable Python environment (3.11 is highly compatible with PyTorch)
FROM python:3.11-slim

# 2. Tell the cloud server where to put our files
WORKDIR /app

# 3. Copy our requirements first (this saves build time on Render)
COPY requirements.txt .

# 4. Install all the ML and SQL libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your actual code and models into the cloud server
COPY . .

# 6. Open the port that FastAPI needs to broadcast on
EXPOSE 8000

# 7. The exact command Render will run to start your intelligence engine
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}