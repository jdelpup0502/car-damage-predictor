FROM python:3.11-slim

# System deps for opencv-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY api_server.py hard_example_miner.py model_registry.json class_mapping.json ./
COPY car_damage_model.keras ./
COPY models_curriculum/final_curriculum_model.keras models_curriculum/class_mapping.json ./models_curriculum/

ENV ENV=production

CMD ["python", "api_server.py"]
