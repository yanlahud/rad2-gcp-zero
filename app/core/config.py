# app/core/config.py
import os

# --- Seus valores reais (já preenchidos!) ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "rad2-459117")
GCS_UPLOAD_BUCKET = os.environ.get("GCS_UPLOAD_BUCKET", "rad2-459117-uploads")
GCS_OUTPUT_BUCKET = os.environ.get("GCS_OUTPUT_BUCKET", "rad2-459117-outputs")
# GCS_MODEL_BUCKET não é necessário por enquanto com a IA Simples
# GCS_MODEL_BUCKET = os.environ.get("GCS_MODEL_BUCKET", "rad2-459117-modelos")
# --- Fim dos seus valores ---

# Firestore Configuration
FIRESTORE_COLLECTION_EXAMS = "exams" # Nome da coleção para os exames