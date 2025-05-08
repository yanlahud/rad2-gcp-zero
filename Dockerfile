# Usar uma imagem Python oficial como base
FROM python:3.10-slim

# Definir variáveis de ambiente
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
ENV PORT 8080 # Porta padrão que o Cloud Run espera

WORKDIR $APP_HOME

# Copiar o arquivo de dependências primeiro
COPY requirements.txt .

# Instalar as dependências
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copiar o código da aplicação
COPY ./app $APP_HOME/app

# Comando para executar a aplicação
# Cloud Run injeta a variável de ambiente PORT, que Uvicorn usará.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}

# test