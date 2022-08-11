# syntax=docker/dockerfile:1
FROM python:3.9-slim

ARG BUILD_DATE

LABEL org.opencontainers.image.title="Caching for ML Models"
LABEL org.opencontainers.image.description="Caching for machine learning models."
LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.authors="6666331+schmidtbri@users.noreply.github.com"
LABEL org.opencontainers.image.source="https://github.com/schmidtbri/caching-for-ml-models"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.licenses="MIT License"
LABEL org.opencontainers.image.base.name="python:3.9-slim"

WORKDIR ./service

# installing git because we need to install the model package from the github repository
RUN apt-get update
RUN apt-get --assume-yes install git

COPY ./ml_model_caching ./ml_model_caching
COPY ./configuration ./configuration
COPY ./LICENSE ./LICENSE
COPY ./service_requirements.txt ./service_requirements.txt

RUN pip install -r service_requirements.txt

CMD ["uvicorn", "rest_model_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
