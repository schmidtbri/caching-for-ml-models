# syntax=docker/dockerfile:1

FROM python:3.9-slim

WORKDIR ./service

RUN apt-get update
RUN apt-get --assume-yes install git

COPY ./data_enrichment ./data_enrichment
COPY ./configuration ./configuration
COPY ./LICENSE ./LICENSE
COPY ./service_requirements.txt ./service_requirements.txt

RUN pip install -r service_requirements.txt

CMD ["uvicorn", "rest_model_service.main:app", "--host", "0.0.0.0", "--port", "8000"]