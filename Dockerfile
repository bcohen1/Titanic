# syntax=docker/dockerfile:1.2

FROM python:3.9-slim-buster

WORKDIR /Docker

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY titanic.py .
COPY train.csv .
COPY test.csv .
COPY gender_submission.csv .