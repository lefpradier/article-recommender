# backend/Dockerfile

FROM python:3.10.6

WORKDIR /APP

COPY frontend/requirements.txt .

COPY frontend/config.yaml .

COPY frontend/frontend.py .

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "frontend.py"]