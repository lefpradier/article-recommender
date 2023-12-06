# backend/Dockerfile

FROM python:3.10.6

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/frontend .

WORKDIR .

RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port=5000"]
