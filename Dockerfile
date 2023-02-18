FROM python:3.8-slim

COPY . .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt 

WORKDIR /src/frontend

EXPOSE 8585

CMD ["streamlit", "run", "./app.py"]
