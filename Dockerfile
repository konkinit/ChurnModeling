FROM python:3.8

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt 

WORKDIR /

EXPOSE 8501

CMD ["streamlit", "run", "./src/frontend/app.py"]
