FROM python:3.8

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt 

WORKDIR /telco_churn

EXPOSE 8501

CMD ["streamlit", "run", "./src/frontend/app.py"]
