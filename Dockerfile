FROM python:3.10-slim

COPY . ./churn_modeling

RUN pip install --upgrade pip \
    && pip install -r requirements.txt 

WORKDIR /churn_modeling

EXPOSE 8085

CMD ["streamlit", "run", "./src/frontend/Onboarding.py"]

