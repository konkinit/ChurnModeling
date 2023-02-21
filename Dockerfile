FROM python:3.10-slim

COPY . .

RUN pip install --upgrade pip 
    && pip install -r requirements.txt 

WORKDIR .

EXPOSE 8085

CMD ["streamlit", "run", "./src/frontend/Onboarding.py"]

