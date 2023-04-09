FROM python:3.10-slim

COPY . ./churn_modeling

WORKDIR /churn_modeling

RUN adduser -D appuser

USER appuser

RUN pip install --upgrade pip && \
    pip install -r requirements.txt --user

EXPOSE 8085

CMD ["streamlit", "run", "./src/frontend/Onboarding.py"]