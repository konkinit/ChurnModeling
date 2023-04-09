FROM python:3.10-slim

COPY . ./churn_modeling

WORKDIR /churn_modeling

ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

USER ${USERNAME}

RUN pip install --upgrade pip && \
    pip install -r requirements.txt --user

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "./src/frontend/Onboarding.py"]
